"""Verify non-blocking session execution and cooperative stop."""

import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_schemas import (
    OutputOptions,
    SessionConfig,
)
from smart_attendance_ai.session_manager import (
    AttendanceSessionManager,
)


def build_session(session_id: str) -> SessionConfig:
    return SessionConfig(
        session_id=session_id,
        class_id="CLASS_A",
        course_id="COURSE_101",
        source="data/raw/videos/development.mp4",
        gallery_path=(
            "data/generated/face_gallery_calibrated.npz"
        ),
        roster_student_ids=[
            f"STU_{number:03d}" for number in range(1, 16)
        ],
        output=OutputOptions(
            save_csv=False,
            save_events=False,
            save_summary=False,
            save_video=False,
        ),
    )


def wait_until_frames(
    manager: AttendanceSessionManager,
    session_id: str,
    minimum_frames: int,
    timeout: float,
) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        snapshot = manager.get_session(session_id)
        if snapshot.frames_processed >= minimum_frames:
            return
        if snapshot.status.value in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }:
            return
        time.sleep(0.05)
    raise TimeoutError(
        f"Session did not reach {minimum_frames} frames."
    )


def main() -> None:
    print("=" * 70)
    print("BACKGROUND SESSION MANAGER TEST")
    print("=" * 70)

    manager = AttendanceSessionManager(root=ROOT)

    try:
        completed_id = "MANAGER_COMPLETE_TEST_001"
        created = manager.create_session(
            build_session(completed_id),
            max_frames=180,
        )
        print(f"Created status       : {created.status.value}")

        started_at = time.perf_counter()
        started = manager.start_session(completed_id)
        start_call_seconds = time.perf_counter() - started_at

        print(f"Start returned status: {started.status.value}")
        print(f"Start call seconds   : {start_call_seconds:.4f}")

        assert start_call_seconds < 1.0
        assert started.status.value == "STARTING"

        last_printed_bucket = -1
        while True:
            snapshot = manager.get_session(completed_id)
            bucket = snapshot.frames_processed // 30
            if bucket != last_printed_bucket:
                last_printed_bucket = bucket
                print(
                    f"Status={snapshot.status.value:9s} | "
                    f"frames={snapshot.frames_processed:3d} | "
                    f"present={snapshot.present_count:2d}"
                )

            if snapshot.status.value in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
            }:
                break
            time.sleep(0.10)

        completed = manager.wait_for_terminal(
            completed_id,
            timeout=120.0,
        )
        result = manager.get_result(completed_id)

        assert completed.status.value == "COMPLETED"
        assert completed.frames_processed == 180
        assert completed.result_available
        assert result is not None
        assert result.metrics.frames_processed == 180
        assert manager.active_session_id is None

        print()
        print("Completed-session assertions: PASSED")

        cancelled_id = "MANAGER_CANCEL_TEST_001"
        manager.create_session(build_session(cancelled_id))
        manager.start_session(cancelled_id)
        wait_until_frames(
            manager,
            cancelled_id,
            minimum_frames=30,
            timeout=60.0,
        )

        stopping = manager.stop_session(cancelled_id)
        print(
            f"Stop requested at     : "
            f"{stopping.frames_processed} frames"
        )

        cancelled = manager.wait_for_terminal(
            cancelled_id,
            timeout=60.0,
        )
        cancelled_result = manager.get_result(cancelled_id)

        print(f"Cancelled status     : {cancelled.status.value}")
        print(f"Cancelled frames     : {cancelled.frames_processed}")

        assert cancelled.status.value == "CANCELLED"
        assert cancelled.stop_requested
        assert cancelled_result is not None
        assert cancelled_result.status.value == "CANCELLED"
        assert manager.active_session_id is None

        print("Cancellation assertions     : PASSED")
        print(f"Models still loaded         : {manager.service.engine.models_loaded}")
        assert manager.service.engine.models_loaded

        print()
        print("BACKGROUND SESSION MANAGER TEST PASSED")

    finally:
        manager.shutdown(wait=True)


if __name__ == "__main__":
    main()