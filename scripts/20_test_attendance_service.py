"""Verify that the service invokes AttendanceEngine internally."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_schemas import OutputOptions, SessionConfig
from smart_attendance_ai.attendance_service import AttendanceService


def main() -> None:
    print("=" * 70)
    print("INTERNAL ATTENDANCE SERVICE TEST")
    print("=" * 70)

    service = AttendanceService(root=ROOT)

    session = SessionConfig(
        session_id="SERVICE_INTERNAL_TEST_001",
        class_id="CLASS_A",
        course_id="COURSE_101",
        source="data/raw/videos/development.mp4",
        gallery_path="data/generated/face_gallery_calibrated.npz",
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

    def on_frame(frame_result) -> None:
        if frame_result.newly_confirmed_student_ids:
            print(
                f"Frame {frame_result.frame_index:4d} confirmed: "
                + ", ".join(
                    frame_result.newly_confirmed_student_ids
                )
            )

    print("Calling AttendanceService.run_session() directly...")
    result = service.run_session(
        session,
        max_frames=180,
        on_frame=on_frame,
    )

    print()
    print(f"Status           : {result.status.value}")
    print(f"Frames processed : {result.metrics.frames_processed}")
    print(f"Present students : {result.present_student_ids}")
    print(f"Absent students  : {result.absent_student_ids}")
    print(f"Events           : {len(result.events)}")
    print(f"Service running  : {service.is_running}")
    print(f"Models loaded    : {service.engine.models_loaded}")

    assert result.status.value == "COMPLETED"
    assert result.metrics.frames_processed == 180
    assert len(result.attendance) == 15
    assert not service.is_running
    assert service.engine.models_loaded
    assert service.engine.session_config is None

    print()
    print("INTERNAL ATTENDANCE SERVICE TEST PASSED")


if __name__ == "__main__":
    main()