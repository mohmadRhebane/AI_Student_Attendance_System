import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.attendance_schemas import OutputOptions, SessionConfig


MAX_FRAMES = 180


def main():
    session = SessionConfig(
        session_id="DEV_PROCESS_FRAME_001",
        class_id="CLASS_A",
        course_id="COURSE_101",
        source="data/raw/videos/development.mp4",
        gallery_path="data/generated/face_gallery_calibrated.npz",
        roster_student_ids=[
            f"STU_{number:03d}" for number in range(1, 16)
        ],
        output=OutputOptions(save_video=False),
    )

    engine = AttendanceEngine(root=ROOT)
    engine.start_session(session)

    source_path = ROOT / str(session.source)
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        capture.release()
        raise RuntimeError("Video FPS must be greater than zero.")

    print("=" * 70)
    print("PROCESS FRAME TEST")
    print("=" * 70)
    print(f"Source          : {source_path}")
    print(f"Maximum frames  : {MAX_FRAMES}")
    print(f"FPS             : {fps:.3f}")
    print("Warming up GPU models...")

    success, warmup_frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError("Cannot read warm-up frame.")

    engine.warm_up(warmup_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    try:
        while frame_index < MAX_FRAMES:
            success, frame = capture.read()
            if not success:
                break

            result = engine.process_frame(
                frame=frame,
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
            )

            if result.newly_confirmed_student_ids:
                print(
                    f"Frame {frame_index:4d} | confirmed: "
                    + ", ".join(result.newly_confirmed_student_ids)
                )

            if frame_index % 30 == 0:
                print(
                    f"Frame {frame_index:4d} | "
                    f"detected={result.detected_faces:2d} | "
                    f"tracks={result.active_tracks:2d} | "
                    f"recognized={result.recognized_faces:2d} | "
                    f"present={len(engine.attendance):2d}"
                )

            frame_index += 1
    finally:
        capture.release()

    info = engine.get_runtime_info()

    assert frame_index > 0
    assert info["frames_processed"] == frame_index
    assert info["detection_calls"] > 0
    assert info["models_warmed_up"] is True
    assert info["session_active"] is True

    print()
    print("=" * 70)
    print("PROCESS FRAME SUMMARY")
    print("=" * 70)
    print(f"Frames processed   : {info['frames_processed']}")
    print(f"Detection calls    : {info['detection_calls']}")
    print(f"Recognition calls  : {info['recognition_calls']}")
    print(f"Recognition faces  : {info['recognition_faces']}")
    print(f"Alignment failures : {info['alignment_failures']}")
    print(f"Present count      : {info['present_count']}")
    print(f"Events             : {len(engine.events)}")
    print(f"Present students   : {sorted(engine.attendance.keys())}")
    print()
    print("PROCESS FRAME TEST PASSED")


if __name__ == "__main__":
    main()