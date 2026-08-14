import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from smart_attendance_ai.attendance_schemas import (
    OutputOptions,
    SessionConfig,
)


def main():
    session = SessionConfig(
        session_id="DEV_DYNAMIC_001",
        class_id="CLASS_A",
        course_id="COURSE_101",
        source="data/raw/videos/development.mp4",
        gallery_path=(
            "data/generated/"
            "face_gallery_calibrated.npz"
        ),
        roster_student_ids=[
            f"STU_{number:03d}"
            for number in range(1, 16)
        ],
        output=OutputOptions(
            save_csv=True,
            save_events=True,
            save_summary=True,
            save_video=False,
        ),
    )

    print("=" * 70)
    print("ATTENDANCE SCHEMAS TEST")
    print("=" * 70)
    print(f"Session ID    : {session.session_id}")
    print(f"Class ID      : {session.class_id}")
    print(f"Course ID     : {session.course_id}")
    print(f"Source        : {session.source}")
    print(f"Gallery       : {session.gallery_path}")
    print(f"Roster count  : {len(session.roster_student_ids)}")
    print(f"Save video    : {session.output.save_video}")
    print(f"Output prefix : {session.output.output_prefix}")
    print()
    print("ATTENDANCE SCHEMAS TEST PASSED")


if __name__ == "__main__":
    main()