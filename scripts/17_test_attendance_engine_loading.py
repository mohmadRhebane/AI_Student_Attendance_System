import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from smart_attendance_ai.attendance_engine import (
    AttendanceEngine,
)
from smart_attendance_ai.attendance_schemas import (
    OutputOptions,
    SessionConfig,
)


def main():
    print("=" * 70)
    print("ATTENDANCE ENGINE LOADING TEST")
    print("=" * 70)

    session = SessionConfig(
        session_id="DEV_DYNAMIC_001",
        class_id="CLASS_A",
        course_id="COURSE_101",
        source=(
            "data/raw/videos/"
            "development.mp4"
        ),
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

    engine = AttendanceEngine(
        root=ROOT
    )

    print("Loading AI models...")
    engine.load_models()

    detector_object_id = id(
        engine.detector
    )

    recognizer_object_id = id(
        engine.recognizer
    )

    # Calling load_models again must not create new models.
    engine.load_models()

    assert id(engine.detector) == detector_object_id
    assert id(engine.recognizer) == recognizer_object_id

    print("Starting dynamic session...")
    engine.start_session(session)

    runtime_info = engine.get_runtime_info()

    print()
    print(
        f"Models loaded          : "
        f"{runtime_info['models_loaded']}"
    )

    print(
        f"Session active         : "
        f"{runtime_info['session_active']}"
    )

    print(
        f"Ready for frames       : "
        f"{runtime_info['ready_for_frames']}"
    )

    print(
        f"Session ID             : "
        f"{runtime_info['session_id']}"
    )

    print(
        f"Class ID               : "
        f"{runtime_info['class_id']}"
    )

    print(
        f"Roster students        : "
        f"{runtime_info['roster_count']}"
    )

    print(
        f"Gallery students       : "
        f"{runtime_info['gallery_student_count']}"
    )

    print(
        f"Detector providers     : "
        f"{runtime_info['detector_providers']}"
    )

    print(
        f"Recognizer providers   : "
        f"{runtime_info['recognizer_providers']}"
    )

    assert engine.models_loaded
    assert engine.session_active
    assert engine.ready_for_frames

    assert runtime_info[
        "roster_count"
    ] == 15

    assert runtime_info[
        "gallery_student_count"
    ] >= 15

    assert (
        runtime_info[
            "detector_providers"
        ][0]
        == "CUDAExecutionProvider"
    )

    assert (
        runtime_info[
            "recognizer_providers"
        ][0]
        == "CUDAExecutionProvider"
    )

    print()
    print(
        "ATTENDANCE ENGINE LOADING TEST PASSED"
    )


if __name__ == "__main__":
    main()