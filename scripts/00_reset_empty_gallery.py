import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from smart_attendance_ai.ai_facade import SmartAttendanceAI


def main():

    print("=" * 70)
    print("EMPTY GALLERY BOOTSTRAP")
    print("=" * 70)

    ai = SmartAttendanceAI(
        project_root=ROOT
    )

    try:

        result = ai.bootstrap_empty_gallery(
            version_id="CLEAN_BOOTSTRAP_V001",
            activate_after_build=True,
        )

        print()
        print("Operation       :", result.operation)
        print("Version ID      :", result.version_id)
        print("Student count   :", result.student_count)
        print("Embedding count :", result.embedding_count)
        print("Gallery path    :", result.gallery_path)

        info = ai.get_active_gallery_info()

        print()
        print("Active source   :", info["source"])
        print("Active version  :", info["active_version_id"])
        print("Active gallery  :", info["gallery_path"])

        print()
        print("RESET SUCCESS")

    finally:
        ai.shutdown(wait=True)


if __name__ == "__main__":
    main()