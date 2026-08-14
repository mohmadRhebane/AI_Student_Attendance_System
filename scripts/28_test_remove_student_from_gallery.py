from pathlib import Path
import sys
import argparse

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from smart_attendance_ai.ai_facade import SmartAttendanceAI

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--student-id",
        required=True,
    )

    parser.add_argument(
        "--version-id",
        required=True,
    )

    args = parser.parse_args()

   # root = Path(__file__).resolve().parents[1]

    ai = SmartAttendanceAI(
        project_root=PROJECT_ROOT
    )

    try:
        active_before = ai.get_active_gallery_path()

        print(
            "ACTIVE BEFORE:",
            active_before,
        )

        with np.load(
            active_before,
            allow_pickle=False,
        ) as gallery:
            old_student_ids = (
                gallery["student_ids"]
                .astype(str)
                .tolist()
            )

            old_image_ids = (
                gallery["image_student_ids"]
                .astype(str)
                .tolist()
            )

        student_id = str(
            args.student_id
        )

        if student_id not in old_student_ids:
            raise RuntimeError(
                f"Student {student_id} is not "
                "present in active gallery."
            )

        old_embedding_count = (
            old_image_ids.count(student_id)
        )

        print(
            "OLD EMBEDDINGS:",
            old_embedding_count,
        )

        result = (
            ai.remove_student_from_gallery(
                student_id=student_id,
                version_id=args.version_id,
                activate_after_build=False,
            )
        )

        print(
            "NEW VERSION:",
            result.version_id,
        )

        print(
            "NEW GALLERY:",
            result.gallery_path,
        )

        print(
            "REMOVED EMBEDDINGS:",
            result.removed_embedding_count,
        )

        assert result.activated is False

        assert (
            result.removed_embedding_count
            == old_embedding_count
        )

        assert (
            result.new_student_count
            == result.base_student_count - 1
        )

        assert (
            result.new_embedding_count
            == result.base_embedding_count
            - result.removed_embedding_count
        )

        with np.load(
            result.gallery_path,
            allow_pickle=False,
        ) as new_gallery:

            new_student_ids = (
                new_gallery["student_ids"]
                .astype(str)
                .tolist()
            )

            new_image_ids = (
                new_gallery[
                    "image_student_ids"
                ]
                .astype(str)
                .tolist()
            )

        assert (
            student_id
            not in new_student_ids
        )

        assert (
            student_id
            not in new_image_ids
        )

        active_after = (
            ai.get_active_gallery_path()
        )

        assert (
            active_after.resolve()
            == active_before.resolve()
        )

        # Important:
        # Original active Gallery must remain unchanged.
        with np.load(
            active_before,
            allow_pickle=False,
        ) as old_gallery_again:

            original_ids = (
                old_gallery_again[
                    "student_ids"
                ]
                .astype(str)
                .tolist()
            )

        assert (
            student_id
            in original_ids
        )

        print()
        print(
            "REMOVE STUDENT TEST PASSED"
        )

        print(
            "Active Gallery was NOT changed."
        )

    finally:
        ai.shutdown()


if __name__ == "__main__":
    main()