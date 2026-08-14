"""Test building a versioned cache from the staged STU_008 package."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from smart_attendance_ai.face_matcher import (
    FaceGalleryMatcher,
)

from smart_attendance_ai.gallery_schemas import (
    GalleryUpdatePolicy,
    GalleryVersionStatus,
)

from smart_attendance_ai.gallery_version_service import (
    GalleryVersionService,
)


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest().upper()


def student_embeddings(
    path: Path,
    student_id: str,
) -> np.ndarray:
    with np.load(
        path,
        allow_pickle=False,
    ) as gallery:
        embeddings = gallery[
            "image_embeddings"
        ].astype(np.float32)

        student_ids = gallery[
            "image_student_ids"
        ].astype(str)

    return embeddings[
        student_ids == student_id
    ]


def gallery_counts(
    path: Path,
):
    with np.load(
        path,
        allow_pickle=False,
    ) as gallery:
        students = gallery[
            "student_ids"
        ].astype(str).tolist()

        image_student_ids = gallery[
            "image_student_ids"
        ].astype(str)

    counts = {
        student_id: int(
            np.sum(
                image_student_ids
                == student_id
            )
        )
        for student_id in students
    }

    return (
        len(students),
        len(image_student_ids),
        counts,
    )


def main() -> None:
    print("=" * 70)
    print(
        "VERSIONED GALLERY CACHE TEST"
    )
    print("=" * 70)

    base_gallery_path = (
        ROOT
        / "data"
        / "generated"
        / "face_gallery_calibrated.npz"
    )

    enrollment_package_path = (
        ROOT
        / "data"
        / "enrollment_staging"
        / "ENROLLMENT_UPDATE_STU_008_SMOKE"
        / "enrollment_package.npz"
    )

    if not base_gallery_path.is_file():
        raise FileNotFoundError(
            "Base gallery not found: "
            f"{base_gallery_path}"
        )

    if not enrollment_package_path.is_file():
        raise FileNotFoundError(
            "Run "
            "scripts\\24_test_enrollment_service.py "
            "first. Missing: "
            f"{enrollment_package_path}"
        )

    version_id = (
        "GALLERY_STU008_REPLACE_"
        "SMOKE_V001"
    )

    version_directory = (
        ROOT
        / "data"
        / "generated"
        / "gallery_versions"
        / version_id
    )

    # This deletes only the previous test
    # version so the test can be repeated.
    if version_directory.exists():
        shutil.rmtree(
            version_directory
        )

    base_hash_before = sha256_file(
        base_gallery_path
    )

    (
        base_student_count,
        base_embedding_count,
        base_counts,
    ) = gallery_counts(
        base_gallery_path
    )

    other_students_before = {
        student_id: student_embeddings(
            base_gallery_path,
            student_id,
        )
        for student_id in base_counts
        if student_id != "STU_008"
    }

    service = GalleryVersionService(
        root=ROOT
    )

    result = (
        service.build_version_from_package(
            version_id=version_id,
            base_gallery_path=(
                base_gallery_path
            ),
            enrollment_package_path=(
                enrollment_package_path
            ),
            update_policy=(
                GalleryUpdatePolicy.REPLACE
            ),
            # Test 24 returned
            # READY_FOR_REVIEW.
            # This represents manual approval.
            approve_reviewed_package=True,
        )
    )

    base_hash_after = sha256_file(
        base_gallery_path
    )

    (
        new_student_count,
        new_embedding_count,
        new_counts,
    ) = gallery_counts(
        result.gallery_path
    )

    other_students_unchanged = all(
        np.array_equal(
            expected_embeddings,
            student_embeddings(
                result.gallery_path,
                student_id,
            ),
        )
        for student_id, expected_embeddings
        in other_students_before.items()
    )

    with np.load(
        enrollment_package_path,
        allow_pickle=False,
    ) as package:
        package_embeddings = package[
            "image_embeddings"
        ].astype(np.float32)

        package_template = package[
            "student_template"
        ].astype(np.float32)

    target_embeddings = student_embeddings(
        result.gallery_path,
        "STU_008",
    )

    matcher = FaceGalleryMatcher(
        result.gallery_path
    )

    identity = matcher.match(
        package_template
    )

    print(
        f"Version ID                : "
        f"{result.version_id}"
    )

    print(
        f"Status                    : "
        f"{result.status.value}"
    )

    print(
        f"Update policy             : "
        f"{result.update_policy.value}"
    )

    print(
        f"Base students             : "
        f"{base_student_count}"
    )

    print(
        f"New students              : "
        f"{new_student_count}"
    )

    print(
        f"Base embeddings           : "
        f"{base_embedding_count}"
    )

    print(
        f"New embeddings            : "
        f"{new_embedding_count}"
    )

    print(
        f"Old STU_008 embeddings    : "
        f"{base_counts['STU_008']}"
    )

    print(
        f"New STU_008 embeddings    : "
        f"{new_counts['STU_008']}"
    )

    print(
        "Removed target embeddings : "
        f"{result.removed_target_embeddings}"
    )

    print(
        "Added target embeddings   : "
        f"{result.added_target_embeddings}"
    )

    print(
        "Target template similarity: "
        f"{result.target_template_similarity:.6f}"
    )

    print(
        "Package identity in cache : "
        f"{identity.student_id} "
        f"({identity.score:.6f})"
    )

    print(
        "Other students unchanged  : "
        f"{other_students_unchanged}"
    )

    print(
        "Base gallery unchanged    : "
        f"{base_hash_before == base_hash_after}"
    )

    print(
        f"Version gallery           : "
        f"{result.gallery_path}"
    )

    print(
        f"Manifest                  : "
        f"{result.manifest_json}"
    )

    if result.warnings:
        print("Warnings:")

        for warning in result.warnings:
            print(
                f"  - {warning}"
            )

    assert result.status in {
        GalleryVersionStatus
        .READY_FOR_ACTIVATION,

        GalleryVersionStatus
        .READY_FOR_REVIEW,
    }

    assert result.gallery_path.is_file()
    assert result.manifest_json.is_file()

    assert (
        base_hash_before
        == base_hash_after
    )

    assert (
        base_student_count
        == new_student_count
    )

    assert (
        new_counts["STU_008"]
        == len(package_embeddings)
    )

    assert np.array_equal(
        target_embeddings,
        package_embeddings,
    )

    assert other_students_unchanged

    assert (
        identity.student_id
        == "STU_008"
    )

    assert identity.status == "MATCH"

    with result.manifest_json.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    assert (
        manifest["gallery_sha256"]
        == sha256_file(
            result.gallery_path
        )
    )

    assert (
        manifest[
            "base_gallery_sha256"
        ]
        == base_hash_before
    )

    assert manifest["activated"] is False

    print()
    print(
        "VERSIONED GALLERY CACHE "
        "TEST PASSED"
    )
    print(
        "The new version was built "
        "but NOT activated."
    )


if __name__ == "__main__":
    main()