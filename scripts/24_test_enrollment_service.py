"""Smoke test for safe staged enrollment using existing STU_008 images."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.enrollment_schemas import (
    EnrollmentMode,
    EnrollmentStatus,
)
from smart_attendance_ai.enrollment_service import EnrollmentService


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def gallery_student_name(gallery_path: Path, student_id: str) -> str:
    with np.load(gallery_path, allow_pickle=False) as gallery:
        student_ids = gallery["student_ids"].astype(str).tolist()
        student_names = gallery["student_names"].astype(str).tolist()
    return dict(zip(student_ids, student_names))[student_id]


def main() -> None:
    print("=" * 70)
    print("SAFE ENROLLMENT STAGING TEST")
    print("=" * 70)

    gallery_path = (
        ROOT / "data" / "generated" / "face_gallery_calibrated.npz"
    )
    images_directory = ROOT / "data" / "raw" / "student_images"
    image_paths = sorted(images_directory.glob("STU_008_*.jpg"))

    if len(image_paths) < 3:
        raise RuntimeError(
            "The test requires at least three STU_008 enrollment images."
        )

    enrollment_id = "ENROLLMENT_UPDATE_STU_008_SMOKE"
    package_directory = (
        ROOT / "data" / "enrollment_staging" / enrollment_id
    )
    if package_directory.exists():
        shutil.rmtree(package_directory)

    gallery_hash_before = sha256_file(gallery_path)
    full_name = gallery_student_name(gallery_path, "STU_008")

    engine = AttendanceEngine(root=ROOT)
    service = EnrollmentService(root=ROOT, engine=engine)

    result = service.prepare_enrollment(
        enrollment_id=enrollment_id,
        student_id="STU_008",
        full_name=full_name,
        image_paths=image_paths,
        active_gallery_path=gallery_path,
        mode=EnrollmentMode.UPDATE,
        minimum_accepted_images=3,
    )

    gallery_hash_after = sha256_file(gallery_path)

    print(f"Student ID          : {result.student_id}")
    print(f"Full name           : {result.full_name}")
    print(f"Mode                : {result.mode.value}")
    print(f"Status              : {result.status.value}")
    print(f"Submitted images    : {result.submitted_images}")
    print(f"Accepted images     : {result.accepted_images}")
    print(f"Rejected images     : {result.rejected_images}")
    print(
        "Identity best match : "
        f"{result.identity_check.best_student_id} "
        f"({result.identity_check.best_score:.6f})"
    )
    print(f"Package directory   : {result.package_directory}")
    print(f"Active gallery same : {gallery_hash_before == gallery_hash_after}")

    assert result.status in {
        EnrollmentStatus.READY,
        EnrollmentStatus.READY_FOR_REVIEW,
    }
    assert result.accepted_images >= 3
    assert result.identity_check.best_student_id == "STU_008"
    assert result.package_npz.exists()
    assert result.manifest_json.exists()
    assert gallery_hash_before == gallery_hash_after

    with result.manifest_json.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    assert manifest["student_id"] == "STU_008"
    assert manifest["accepted_images"] == result.accepted_images

    with np.load(result.package_npz, allow_pickle=False) as package:
        embeddings = package["image_embeddings"]
        template = package["student_template"]
        assert embeddings.shape[0] == result.accepted_images
        assert embeddings.shape[1] == 512
        assert template.shape == (512,)
        assert np.allclose(
            np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5
        )
        assert np.isclose(np.linalg.norm(template), 1.0, atol=1e-5)

    print()
    print("SAFE ENROLLMENT STAGING TEST PASSED")


if __name__ == "__main__":
    main()