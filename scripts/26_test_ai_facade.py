"""Integration test for the stable SmartAttendanceAI facade."""

import hashlib
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.ai_facade import SmartAttendanceAI


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> None:
    print("=" * 70)
    print("AI FACADE INTEGRATION TEST")
    print("=" * 70)

    enrollment_id = "AI_FACADE_UPDATE_STU_008_SMOKE"
    version_id = "AI_FACADE_GALLERY_STU008_SMOKE_V001"

    staging_directory = (
        ROOT / "data" / "enrollment_staging" / enrollment_id
    )
    version_directory = (
        ROOT / "data" / "generated" / "gallery_versions" / version_id
    )

    # Repeatable smoke-test cleanup. It only removes this test's artifacts.
    if staging_directory.exists():
        shutil.rmtree(staging_directory)
    if version_directory.exists():
        shutil.rmtree(version_directory)

    image_paths = sorted(
        (ROOT / "data" / "raw" / "student_images").glob("STU_008_*.jpg")
    )
    if len(image_paths) < 3:
        raise RuntimeError(
            "Expected at least three STU_008 images in data/raw/student_images."
        )

    ai = SmartAttendanceAI(project_root=ROOT)

    try:
        health = ai.get_health()
        active_before = ai.get_active_gallery_info()
        active_path = Path(active_before["gallery_path"])
        active_hash_before = sha256_file(active_path)

        print(f"Facade version       : {health['facade_version']}")
        print(f"Runtime status       : {health['status']}")
        print(f"Active gallery       : {active_before['active_version_id']}")
        print(f"Submitted images     : {len(image_paths)}")

        enrollment = ai.prepare_student_enrollment(
            enrollment_id=enrollment_id,
            student_id="STU_008",
            full_name="يحيى ساعور",
            image_paths=image_paths,
            mode="UPDATE",
            minimum_accepted_images=3,
        )

        print(f"Enrollment status    : {enrollment.status}")
        print(f"Accepted images      : {enrollment.accepted_images}")
        print(f"Rejected images      : {enrollment.rejected_images}")
        print(f"Embedding shape      : {enrollment.embeddings.shape}")
        print(f"Template shape       : {enrollment.student_template.shape}")
        print(
            "Identity best match  : "
            f"{enrollment.identity_check['best_student_id']} "
            f"({enrollment.identity_check['best_score']})"
        )

        assert enrollment.ready
        assert enrollment.accepted_images == len(enrollment.embeddings)
        assert enrollment.embeddings.shape[1] == 512
        assert enrollment.student_template.shape == (512,)
        assert enrollment.identity_check["best_student_id"] == "STU_008"

        db_records = list(enrollment.iter_embedding_records())
        assert len(db_records) == enrollment.accepted_images
        assert all(record["embedding"].shape == (512,) for record in db_records)
        assert "embeddings" not in enrollment.to_dict(include_vectors=False)
        assert len(enrollment.to_dict(include_vectors=True)["embeddings"]) == (
            enrollment.accepted_images
        )

        active_hash_after_enrollment = sha256_file(active_path)
        assert active_hash_after_enrollment == active_hash_before

        gallery = ai.build_gallery_version(
            enrollment=enrollment,
            version_id=version_id,
            update_policy="REPLACE",
            approve_reviewed_package=True,
            activate_after_build=False,
        )

        active_after_build = ai.get_active_gallery_info()
        print(f"Gallery status       : {gallery.status}")
        print(f"Gallery version      : {gallery.version_id}")
        print(f"New cache path       : {gallery.gallery_path}")
        print(f"Cache activated      : {gallery.activated}")
        print(
            "Active gallery same  : "
            f"{active_before['gallery_sha256'] == active_after_build['gallery_sha256']}"
        )

        assert gallery.gallery_path.is_file()
        assert gallery.manifest_json.is_file()
        assert gallery.activated is False
        assert (
            active_before["gallery_sha256"]
            == active_after_build["gallery_sha256"]
        )

        roster = [f"STU_{index:03d}" for index in range(1, 16)]
        start = ai.start_attendance(
            session_id="AI_FACADE_ATTENDANCE_SMOKE",
            class_id="CLASS_A",
            course_id="COURSE_101",
            source="data/raw/videos/development.mp4",
            roster_student_ids=roster,
            max_frames=60,
        )
        print(f"Attendance start     : {start['status']}")

        terminal = ai.wait_for_attendance(
            "AI_FACADE_ATTENDANCE_SMOKE",
            timeout=180,
        )
        attendance_result = ai.get_attendance_result(
            "AI_FACADE_ATTENDANCE_SMOKE"
        )

        print(f"Attendance terminal  : {terminal['status']}")
        print(f"Frames processed     : {terminal['frames_processed']}")
        print(
            "Models still loaded  : "
            f"{ai.get_health()['runtime']['models_loaded']}"
        )

        assert terminal["status"] == "COMPLETED"
        assert terminal["frames_processed"] == 60
        assert attendance_result is not None
        assert len(attendance_result["attendance"]) == 15
        assert ai.get_health()["runtime"]["models_loaded"] is True

        print()
        print("AI FACADE INTEGRATION TEST PASSED")
        print("The test gallery version was NOT activated.")

    finally:
        ai.shutdown(wait=True)


if __name__ == "__main__":
    main()