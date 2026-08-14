"""Prepare student enrollment packages without modifying the active gallery."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

from smart_attendance_ai.arcface_recognizer import l2_normalize
from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.enrollment_schemas import (
    EnrollmentIdentityCheck,
    EnrollmentImageResult,
    EnrollmentMode,
    EnrollmentResult,
    EnrollmentStatus,
)
from smart_attendance_ai.face_alignment import (
    align_face,
    calculate_face_quality,
)
from smart_attendance_ai.face_matcher import FaceGalleryMatcher


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class EnrollmentValidationError(ValueError):
    """Raised when an enrollment package cannot be safely prepared."""


class EnrollmentService:
    """
    Validate images and produce a staged, versionable enrollment artifact.

    This service never edits the current face gallery. A later GalleryService
    consumes the generated package and builds a new gallery version atomically.
    """

    def __init__(
        self,
        root: Path,
        engine: Optional[AttendanceEngine] = None,
        staging_root: Optional[Path] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.engine = engine or AttendanceEngine(root=self.root)
        self.engine.load_models()

        self.staging_root = Path(
            staging_root
            or self.root / "data" / "enrollment_staging"
        ).resolve()
        self.staging_root.mkdir(parents=True, exist_ok=True)

    def prepare_enrollment(
        self,
        *,
        enrollment_id: str,
        student_id: str,
        full_name: str,
        image_paths: Iterable[Path],
        active_gallery_path: Path,
        mode: EnrollmentMode = EnrollmentMode.NEW,
        minimum_accepted_images: int = 3,
    ) -> EnrollmentResult:
        """Create one immutable staged package and return structured metadata."""
        enrollment_id = self._validate_identifier(
            enrollment_id,
            field_name="enrollment_id",
        )
        student_id = self._validate_identifier(
            student_id,
            field_name="student_id",
        )
        full_name = str(full_name).strip()
        if not full_name:
            raise EnrollmentValidationError("full_name cannot be empty.")

        if not isinstance(mode, EnrollmentMode):
            mode = EnrollmentMode(str(mode).upper())

        minimum_accepted_images = int(minimum_accepted_images)
        if minimum_accepted_images < 1:
            raise EnrollmentValidationError(
                "minimum_accepted_images must be at least 1."
            )

        if self.engine.session_active:
            raise RuntimeError(
                "Enrollment cannot run while an attendance session is active."
            )

        gallery_path = self._resolve_project_path(active_gallery_path)
        if not gallery_path.exists():
            raise FileNotFoundError(
                f"Active gallery does not exist: {gallery_path}"
            )

        matcher = self._create_matcher(gallery_path)
        student_exists = student_id in matcher.student_names
        if mode == EnrollmentMode.NEW and student_exists:
            raise EnrollmentValidationError(
                f"Student already exists in the active gallery: {student_id}"
            )
        if mode == EnrollmentMode.UPDATE and not student_exists:
            raise EnrollmentValidationError(
                f"Cannot update a student missing from the gallery: {student_id}"
            )
        if mode == EnrollmentMode.UPDATE:
            existing_name = matcher.student_names.get(student_id, "").strip()
            if existing_name and existing_name != full_name:
                raise EnrollmentValidationError(
                    "The supplied full_name does not match the active gallery: "
                    f"{existing_name!r}"
                )

        normalized_paths = self._normalize_image_paths(image_paths)
        if len(normalized_paths) < minimum_accepted_images:
            raise EnrollmentValidationError(
                "Not enough submitted images. "
                f"Required: {minimum_accepted_images}; "
                f"submitted: {len(normalized_paths)}."
            )

        final_directory = self.staging_root / enrollment_id
        if final_directory.exists():
            raise FileExistsError(
                "Enrollment ID already has a staged package: "
                f"{final_directory}"
            )

        temporary_directory = self.staging_root / (
            f".{enrollment_id}.tmp-{uuid.uuid4().hex}"
        )
        original_directory = temporary_directory / "original"
        aligned_directory = temporary_directory / "aligned"
        original_directory.mkdir(parents=True, exist_ok=False)
        aligned_directory.mkdir(parents=True, exist_ok=False)

        created_at = datetime.now(timezone.utc)

        try:
            processed = self._process_images(
                image_paths=normalized_paths,
                original_directory=original_directory,
                aligned_directory=aligned_directory,
            )
            accepted = [item for item in processed if item[0].accepted]

            if len(accepted) < minimum_accepted_images:
                reasons = [
                    f"{item.original_filename}: {item.reason}"
                    for item, _ in processed
                    if not item.accepted
                ]
                raise EnrollmentValidationError(
                    "Not enough images passed enrollment validation. "
                    f"Required: {minimum_accepted_images}; "
                    f"accepted: {len(accepted)}. "
                    + ("Rejected: " + " | ".join(reasons) if reasons else "")
                )

            aligned_faces = [face for _, face in accepted]
            self.engine.recognizer.warm_up()
            embeddings = self.engine.recognizer.embed_faces(aligned_faces)

            if embeddings.shape != (
                len(aligned_faces),
                self.engine.recognizer.embedding_size,
            ):
                raise RuntimeError(
                    f"Unexpected enrollment embedding shape: {embeddings.shape}"
                )

            norms = np.linalg.norm(embeddings, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-5):
                raise RuntimeError(
                    "Enrollment embeddings are not L2-normalized."
                )

            student_template = l2_normalize(
                embeddings.mean(axis=0, keepdims=True)
            )[0]

            identity_check = self._run_identity_check(
                matcher=matcher,
                student_template=student_template,
            )

            warnings = self._build_warnings(
                mode=mode,
                student_id=student_id,
                identity_check=identity_check,
                embeddings=embeddings,
            )
            status = (
                EnrollmentStatus.READY_FOR_REVIEW
                if warnings
                else EnrollmentStatus.READY
            )

            model_path = self._resolve_project_path(
                self.engine.config["models"]["recognizer"]["path"]
            )
            expected_model_hash = str(
                self.engine.config["models"]["recognizer"]["sha256"]
            ).upper()
            actual_model_hash = self._sha256_file(model_path).upper()
            if expected_model_hash != actual_model_hash:
                raise RuntimeError(
                    "Recognizer model SHA256 does not match ai.yaml."
                )

            package_npz = temporary_directory / "enrollment_package.npz"
            manifest_json = temporary_directory / "manifest.json"

            accepted_results = [item for item, _ in accepted]

            # Store final immutable paths in the returned result/manifest.
            for item in accepted_results:
                if item.stored_original_path is not None:
                    item.stored_original_path = (
                        final_directory
                        / item.stored_original_path.relative_to(
                            temporary_directory
                        )
                    )
                if item.stored_aligned_path is not None:
                    item.stored_aligned_path = (
                        final_directory
                        / item.stored_aligned_path.relative_to(
                            temporary_directory
                        )
                    )

            self._write_npz_atomic(
                path=package_npz,
                enrollment_id=enrollment_id,
                student_id=student_id,
                full_name=full_name,
                mode=mode,
                created_at=created_at,
                model_name=str(
                    self.engine.config["recognizer"]["model_name"]
                ),
                model_sha256=actual_model_hash,
                embeddings=embeddings,
                student_template=student_template,
                accepted_results=accepted_results,
                identity_check=identity_check,
            )

            result = EnrollmentResult(
                enrollment_id=enrollment_id,
                student_id=student_id,
                full_name=full_name,
                mode=mode,
                status=status,
                submitted_images=len(processed),
                accepted_images=len(accepted_results),
                rejected_images=len(processed) - len(accepted_results),
                package_directory=final_directory,
                package_npz=final_directory / package_npz.name,
                manifest_json=final_directory / manifest_json.name,
                image_results=[item for item, _ in processed],
                identity_check=identity_check,
                warnings=warnings,
                created_at=created_at,
            )

            self._write_json_atomic(manifest_json, result.to_dict())
            temporary_directory.replace(final_directory)
            return result

        except Exception:
            shutil.rmtree(temporary_directory, ignore_errors=True)
            raise

    def _process_images(
        self,
        *,
        image_paths: List[Path],
        original_directory: Path,
        aligned_directory: Path,
    ) -> list:
        detector_config = self.engine.config["detector"]
        quality_config = self.engine.config["video_recognition"]

        minimum_detection_score = float(
            detector_config["enrollment_min_score"]
        )
        minimum_face_size = float(detector_config["min_face_size"])
        minimum_blur = float(quality_config["min_blur_score"])
        minimum_brightness = float(quality_config["min_brightness"])
        maximum_brightness = float(quality_config["max_brightness"])

        results = []
        observed_hashes = set()

        for index, image_path in enumerate(image_paths, start=1):
            image_hash = self._sha256_file(image_path)
            item = EnrollmentImageResult(
                source_path=image_path,
                original_filename=image_path.name,
                accepted=False,
                image_sha256=image_hash,
            )

            if image_hash in observed_hashes:
                item.reason = "duplicate_image_content"
                results.append((item, None))
                continue
            observed_hashes.add(image_hash)

            image = self._load_image(image_path)
            if image is None:
                item.reason = "unreadable_image"
                results.append((item, None))
                continue

            detections = self.engine.detector.detect(image)
            eligible = [
                detection
                for detection in detections
                if detection.score >= minimum_detection_score
                and detection.width >= minimum_face_size
                and detection.height >= minimum_face_size
            ]

            if not eligible:
                item.reason = "no_eligible_face"
                results.append((item, None))
                continue
            if len(eligible) > 1:
                item.reason = "multiple_eligible_faces"
                results.append((item, None))
                continue

            detection = eligible[0]
            item.detection_score = round(float(detection.score), 6)
            item.face_width = round(float(detection.width), 3)
            item.face_height = round(float(detection.height), 3)

            try:
                aligned_face, _ = align_face(
                    image,
                    detection.landmarks,
                    output_size=112,
                )
                quality = calculate_face_quality(aligned_face)
            except Exception as exc:
                item.reason = f"alignment_failed:{type(exc).__name__}"
                results.append((item, None))
                continue

            item.blur_score = float(quality["blur_score"])
            item.brightness = float(quality["brightness"])
            item.contrast = float(quality["contrast"])
            item.dark_ratio = float(quality["dark_ratio"])
            item.bright_ratio = float(quality["bright_ratio"])

            if item.blur_score < minimum_blur:
                item.reason = "blur_below_threshold"
                results.append((item, None))
                continue
            if not minimum_brightness <= item.brightness <= maximum_brightness:
                item.reason = "brightness_outside_threshold"
                results.append((item, None))
                continue

            original_output = original_directory / f"{index:04d}_original.jpg"
            aligned_output = aligned_directory / f"{index:04d}_aligned.jpg"
            self._save_image(original_output, image)
            self._save_image(aligned_output, aligned_face)

            item.accepted = True
            item.reason = "accepted"
            item.stored_original_path = original_output
            item.stored_aligned_path = aligned_output
            results.append((item, aligned_face))

        return results

    def _run_identity_check(
        self,
        *,
        matcher: FaceGalleryMatcher,
        student_template: np.ndarray,
    ) -> EnrollmentIdentityCheck:
        result = matcher.match(student_template)
        candidates = [
            {
                "student_id": candidate.student_id,
                "full_name": candidate.full_name,
                "score": round(float(candidate.score), 6),
                "top_reference_images": list(
                    candidate.top_reference_images
                ),
            }
            for candidate in result.candidates
        ]
        return EnrollmentIdentityCheck(
            status=result.status,
            best_student_id=result.student_id,
            best_full_name=result.full_name,
            best_score=round(float(result.score), 6),
            second_score=round(float(result.second_score), 6),
            margin=round(float(result.margin), 6),
            top_candidates=candidates,
        )

    def _build_warnings(
        self,
        *,
        mode: EnrollmentMode,
        student_id: str,
        identity_check: EnrollmentIdentityCheck,
        embeddings: np.ndarray,
    ) -> List[str]:
        warnings = []
        if len(embeddings) > 1:
            similarity = embeddings @ embeddings.T
            pair_values = similarity[
                np.triu_indices(len(embeddings), k=1)
            ]
            if pair_values.size:
                minimum = float(pair_values.min())
                if minimum < float(self.engine.config["matcher"]["min_score"]):
                    warnings.append(
                        "Low similarity exists between submitted images; "
                        f"minimum pair similarity={minimum:.6f}."
                    )

        if mode == EnrollmentMode.UPDATE:
            if identity_check.status == "MATCH" and (
                identity_check.best_student_id != student_id
            ):
                warnings.append(
                    "Update images strongly match a different gallery student: "
                    f"{identity_check.best_student_id}."
                )
            elif identity_check.best_student_id != student_id:
                warnings.append(
                    "Update identity check did not rank the expected student first."
                )
        elif identity_check.status == "MATCH":
            warnings.append(
                "New-student images match an existing gallery identity and "
                "require manual duplicate review: "
                f"{identity_check.best_student_id}."
            )

        return warnings

    def _create_matcher(self, gallery_path: Path) -> FaceGalleryMatcher:
        matcher_config = self.engine.config["matcher"]
        return FaceGalleryMatcher(
            gallery_path=gallery_path,
            top_k=matcher_config["top_k"],
            min_score=matcher_config["min_score"],
            min_margin=matcher_config["min_margin"],
            max_candidates=matcher_config["max_candidates"],
        )

    def _normalize_image_paths(
        self,
        image_paths: Iterable[Path],
    ) -> List[Path]:
        normalized = []
        seen_paths = set()
        for raw_path in image_paths:
            path = self._resolve_project_path(raw_path)
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.exists() or not path.is_file():
                raise FileNotFoundError(f"Enrollment image not found: {path}")
            if path.suffix.lower() not in _ALLOWED_IMAGE_EXTENSIONS:
                raise EnrollmentValidationError(
                    f"Unsupported enrollment image extension: {path.suffix}"
                )
            normalized.append(path)

        if not normalized:
            raise EnrollmentValidationError("No enrollment images supplied.")
        return normalized

    def _resolve_project_path(self, value: Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    @staticmethod
    def _validate_identifier(value: str, *, field_name: str) -> str:
        text = str(value).strip()
        if not _SAFE_IDENTIFIER.fullmatch(text):
            raise EnrollmentValidationError(
                f"{field_name} must contain 1-64 letters, numbers, '_' or '-'."
            )
        return text

    @staticmethod
    def _load_image(path: Path):
        try:
            encoded = np.fromfile(str(path), dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def _save_image(path: Path, image: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not success:
            raise RuntimeError(f"Failed to encode image: {path}")
        encoded.tofile(str(path))

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest().upper()

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        temporary.replace(path)

    @staticmethod
    def _write_npz_atomic(
        *,
        path: Path,
        enrollment_id: str,
        student_id: str,
        full_name: str,
        mode: EnrollmentMode,
        created_at: datetime,
        model_name: str,
        model_sha256: str,
        embeddings: np.ndarray,
        student_template: np.ndarray,
        accepted_results: List[EnrollmentImageResult],
        identity_check: EnrollmentIdentityCheck,
    ) -> None:
        temporary = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray("1.0"),
            enrollment_id=np.asarray(enrollment_id),
            enrollment_mode=np.asarray(mode.value),
            student_id=np.asarray(student_id),
            full_name=np.asarray(full_name),
            created_at_utc=np.asarray(created_at.isoformat()),
            model_name=np.asarray(model_name),
            model_sha256=np.asarray(model_sha256),
            embedding_dimension=np.asarray(
                embeddings.shape[1], dtype=np.int32
            ),
            image_embeddings=embeddings.astype(np.float32),
            student_template=student_template.astype(np.float32),
            original_filenames=np.asarray(
                [item.original_filename for item in accepted_results],
                dtype=np.str_,
            ),
            stored_original_filenames=np.asarray(
                [
                    item.stored_original_path.name
                    for item in accepted_results
                    if item.stored_original_path
                ],
                dtype=np.str_,
            ),
            stored_aligned_filenames=np.asarray(
                [
                    item.stored_aligned_path.name
                    for item in accepted_results
                    if item.stored_aligned_path
                ],
                dtype=np.str_,
            ),
            image_sha256=np.asarray(
                [item.image_sha256 for item in accepted_results],
                dtype=np.str_,
            ),
            detection_scores=np.asarray(
                [item.detection_score for item in accepted_results],
                dtype=np.float32,
            ),
            blur_scores=np.asarray(
                [item.blur_score for item in accepted_results],
                dtype=np.float32,
            ),
            brightness=np.asarray(
                [item.brightness for item in accepted_results],
                dtype=np.float32,
            ),
            identity_check_status=np.asarray(identity_check.status),
            identity_check_student_id=np.asarray(
                identity_check.best_student_id
            ),
            identity_check_score=np.asarray(
                identity_check.best_score, dtype=np.float32
            ),
            identity_check_margin=np.asarray(
                identity_check.margin, dtype=np.float32
            ),
        )
        temporary.replace(path)