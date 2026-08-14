"""Build immutable NPZ gallery-cache versions from enrollment packages."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from smart_attendance_ai.face_matcher import FaceGalleryMatcher
from smart_attendance_ai.gallery_schemas import (
    GalleryRebuildResult,
    GalleryStudentRemovalResult,
    GalleryUpdatePolicy,
    GalleryVersionResult,
    GalleryVersionStatus,
)


_SAFE_ID = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$"
)


class GalleryVersionValidationError(ValueError):
    """Raised when a gallery version cannot be built safely."""


class GalleryVersionService:
    """
    Build immutable Gallery NPZ cache versions.

    Guarantees:
    - The base Gallery is never overwritten.
    - The new version is written to a temporary directory first.
    - The final version appears only after successful validation.
    - Activation only updates a small JSON pointer.
    """

    REQUIRED_GALLERY_KEYS = {
        "model_name",
        "model_sha256",
        "embedding_dimension",
        "image_embeddings",
        "image_student_ids",
        "image_full_names",
        "image_filenames",
        "student_ids",
        "student_names",
    }

    REQUIRED_PACKAGE_KEYS = {
        "enrollment_id",
        "enrollment_mode",
        "student_id",
        "full_name",
        "model_name",
        "model_sha256",
        "embedding_dimension",
        "image_embeddings",
        "student_template",
        "original_filenames",
    }

    def __init__(
        self,
        root: Path,
        versions_root: Optional[Path] = None,
        active_pointer_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(root).resolve()

        self.versions_root = self._resolve(
            versions_root
            or Path(
                "data/generated/gallery_versions"
            )
        )

        self.active_pointer_path = self._resolve(
            active_pointer_path
            or Path(
                "data/generated/active_gallery.json"
            )
        )

        self.versions_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.active_pointer_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def build_version_from_package(
        self,
        *,
        version_id: str,
        base_gallery_path: Path,
        enrollment_package_path: Path,
        update_policy: GalleryUpdatePolicy = (
            GalleryUpdatePolicy.REPLACE
        ),
        approve_reviewed_package: bool = False,
    ) -> GalleryVersionResult:
        """
        Build a new Gallery version without activating it.

        APPEND:
            Keep old student embeddings and append new ones.

        REPLACE:
            Remove the target student's old embeddings from the
            new cache and replace them with package embeddings.
            The base Gallery itself remains unchanged.
        """

        version_id = self._validate_id(
            version_id,
            "version_id",
        )

        policy = self._policy(update_policy)

        base_path = self._resolve(
            base_gallery_path
        )

        package_path = self._resolve(
            enrollment_package_path
        )

        if not base_path.is_file():
            raise FileNotFoundError(
                f"Base gallery not found: {base_path}"
            )

        if not package_path.is_file():
            raise FileNotFoundError(
                "Enrollment package not found: "
                f"{package_path}"
            )

        enrollment_manifest_path = (
            package_path.parent / "manifest.json"
        )

        enrollment_manifest = self._read_json(
            enrollment_manifest_path
        )

        enrollment_status = str(
            enrollment_manifest.get(
                "status",
                "",
            )
        ).upper()

        if enrollment_status == "REJECTED":
            raise GalleryVersionValidationError(
                "Rejected enrollment package."
            )

        if (
            enrollment_status
            == "READY_FOR_REVIEW"
            and not approve_reviewed_package
        ):
            raise GalleryVersionValidationError(
                "Package needs review approval. "
                "Pass approve_reviewed_package=True "
                "after manual review."
            )

        final_directory = (
            self.versions_root / version_id
        )

        if final_directory.exists():
            raise FileExistsError(
                "Gallery version already exists: "
                f"{final_directory}"
            )

        temporary_directory = (
            self.versions_root
            / (
                f".{version_id}.tmp-"
                f"{uuid.uuid4().hex}"
            )
        )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        created_at = datetime.now(
            timezone.utc
        )

        base_hash = self._sha256(
            base_path
        )

        try:
            base = self._read_npz(
                base_path,
                self.REQUIRED_GALLERY_KEYS,
            )

            package = self._read_npz(
                package_path,
                self.REQUIRED_PACKAGE_KEYS,
            )

            metadata = (
                self._validate_compatibility(
                    base,
                    package,
                )
            )

            base_student_ids = self._strings(
                base["student_ids"]
            )

            base_student_names = self._strings(
                base["student_names"]
            )

            if (
                len(base_student_ids)
                != len(base_student_names)
            ):
                raise GalleryVersionValidationError(
                    "Invalid student ID/name arrays."
                )

            if (
                len(set(base_student_ids))
                != len(base_student_ids)
            ):
                raise GalleryVersionValidationError(
                    "Duplicate student IDs "
                    "in base gallery."
                )

            student_name_map = dict(
                zip(
                    base_student_ids,
                    base_student_names,
                )
            )

            student_exists = (
                metadata["student_id"]
                in student_name_map
            )

            self._validate_operation(
                mode=metadata[
                    "enrollment_mode"
                ],
                student_id=metadata[
                    "student_id"
                ],
                full_name=metadata[
                    "full_name"
                ],
                exists=student_exists,
                existing_name=(
                    student_name_map.get(
                        metadata["student_id"]
                    )
                ),
                policy=policy,
            )

            arrays, merge_info = self._merge(
                base=base,
                package=package,
                enrollment_id=metadata[
                    "enrollment_id"
                ],
                enrollment_mode=metadata[
                    "enrollment_mode"
                ],
                student_id=metadata[
                    "student_id"
                ],
                full_name=metadata[
                    "full_name"
                ],
                policy=policy,
                ordered_student_ids=(
                    base_student_ids
                ),
            )

            (
                student_templates,
                ordered_student_ids,
                ordered_student_names,
                student_image_counts,
            ) = self._templates(
                embeddings=arrays[
                    "image_embeddings"
                ],
                image_student_ids=(
                    self._strings(
                        arrays[
                            "image_student_ids"
                        ]
                    )
                ),
                image_full_names=(
                    self._strings(
                        arrays[
                            "image_full_names"
                        ]
                    )
                ),
                ordered_student_ids=(
                    merge_info[
                        "ordered_student_ids"
                    ]
                ),
            )

            arrays.update(
                {
                    "student_templates": (
                        student_templates
                    ),
                    "student_ids": (
                        np.asarray(
                            ordered_student_ids,
                            dtype=np.str_,
                        )
                    ),
                    "student_names": (
                        np.asarray(
                            ordered_student_names,
                            dtype=np.str_,
                        )
                    ),
                    "student_image_counts": (
                        np.asarray(
                            student_image_counts,
                            dtype=np.int32,
                        )
                    ),
                }
            )

            package_template = self._normalize(
                np.asarray(
                    package[
                        "student_template"
                    ],
                    dtype=np.float32,
                ).reshape(-1)
            )

            target_index = (
                ordered_student_ids.index(
                    metadata["student_id"]
                )
            )

            target_similarity = float(
                student_templates[
                    target_index
                ]
                @ package_template
            )

            warnings = self._audit(
                templates=student_templates,
                student_ids=(
                    ordered_student_ids
                ),
                target_student_id=metadata[
                    "student_id"
                ],
                target_similarity=(
                    target_similarity
                ),
            )

            status = (
                GalleryVersionStatus
                .READY_FOR_REVIEW
                if warnings
                else GalleryVersionStatus
                .READY_FOR_ACTIVATION
            )

            temporary_gallery_path = (
                temporary_directory
                / "face_gallery.npz"
            )

            self._write_gallery(
                temporary_gallery_path,
                version_id=version_id,
                created_at=created_at,
                base_hash=base_hash,
                enrollment_id=metadata[
                    "enrollment_id"
                ],
                model_name=metadata[
                    "model_name"
                ],
                model_hash=metadata[
                    "model_sha256"
                ],
                dimension=metadata[
                    "embedding_dimension"
                ],
                arrays=arrays,
            )

            # Ensure the produced cache is compatible
            # with the current matcher.
            FaceGalleryMatcher(
                temporary_gallery_path
            )

            gallery_hash = self._sha256(
                temporary_gallery_path
            )

            result = GalleryVersionResult(
                version_id=version_id,
                status=status,
                update_policy=policy,
                enrollment_id=metadata[
                    "enrollment_id"
                ],
                student_id=metadata[
                    "student_id"
                ],
                full_name=metadata[
                    "full_name"
                ],
                base_gallery_path=base_path,
                base_gallery_sha256=base_hash,
                version_directory=(
                    final_directory
                ),
                gallery_path=(
                    final_directory
                    / "face_gallery.npz"
                ),
                gallery_sha256=gallery_hash,
                manifest_json=(
                    final_directory
                    / "manifest.json"
                ),
                base_student_count=len(
                    base_student_ids
                ),
                new_student_count=len(
                    ordered_student_ids
                ),
                base_embedding_count=len(
                    base["image_embeddings"]
                ),
                new_embedding_count=len(
                    arrays[
                        "image_embeddings"
                    ]
                ),
                removed_target_embeddings=(
                    merge_info["removed"]
                ),
                added_target_embeddings=(
                    merge_info["added"]
                ),
                target_embedding_count=(
                    merge_info["target_count"]
                ),
                target_template_similarity=(
                    round(
                        target_similarity,
                        6,
                    )
                ),
                warnings=warnings,
                created_at=created_at,
            )

            manifest = result.to_dict()

            manifest.update(
                {
                    "activated": False,
                    "enrollment_package_path": (
                        str(package_path)
                    ),
                    "enrollment_manifest_status": (
                        enrollment_status
                    ),
                }
            )

            self._write_json(
                temporary_directory
                / "manifest.json",
                manifest,
            )

            # Publish the complete version directory.
            temporary_directory.replace(
                final_directory
            )

            return result

        except Exception:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )
            raise

#هذه الداله تقوم ب بناء الاستديو بناءا على حذف الطالب المراد حذفه
    def build_version_without_student(
        self,
        *,
        version_id: str,
        base_gallery_path: Path,
        student_id: str,
    ) -> GalleryStudentRemovalResult:
        """
        Build a new immutable Gallery version without one student.

        The base Gallery is never modified.
        The produced version is not activated automatically.
        """

        version_id = self._validate_id(
            version_id,
            "version_id",
        )

        student_id = self._validate_id(
            student_id,
            "student_id",
        )

        base_path = self._resolve(
            base_gallery_path
        )

        if not base_path.is_file():
            raise FileNotFoundError(
                f"Base gallery not found: {base_path}"
            )

        final_directory = (
            self.versions_root / version_id
        )

        if final_directory.exists():
            raise FileExistsError(
                f"Gallery version already exists: {final_directory}"
            )

        temporary_directory = (
            self.versions_root
            / f".{version_id}.tmp-{uuid.uuid4().hex}"
        )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        created_at = datetime.now(
            timezone.utc
        )

        base_hash = self._sha256(
            base_path
        )

        try:
            base = self._read_npz(
                base_path,
                self.REQUIRED_GALLERY_KEYS,
            )

            base_embeddings = np.asarray(
                base["image_embeddings"],
                dtype=np.float32,
            )

            if base_embeddings.ndim != 2:
                raise GalleryVersionValidationError(
                    "Invalid base image_embeddings shape."
                )

            base_count = len(
                base_embeddings
            )

            dimension = int(
                np.asarray(
                    base["embedding_dimension"]
                ).item()
            )

            model_name = self._scalar(
                base["model_name"]
            )

            model_hash = self._scalar(
                base["model_sha256"]
            ).upper()

            base_ids = self._required_array(
                base,
                "image_student_ids",
                base_count,
            )

            base_names = self._required_array(
                base,
                "image_full_names",
                base_count,
            )

            base_filenames = self._required_array(
                base,
                "image_filenames",
                base_count,
            )

            base_sources = self._optional_array(
                base,
                "image_sources",
                base_count,
                "legacy",
            )

            base_candidate_ids = self._optional_array(
                base,
                "image_candidate_ids",
                base_count,
                "",
            )

            base_hashes = self._optional_array(
                base,
                "image_sha256",
                base_count,
                "",
            )

            ordered_student_ids = self._strings(
                base["student_ids"]
            )

            ordered_student_names = self._strings(
                base["student_names"]
            )

            if (
                len(ordered_student_ids)
                != len(ordered_student_names)
            ):
                raise GalleryVersionValidationError(
                    "Invalid student ID/name arrays."
                )

            if (
                len(set(ordered_student_ids))
                != len(ordered_student_ids)
            ):
                raise GalleryVersionValidationError(
                    "Duplicate student IDs in gallery."
                )

            student_name_map = dict(
                zip(
                    ordered_student_ids,
                    ordered_student_names,
                )
            )

            if student_id not in student_name_map:
                raise GalleryVersionValidationError(
                    f"Student does not exist in gallery: {student_id}"
                )

            full_name = student_name_map[
                student_id
            ]

            removed_embedding_count = int(
                np.sum(
                    base_ids == student_id
                )
            )

            if removed_embedding_count <= 0:
                raise GalleryVersionValidationError(
                    f"No embeddings found for student: {student_id}"
                )

            keep_mask = (
                base_ids != student_id
            )

            remaining_embeddings = (
                base_embeddings[keep_mask]
                .astype(np.float32)
            )

            remaining_ids = (
                base_ids[keep_mask]
            )

            remaining_names = (
                base_names[keep_mask]
            )

            arrays = {
                "image_embeddings": remaining_embeddings,
                "image_student_ids": remaining_ids,
                "image_full_names": remaining_names,
                "image_filenames": base_filenames[
                    keep_mask
                ],
                "image_sources": base_sources[
                    keep_mask
                ],
                "image_candidate_ids": (
                    base_candidate_ids[
                        keep_mask
                    ]
                ),
                "image_sha256": base_hashes[
                    keep_mask
                ],
            }

            new_order = [
                value
                for value in ordered_student_ids
                if value != student_id
            ]

            if new_order:
                (
                    student_templates,
                    new_order,
                    new_names,
                    student_image_counts,
                ) = self._templates(
                    embeddings=remaining_embeddings,
                    image_student_ids=(
                        remaining_ids
                        .astype(str)
                        .tolist()
                    ),
                    image_full_names=(
                        remaining_names
                        .astype(str)
                        .tolist()
                    ),
                    ordered_student_ids=new_order,
                )

            else:
                # Valid empty Gallery.
                student_templates = np.empty(
                    (0, dimension),
                    dtype=np.float32,
                )

                new_names = []

                student_image_counts = []

            arrays.update(
                {
                    "student_templates": (
                        student_templates
                    ),
                    "student_ids": np.asarray(
                        new_order,
                        dtype=np.str_,
                    ),
                    "student_names": np.asarray(
                        new_names,
                        dtype=np.str_,
                    ),
                    "student_image_counts": np.asarray(
                        student_image_counts,
                        dtype=np.int32,
                    ),
                }
            )

            temporary_gallery_path = (
                temporary_directory
                / "face_gallery.npz"
            )

            self._write_gallery(
                temporary_gallery_path,
                version_id=version_id,
                created_at=created_at,
                base_hash=base_hash,

                # Keep NPZ schema 2.0 compatible.
                # This is operation metadata, not a real enrollment.
                enrollment_id=(
                    f"REMOVE_{student_id}"
                ),

                model_name=model_name,
                model_hash=model_hash,
                dimension=dimension,
                arrays=arrays,
            )

            # Ensure matcher can load the produced cache.
            FaceGalleryMatcher(
                temporary_gallery_path
            )

            gallery_hash = self._sha256(
                temporary_gallery_path
            )

            result = GalleryStudentRemovalResult(
                version_id=version_id,
                status=(
                    GalleryVersionStatus
                    .READY_FOR_ACTIVATION
                ),
                student_id=student_id,
                full_name=full_name,
                base_gallery_path=base_path,
                base_gallery_sha256=base_hash,
                version_directory=(
                    final_directory
                ),
                gallery_path=(
                    final_directory
                    / "face_gallery.npz"
                ),
                gallery_sha256=gallery_hash,
                manifest_json=(
                    final_directory
                    / "manifest.json"
                ),
                base_student_count=len(
                    ordered_student_ids
                ),
                new_student_count=len(
                    new_order
                ),
                base_embedding_count=(
                    base_count
                ),
                new_embedding_count=len(
                    remaining_embeddings
                ),
                removed_embedding_count=(
                    removed_embedding_count
                ),
                created_at=created_at,
            )

            manifest = result.to_dict()

            manifest.update(
                {
                    "operation": (
                        "REMOVE_STUDENT"
                    ),
                    "activated": False,
                }
            )

            self._write_json(
                temporary_directory
                / "manifest.json",
                manifest,
            )

            temporary_directory.replace(
                final_directory
            )

            return result

        except Exception:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )
            raise


#هذه الداله تقوم ب
    def build_version_from_records(
        self,
        *,
        version_id: str,
        records: Iterable[Mapping[str, Any]],
        model_name: str,
        model_sha256: str,
        embedding_dimension: int,
    ) -> GalleryRebuildResult:
        """
        Build a complete immutable Gallery from embedding records.

        An empty records iterable produces a valid empty Gallery
        and therefore acts as the system bootstrap operation.

        This method never activates the produced version.
        """

        version_id = self._validate_id(
            version_id,
            "version_id",
        )

        model_name = str(
            model_name
        ).strip()

        if not model_name:
            raise GalleryVersionValidationError(
                "model_name cannot be empty."
            )

        model_sha256 = str(
            model_sha256
        ).strip().upper()

        if not model_sha256:
            raise GalleryVersionValidationError(
                "model_sha256 cannot be empty."
            )

        dimension = int(
            embedding_dimension
        )

        if dimension < 1:
            raise GalleryVersionValidationError(
                "embedding_dimension must be positive."
            )

        source_records = list(
            records
        )

        final_directory = (
            self.versions_root
            / version_id
        )

        if final_directory.exists():
            raise FileExistsError(
                "Gallery version already exists: "
                f"{final_directory}"
            )

        temporary_directory = (
            self.versions_root
            / (
                f".{version_id}.tmp-"
                f"{uuid.uuid4().hex}"
            )
        )

        temporary_directory.mkdir(
            parents=True,
            exist_ok=False,
        )

        created_at = datetime.now(
            timezone.utc
        )

        try:
            embeddings = []

            image_student_ids = []
            image_full_names = []
            image_filenames = []
            image_sources = []
            image_candidate_ids = []
            image_hashes = []

            student_names: Dict[
                str,
                str,
            ] = {}

            ordered_student_ids = []

            for index, raw_record in enumerate(
                source_records,
                start=1,
            ):
                if not isinstance(
                    raw_record,
                    Mapping,
                ):
                    raise GalleryVersionValidationError(
                        "Every Gallery rebuild record "
                        "must be a mapping."
                    )

                if "student_id" not in raw_record:
                    raise GalleryVersionValidationError(
                        "Gallery rebuild record is "
                        "missing student_id."
                    )

                if "full_name" not in raw_record:
                    raise GalleryVersionValidationError(
                        "Gallery rebuild record is "
                        "missing full_name."
                    )

                if "embedding" not in raw_record:
                    raise GalleryVersionValidationError(
                        "Gallery rebuild record is "
                        "missing embedding."
                    )

                student_id = self._validate_id(
                    raw_record["student_id"],
                    "student_id",
                )

                full_name = str(
                    raw_record["full_name"]
                ).strip()

                if not full_name:
                    raise GalleryVersionValidationError(
                        "Student full_name cannot "
                        "be empty."
                    )

                existing_name = (
                    student_names.get(
                        student_id
                    )
                )

                if (
                    existing_name is not None
                    and existing_name != full_name
                ):
                    raise GalleryVersionValidationError(
                        "Inconsistent full_name for "
                        f"student {student_id}: "
                        f"{existing_name!r} != "
                        f"{full_name!r}"
                    )

                if student_id not in student_names:
                    student_names[
                        student_id
                    ] = full_name

                    ordered_student_ids.append(
                        student_id
                    )

                embedding = np.asarray(
                    raw_record["embedding"],
                    dtype=np.float32,
                ).reshape(-1)

                if embedding.size != dimension:
                    raise GalleryVersionValidationError(
                        "Embedding dimension mismatch "
                        f"for student {student_id}: "
                        f"expected {dimension}, "
                        f"got {embedding.size}."
                    )

                if not np.all(
                    np.isfinite(
                        embedding
                    )
                ):
                    raise GalleryVersionValidationError(
                        "Embedding contains non-finite "
                        f"values for student {student_id}."
                    )

                norm = float(
                    np.linalg.norm(
                        embedding
                    )
                )

                if norm <= 1e-12:
                    raise GalleryVersionValidationError(
                        "Zero-norm embedding for "
                        f"student {student_id}."
                    )

                embedding = (
                    embedding / norm
                ).astype(
                    np.float32
                )

                record_model_name = (
                    raw_record.get(
                        "model_name"
                    )
                )

                if (
                    record_model_name
                    not in (None, "")
                    and str(
                        record_model_name
                    ).strip()
                    != model_name
                ):
                    raise GalleryVersionValidationError(
                        "Record model_name does not "
                        "match the active recognizer."
                    )

                record_model_hash = (
                    raw_record.get(
                        "model_sha256"
                    )
                )

                if (
                    record_model_hash
                    not in (None, "")
                    and str(
                        record_model_hash
                    ).strip().upper()
                    != model_sha256
                ):
                    raise GalleryVersionValidationError(
                        "Record model_sha256 does not "
                        "match the active recognizer."
                    )

                record_dimension = (
                    raw_record.get(
                        "embedding_dimension"
                    )
                )

                if (
                    record_dimension is not None
                    and int(
                        record_dimension
                    )
                    != dimension
                ):
                    raise GalleryVersionValidationError(
                        "Record embedding_dimension "
                        "does not match the active "
                        "recognizer."
                    )

                filename = str(
                    raw_record.get(
                        "image_filename"
                    )
                    or raw_record.get(
                        "image_path"
                    )
                    or (
                        f"db/{student_id}/"
                        f"{index:06d}"
                    )
                )

                source = str(
                    raw_record.get(
                        "source",
                        "database_rebuild",
                    )
                )

                candidate_id = str(
                    raw_record.get(
                        "record_id",
                        "",
                    )
                )

                image_sha256 = str(
                    raw_record.get(
                        "image_sha256",
                        "",
                    )
                ).upper()

                embeddings.append(
                    embedding
                )

                image_student_ids.append(
                    student_id
                )

                image_full_names.append(
                    full_name
                )

                image_filenames.append(
                    filename
                )

                image_sources.append(
                    source
                )

                image_candidate_ids.append(
                    candidate_id
                )

                image_hashes.append(
                    image_sha256
                )

            if embeddings:
                image_embeddings = np.stack(
                    embeddings,
                    axis=0,
                ).astype(
                    np.float32
                )

                (
                    student_templates,
                    ordered_student_ids,
                    ordered_student_names,
                    student_image_counts,
                ) = self._templates(
                    embeddings=(
                        image_embeddings
                    ),
                    image_student_ids=list(
                        image_student_ids
                    ),
                    image_full_names=list(
                        image_full_names
                    ),
                    ordered_student_ids=list(
                        ordered_student_ids
                    ),
                )

            else:
                # Valid zero-student bootstrap Gallery.
                image_embeddings = np.empty(
                    (0, dimension),
                    dtype=np.float32,
                )

                student_templates = np.empty(
                    (0, dimension),
                    dtype=np.float32,
                )

                ordered_student_names = []

                student_image_counts = []

            arrays = {
                "image_embeddings": (
                    image_embeddings
                ),
                "image_student_ids": np.asarray(
                    image_student_ids,
                    dtype=np.str_,
                ),
                "image_full_names": np.asarray(
                    image_full_names,
                    dtype=np.str_,
                ),
                "image_filenames": np.asarray(
                    image_filenames,
                    dtype=np.str_,
                ),
                "image_sources": np.asarray(
                    image_sources,
                    dtype=np.str_,
                ),
                "image_candidate_ids": np.asarray(
                    image_candidate_ids,
                    dtype=np.str_,
                ),
                "image_sha256": np.asarray(
                    image_hashes,
                    dtype=np.str_,
                ),
                "student_templates": (
                    student_templates
                ),
                "student_ids": np.asarray(
                    ordered_student_ids,
                    dtype=np.str_,
                ),
                "student_names": np.asarray(
                    ordered_student_names,
                    dtype=np.str_,
                ),
                "student_image_counts": np.asarray(
                    student_image_counts,
                    dtype=np.int32,
                ),
            }

            operation = (
                "BOOTSTRAP_EMPTY"
                if not source_records
                else "REBUILD_FROM_RECORDS"
            )

            temporary_gallery_path = (
                temporary_directory
                / "face_gallery.npz"
            )

            self._write_gallery(
                temporary_gallery_path,
                version_id=version_id,
                created_at=created_at,

                # A rebuild has no previous Gallery.
                base_hash="NONE",

                # Keep the existing NPZ schema compatible.
                enrollment_id=(
                    f"REBUILD_{version_id}"
                ),

                model_name=model_name,
                model_hash=model_sha256,
                dimension=dimension,
                arrays=arrays,
            )

            # Structural/runtime validation.
            FaceGalleryMatcher(
                temporary_gallery_path
            )

            gallery_hash = self._sha256(
                temporary_gallery_path
            )

            result = GalleryRebuildResult(
                version_id=version_id,
                status=(
                    GalleryVersionStatus
                    .READY_FOR_ACTIVATION
                ),
                operation=operation,
                model_name=model_name,
                model_sha256=model_sha256,
                embedding_dimension=dimension,
                version_directory=(
                    final_directory
                ),
                gallery_path=(
                    final_directory
                    / "face_gallery.npz"
                ),
                gallery_sha256=gallery_hash,
                manifest_json=(
                    final_directory
                    / "manifest.json"
                ),
                student_count=len(
                    ordered_student_ids
                ),
                embedding_count=len(
                    image_embeddings
                ),
                warnings=[],
                created_at=created_at,
            )

            manifest = result.to_dict()

            manifest.update(
                {
                    "activated": False,
                    "source_record_count": len(
                        source_records
                    ),
                }
            )

            self._write_json(
                temporary_directory
                / "manifest.json",
                manifest,
            )

            temporary_directory.replace(
                final_directory
            )

            return result

        except Exception:
            shutil.rmtree(
                temporary_directory,
                ignore_errors=True,
            )
            raise

    def activate_version(
        self,
        version_id: str,
    ) -> dict:
        """
        Activate an existing version.

        This does not copy or overwrite the Gallery.
        It only writes active_gallery.json atomically.
        """

        version_id = self._validate_id(
            version_id,
            "version_id",
        )

        version_directory = (
            self.versions_root / version_id
        )

        gallery_path = (
            version_directory
            / "face_gallery.npz"
        )

        manifest_path = (
            version_directory
            / "manifest.json"
        )

        if (
            not gallery_path.is_file()
            or not manifest_path.is_file()
        ):
            raise FileNotFoundError(
                "Incomplete gallery version: "
                f"{version_directory}"
            )

        manifest = self._read_json(
            manifest_path
        )

        expected_hash = str(
            manifest.get(
                "gallery_sha256",
                "",
            )
        ).upper()

        actual_hash = self._sha256(
            gallery_path
        )

        if expected_hash != actual_hash:
            raise GalleryVersionValidationError(
                "Gallery SHA256 mismatch."
            )

        FaceGalleryMatcher(
            gallery_path
        )

        pointer = {
            "active_version_id": version_id,
            "gallery_path": str(
                gallery_path
            ),
            "gallery_sha256": (
                actual_hash
            ),
            "activated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        }

        self._write_json(
            self.active_pointer_path,
            pointer,
        )

        return pointer

    def get_active_gallery_path(
        self,
        fallback_path: Optional[
            Path
        ] = None,
    ) -> Path:
        """
        Return the activated Gallery path.

        If no version was activated, an optional legacy
        Gallery path may be supplied as fallback.
        """

        if self.active_pointer_path.is_file():
            pointer = self._read_json(
                self.active_pointer_path
            )

            gallery_path = Path(
                pointer["gallery_path"]
            ).resolve()

            if not gallery_path.is_file():
                raise FileNotFoundError(
                    "Broken active gallery "
                    f"pointer: {gallery_path}"
                )

            expected_hash = str(
                pointer[
                    "gallery_sha256"
                ]
            ).upper()

            actual_hash = self._sha256(
                gallery_path
            )

            if expected_hash != actual_hash:
                raise GalleryVersionValidationError(
                    "Active gallery hash mismatch."
                )

            return gallery_path

        if fallback_path is None:
            raise FileNotFoundError(
                "No active gallery pointer exists."
            )

        fallback = self._resolve(
            fallback_path
        )

        if not fallback.is_file():
            raise FileNotFoundError(
                "Fallback gallery not found: "
                f"{fallback}"
            )

        return fallback

    def _merge(
        self,
        *,
        base: Dict[
            str,
            np.ndarray,
        ],
        package: Dict[
            str,
            np.ndarray,
        ],
        enrollment_id: str,
        enrollment_mode: str,
        student_id: str,
        full_name: str,
        policy: GalleryUpdatePolicy,
        ordered_student_ids: List[str],
    ) -> Tuple[dict, dict]:
        base_embeddings = np.asarray(
            base["image_embeddings"],
            dtype=np.float32,
        )

        package_embeddings = np.asarray(
            package["image_embeddings"],
            dtype=np.float32,
        )

        self._validate_embeddings(
            base_embeddings,
            "base",
            allow_empty=True,
        )

        self._validate_embeddings(
            package_embeddings,
            "package",
        )

        

        base_count = len(
            base_embeddings
        )

        package_count = len(
            package_embeddings
        )

        base_ids = self._required_array(
            base,
            "image_student_ids",
            base_count,
        )

        base_names = self._required_array(
            base,
            "image_full_names",
            base_count,
        )

        base_filenames = (
            self._required_array(
                base,
                "image_filenames",
                base_count,
            )
        )

        base_sources = (
            self._optional_array(
                base,
                "image_sources",
                base_count,
                "legacy",
            )
        )

        base_candidate_ids = (
            self._optional_array(
                base,
                "image_candidate_ids",
                base_count,
                "",
            )
        )

        base_hashes = (
            self._optional_array(
                base,
                "image_sha256",
                base_count,
                "",
            )
        )

        old_target_count = int(
            np.sum(
                base_ids == student_id
            )
        )

        keep_mask = np.ones(
            base_count,
            dtype=bool,
        )

        removed_count = 0

        if (
            enrollment_mode == "UPDATE"
            and policy
            == GalleryUpdatePolicy.REPLACE
        ):
            keep_mask = (
                base_ids != student_id
            )

            removed_count = (
                old_target_count
            )

        filename_key = (
            "stored_aligned_filenames"
            if (
                "stored_aligned_filenames"
                in package
            )
            else "original_filenames"
        )

        package_filenames = np.asarray(
            package[filename_key]
        ).astype(str)

        if (
            len(package_filenames)
            != package_count
        ):
            raise GalleryVersionValidationError(
                "Package filename count mismatch."
            )

        package_filenames = np.asarray(
            [
                f"{enrollment_id}/{name}"
                for name
                in package_filenames
            ],
            dtype=np.str_,
        )

        package_hashes = (
            self._optional_array(
                package,
                "image_sha256",
                package_count,
                "",
            )
        )

        package_source = (
            "enrollment_new"
            if enrollment_mode == "NEW"
            else "enrollment_update"
        )

        arrays = {
            "image_embeddings": (
                np.concatenate(
                    [
                        base_embeddings[
                            keep_mask
                        ],
                        package_embeddings,
                    ],
                    axis=0,
                ).astype(np.float32)
            ),
            "image_student_ids": (
                np.concatenate(
                    [
                        base_ids[
                            keep_mask
                        ],
                        np.asarray(
                            [
                                student_id
                            ]
                            * package_count,
                            dtype=np.str_,
                        ),
                    ]
                )
            ),
            "image_full_names": (
                np.concatenate(
                    [
                        base_names[
                            keep_mask
                        ],
                        np.asarray(
                            [
                                full_name
                            ]
                            * package_count,
                            dtype=np.str_,
                        ),
                    ]
                )
            ),
            "image_filenames": (
                np.concatenate(
                    [
                        base_filenames[
                            keep_mask
                        ],
                        package_filenames,
                    ]
                )
            ),
            "image_sources": (
                np.concatenate(
                    [
                        base_sources[
                            keep_mask
                        ],
                        np.asarray(
                            [
                                package_source
                            ]
                            * package_count,
                            dtype=np.str_,
                        ),
                    ]
                )
            ),
            "image_candidate_ids": (
                np.concatenate(
                    [
                        base_candidate_ids[
                            keep_mask
                        ],
                        np.asarray(
                            [
                                enrollment_id
                            ]
                            * package_count,
                            dtype=np.str_,
                        ),
                    ]
                )
            ),
            "image_sha256": (
                np.concatenate(
                    [
                        base_hashes[
                            keep_mask
                        ],
                        package_hashes,
                    ]
                )
            ),
        }

        new_order = list(
            ordered_student_ids
        )

        if student_id not in new_order:
            new_order.append(
                student_id
            )

        merged_ids = arrays[
            "image_student_ids"
        ].astype(str)

        merge_info = {
            "ordered_student_ids": (
                new_order
            ),
            "removed": removed_count,
            "added": package_count,
            "target_count": int(
                np.sum(
                    merged_ids
                    == student_id
                )
            ),
        }

        return arrays, merge_info

    def _templates(
        self,
        *,
        embeddings: np.ndarray,
        image_student_ids: List[str],
        image_full_names: List[str],
        ordered_student_ids: List[str],
    ) -> Tuple[
        np.ndarray,
        List[str],
        List[str],
        List[int],
    ]:
        templates = []
        names = []
        counts = []

        for student_id in (
            ordered_student_ids
        ):
            indices = [
                index
                for index, value
                in enumerate(
                    image_student_ids
                )
                if value == student_id
            ]

            if not indices:
                raise GalleryVersionValidationError(
                    "No embeddings for "
                    f"{student_id}."
                )

            student_names = {
                image_full_names[index]
                for index in indices
            }

            if len(student_names) != 1:
                raise GalleryVersionValidationError(
                    "Inconsistent names for "
                    f"{student_id}: "
                    f"{sorted(student_names)}"
                )

            mean_embedding = embeddings[
                indices
            ].mean(
                axis=0,
                keepdims=True,
            )

            template = self._l2(
                mean_embedding
            )[0]

            templates.append(
                template
            )

            names.append(
                next(
                    iter(student_names)
                )
            )

            counts.append(
                len(indices)
            )

        return (
            np.stack(
                templates
            ).astype(np.float32),
            ordered_student_ids,
            names,
            counts,
        )

    def _audit(
        self,
        *,
        templates: np.ndarray,
        student_ids: List[str],
        target_student_id: str,
        target_similarity: float,
    ) -> List[str]:
        self._validate_embeddings(
            templates,
            "templates",
        )

        warnings = []

        if target_similarity < 0.75:
            warnings.append(
                "Target template differs "
                "from package template: "
                f"{target_similarity:.6f}."
            )

        target_index = student_ids.index(
            target_student_id
        )

        if len(student_ids) > 1:
            scores = (
                templates
                @ templates[target_index]
            )

            scores[target_index] = -1.0

            closest_index = int(
                np.argmax(scores)
            )

            closest_score = float(
                scores[closest_index]
            )

            if closest_score >= 0.70:
                warnings.append(
                    "Target is unusually close "
                    "to another student: "
                    f"{student_ids[closest_index]} "
                    f"({closest_score:.6f})."
                )

        return warnings

    def _validate_compatibility(
        self,
        base: Dict[
            str,
            np.ndarray,
        ],
        package: Dict[
            str,
            np.ndarray,
        ],
    ) -> dict:
        base_model = self._scalar(
            base["model_name"]
        )

        package_model = self._scalar(
            package["model_name"]
        )

        base_hash = self._scalar(
            base["model_sha256"]
        ).upper()

        package_hash = self._scalar(
            package["model_sha256"]
        ).upper()

        base_dimension = int(
            np.asarray(
                base[
                    "embedding_dimension"
                ]
            ).item()
        )

        package_dimension = int(
            np.asarray(
                package[
                    "embedding_dimension"
                ]
            ).item()
        )

        if base_model != package_model:
            raise GalleryVersionValidationError(
                "Recognition model names differ."
            )

        if base_hash != package_hash:
            raise GalleryVersionValidationError(
                "Recognition model hashes differ."
            )

        if (
            base_dimension
            != package_dimension
        ):
            raise GalleryVersionValidationError(
                "Embedding dimensions differ."
            )

        enrollment_mode = (
            self._scalar(
                package[
                    "enrollment_mode"
                ]
            ).upper()
        )

        if enrollment_mode not in {
            "NEW",
            "UPDATE",
        }:
            raise GalleryVersionValidationError(
                "Unsupported mode: "
                f"{enrollment_mode}"
            )

        student_id = self._validate_id(
            self._scalar(
                package["student_id"]
            ),
            "student_id",
        )

        enrollment_id = (
            self._validate_id(
                self._scalar(
                    package[
                        "enrollment_id"
                    ]
                ),
                "enrollment_id",
            )
        )

        full_name = self._scalar(
            package["full_name"]
        ).strip()

        if not full_name:
            raise GalleryVersionValidationError(
                "Student full name is empty."
            )

        package_embeddings = np.asarray(
            package["image_embeddings"],
            dtype=np.float32,
        )

        package_template = np.asarray(
            package["student_template"],
            dtype=np.float32,
        ).reshape(-1)

        if (
            package_embeddings.ndim != 2
            or package_embeddings.shape[1]
            != base_dimension
        ):
            raise GalleryVersionValidationError(
                "Invalid package shape: "
                f"{package_embeddings.shape}"
            )

        if (
            package_template.size
            != base_dimension
        ):
            raise GalleryVersionValidationError(
                "Invalid package template "
                "dimension."
            )

        return {
            "enrollment_id": enrollment_id,
            "enrollment_mode": (
                enrollment_mode
            ),
            "student_id": student_id,
            "full_name": full_name,
            "model_name": base_model,
            "model_sha256": base_hash,
            "embedding_dimension": (
                base_dimension
            ),
        }

    @staticmethod
    def _validate_operation(
        *,
        mode: str,
        student_id: str,
        full_name: str,
        exists: bool,
        existing_name: Optional[str],
        policy: GalleryUpdatePolicy,
    ) -> None:
        if mode == "NEW" and exists:
            raise GalleryVersionValidationError(
                "Student already exists: "
                f"{student_id}"
            )

        if (
            mode == "UPDATE"
            and not exists
        ):
            raise GalleryVersionValidationError(
                "Student does not exist: "
                f"{student_id}"
            )

        if (
            mode == "NEW"
            and policy
            != GalleryUpdatePolicy.APPEND
        ):
            raise GalleryVersionValidationError(
                "NEW enrollment must use APPEND."
            )

        if (
            exists
            and existing_name
            != full_name
        ):
            raise GalleryVersionValidationError(
                "Student name differs "
                "from base gallery."
            )

    def _write_gallery(
        self,
        path: Path,
        *,
        version_id: str,
        created_at: datetime,
        base_hash: str,
        enrollment_id: str,
        model_name: str,
        model_hash: str,
        dimension: int,
        arrays: dict,
    ) -> None:
        temporary = path.with_name(
            path.stem + ".tmp.npz"
        )

        np.savez_compressed(
            temporary,
            schema_version=np.asarray(
                "2.0"
            ),
            gallery_version_id=np.asarray(
                version_id
            ),
            created_at_utc=np.asarray(
                created_at.isoformat()
            ),
            base_gallery_sha256=np.asarray(
                base_hash
            ),
            enrollment_id=np.asarray(
                enrollment_id
            ),
            model_name=np.asarray(
                model_name
            ),
            model_sha256=np.asarray(
                model_hash
            ),
            embedding_dimension=np.asarray(
                dimension,
                dtype=np.int32,
            ),
            **arrays,
        )

        temporary.replace(
            path
        )

    @staticmethod
    def _read_npz(
        path: Path,
        required: set,
    ) -> Dict[str, np.ndarray]:
        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            missing = sorted(
                required
                - set(archive.files)
            )

            if missing:
                raise GalleryVersionValidationError(
                    "Missing NPZ keys: "
                    f"{missing}"
                )

            return {
                key: np.asarray(
                    archive[key]
                )
                for key in archive.files
            }

    @staticmethod
    def _required_array(
        data: Dict[
            str,
            np.ndarray,
        ],
        key: str,
        count: int,
    ) -> np.ndarray:
        values = np.asarray(
            data[key]
        ).astype(str)

        if len(values) != count:
            raise GalleryVersionValidationError(
                f"Invalid length for {key}."
            )

        return values

    @staticmethod
    def _optional_array(
        data: Dict[
            str,
            np.ndarray,
        ],
        key: str,
        count: int,
        default: str,
    ) -> np.ndarray:
        if key not in data:
            return np.asarray(
                [default] * count,
                dtype=np.str_,
            )

        values = np.asarray(
            data[key]
        ).astype(str)

        if len(values) != count:
            raise GalleryVersionValidationError(
                f"Invalid length for {key}."
            )

        return values

    @staticmethod
    def _validate_embeddings(
        values: np.ndarray,
        label: str,
        *,
        allow_empty: bool = False,
    ) -> None:
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        if (
            values.ndim != 2
            or not np.all(
                np.isfinite(values)
            )
        ):
            raise GalleryVersionValidationError(
                f"Invalid {label} embeddings."
            )

        if len(values) == 0:
            if (
                allow_empty
                and values.shape[1] > 0
            ):
                return

            raise GalleryVersionValidationError(
                f"Invalid {label} embeddings."
            )

        norms = np.linalg.norm(
            values,
            axis=1,
        )

        if not np.allclose(
            norms,
            1.0,
            atol=1e-5,
        ):
            raise GalleryVersionValidationError(
                f"{label} embeddings are not normalized."
            )
    @staticmethod
    def _l2(
        values: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            values,
            axis=1,
            keepdims=True,
        )

        return (
            values
            / np.maximum(
                norms,
                1e-12,
            )
        ).astype(np.float32)

    @staticmethod
    def _normalize(
        value: np.ndarray,
    ) -> np.ndarray:
        value = np.asarray(
            value,
            dtype=np.float32,
        ).reshape(-1)

        norm = float(
            np.linalg.norm(value)
        )

        if norm <= 1e-12:
            raise GalleryVersionValidationError(
                "Zero-norm embedding."
            )

        return (
            value / norm
        ).astype(np.float32)

    @staticmethod
    def _strings(
        value: np.ndarray,
    ) -> List[str]:
        return (
            np.asarray(value)
            .astype(str)
            .reshape(-1)
            .tolist()
        )

    @staticmethod
    def _scalar(
        value: np.ndarray,
    ) -> str:
        array = np.asarray(
            value
        )

        if array.size != 1:
            raise GalleryVersionValidationError(
                "Expected scalar NPZ metadata."
            )

        return str(
            array.reshape(-1)[0]
        )

    @staticmethod
    def _policy(
        value,
    ) -> GalleryUpdatePolicy:
        if isinstance(
            value,
            GalleryUpdatePolicy,
        ):
            return value

        return GalleryUpdatePolicy(
            str(value).upper()
        )

    @staticmethod
    def _validate_id(
        value: str,
        field: str,
    ) -> str:
        value = str(
            value
        ).strip()

        if not _SAFE_ID.fullmatch(
            value
        ):
            raise GalleryVersionValidationError(
                f"Invalid {field}: {value!r}"
            )

        return value

    def _resolve(
        self,
        value: Path,
    ) -> Path:
        path = Path(value)

        if not path.is_absolute():
            path = self.root / path

        return path.resolve()

    @staticmethod
    def _sha256(
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

    @staticmethod
    def _read_json(
        path: Path,
    ) -> dict:
        if not path.is_file():
            raise FileNotFoundError(
                f"JSON not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            value = json.load(file)

        if not isinstance(
            value,
            dict,
        ):
            raise GalleryVersionValidationError(
                "JSON object expected: "
                f"{path}"
            )

        return value

    @staticmethod
    def _write_json(
        path: Path,
        value: dict,
    ) -> None:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = path.with_suffix(
            path.suffix + ".tmp"
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                value,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary.replace(
            path
        )