"""Schemas used by the versioned face-gallery cache service."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional


class GalleryUpdatePolicy(str, Enum):
    APPEND = "APPEND"
    REPLACE = "REPLACE"


class GalleryVersionStatus(str, Enum):
    READY_FOR_ACTIVATION = "READY_FOR_ACTIVATION"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"


@dataclass
class GalleryVersionResult:
    version_id: str
    status: GalleryVersionStatus
    update_policy: GalleryUpdatePolicy
    enrollment_id: str
    student_id: str
    full_name: str

    base_gallery_path: Path
    base_gallery_sha256: str

    version_directory: Path
    gallery_path: Path
    gallery_sha256: str
    manifest_json: Path

    base_student_count: int
    new_student_count: int
    base_embedding_count: int
    new_embedding_count: int

    removed_target_embeddings: int
    added_target_embeddings: int
    target_embedding_count: int
    target_template_similarity: float

    warnings: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "status": self.status.value,
            "update_policy": self.update_policy.value,
            "enrollment_id": self.enrollment_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "base_gallery_path": str(self.base_gallery_path),
            "base_gallery_sha256": self.base_gallery_sha256,
            "version_directory": str(self.version_directory),
            "gallery_path": str(self.gallery_path),
            "gallery_sha256": self.gallery_sha256,
            "manifest_json": str(self.manifest_json),
            "base_student_count": self.base_student_count,
            "new_student_count": self.new_student_count,
            "base_embedding_count": self.base_embedding_count,
            "new_embedding_count": self.new_embedding_count,
            "removed_target_embeddings": (
                self.removed_target_embeddings
            ),
            "added_target_embeddings": (
                self.added_target_embeddings
            ),
            "target_embedding_count": (
                self.target_embedding_count
            ),
            "target_template_similarity": (
                self.target_template_similarity
            ),
            "warnings": list(self.warnings),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }


@dataclass
class GalleryStudentRemovalResult:
    version_id: str
    status: GalleryVersionStatus

    student_id: str
    full_name: str

    base_gallery_path: Path
    base_gallery_sha256: str

    version_directory: Path
    gallery_path: Path
    gallery_sha256: str
    manifest_json: Path

    base_student_count: int
    new_student_count: int

    base_embedding_count: int
    new_embedding_count: int

    removed_embedding_count: int

    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "status": self.status.value,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "base_gallery_path": str(self.base_gallery_path),
            "base_gallery_sha256": self.base_gallery_sha256,
            "version_directory": str(self.version_directory),
            "gallery_path": str(self.gallery_path),
            "gallery_sha256": self.gallery_sha256,
            "manifest_json": str(self.manifest_json),
            "base_student_count": self.base_student_count,
            "new_student_count": self.new_student_count,
            "base_embedding_count": self.base_embedding_count,
            "new_embedding_count": self.new_embedding_count,
            "removed_embedding_count": self.removed_embedding_count,
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }

@dataclass
class GalleryRebuildResult:
    version_id: str
    status: GalleryVersionStatus
    operation: str

    model_name: str
    model_sha256: str
    embedding_dimension: int

    version_directory: Path
    gallery_path: Path
    gallery_sha256: str
    manifest_json: Path

    student_count: int
    embedding_count: int

    warnings: List[str] = field(
        default_factory=list
    )

    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "version_id": self.version_id,
            "status": self.status.value,
            "operation": self.operation,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "version_directory": str(
                self.version_directory
            ),
            "gallery_path": str(
                self.gallery_path
            ),
            "gallery_sha256": (
                self.gallery_sha256
            ),
            "manifest_json": str(
                self.manifest_json
            ),
            "student_count": (
                self.student_count
            ),
            "embedding_count": (
                self.embedding_count
            ),
            "warnings": list(
                self.warnings
            ),
            "created_at": (
                self.created_at.isoformat()
                if self.created_at
                else None
            ),
        }