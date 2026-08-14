"""Schemas for safe, staged student face enrollment."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class EnrollmentMode(str, Enum):
    NEW = "NEW"
    UPDATE = "UPDATE"


class EnrollmentStatus(str, Enum):
    READY = "READY"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REJECTED = "REJECTED"


@dataclass
class EnrollmentImageResult:
    source_path: Path
    original_filename: str
    accepted: bool
    reason: str = ""
    image_sha256: str = ""
    stored_original_path: Optional[Path] = None
    stored_aligned_path: Optional[Path] = None
    detection_score: Optional[float] = None
    face_width: Optional[float] = None
    face_height: Optional[float] = None
    blur_score: Optional[float] = None
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    dark_ratio: Optional[float] = None
    bright_ratio: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "source_path": str(self.source_path),
            "original_filename": self.original_filename,
            "accepted": self.accepted,
            "reason": self.reason,
            "image_sha256": self.image_sha256,
            "stored_original_path": (
                str(self.stored_original_path)
                if self.stored_original_path
                else None
            ),
            "stored_aligned_path": (
                str(self.stored_aligned_path)
                if self.stored_aligned_path
                else None
            ),
            "detection_score": self.detection_score,
            "face_width": self.face_width,
            "face_height": self.face_height,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "dark_ratio": self.dark_ratio,
            "bright_ratio": self.bright_ratio,
        }


@dataclass
class EnrollmentIdentityCheck:
    status: str = "NOT_RUN"
    best_student_id: str = ""
    best_full_name: str = ""
    best_score: float = 0.0
    second_score: float = 0.0
    margin: float = 0.0
    top_candidates: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "best_student_id": self.best_student_id,
            "best_full_name": self.best_full_name,
            "best_score": self.best_score,
            "second_score": self.second_score,
            "margin": self.margin,
            "top_candidates": list(self.top_candidates),
        }


@dataclass
class EnrollmentResult:
    enrollment_id: str
    student_id: str
    full_name: str
    mode: EnrollmentMode
    status: EnrollmentStatus
    submitted_images: int
    accepted_images: int
    rejected_images: int
    package_directory: Path
    package_npz: Path
    manifest_json: Path
    image_results: List[EnrollmentImageResult]
    identity_check: EnrollmentIdentityCheck
    warnings: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "enrollment_id": self.enrollment_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "mode": self.mode.value,
            "status": self.status.value,
            "submitted_images": self.submitted_images,
            "accepted_images": self.accepted_images,
            "rejected_images": self.rejected_images,
            "package_directory": str(self.package_directory),
            "package_npz": str(self.package_npz),
            "manifest_json": str(self.manifest_json),
            "warnings": list(self.warnings),
            "identity_check": self.identity_check.to_dict(),
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
            "images": [item.to_dict() for item in self.image_results],
        }