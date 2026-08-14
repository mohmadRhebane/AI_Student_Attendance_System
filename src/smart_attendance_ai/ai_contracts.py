"""Stable public contracts exposed by :mod:`smart_attendance_ai.ai_facade`.

The backend may depend on these dataclasses while the internal detector,
recognizer, enrollment, gallery, attendance service, and worker continue to
evolve. Every public contract can be converted to JSON-friendly primitives
with ``to_dict()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Union

import numpy as np


SourceValue = Union[str, int, Path]


def _json_safe(value: Any) -> Any:
    """Recursively convert common Python/NumPy values to JSON-safe values."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _json_safe(to_dict())
    return str(value)


class AIContractMixin:
    """Provide dict-like compatibility without exposing internal dictionaries."""

    def to_dict(self) -> Dict[str, Any]:  # pragma: no cover - implemented below
        raise NotImplementedError

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self):
        return self.to_dict().keys()

    def items(self):
        return self.to_dict().items()

    def values(self):
        return self.to_dict().values()


@dataclass(frozen=True)
class AISettings:
    """Configurable paths and defaults for one AI facade instance."""

    project_root: Path
    ai_config_path: Optional[Path] = None
    legacy_gallery_path: Path = Path(
        "data/generated/face_gallery_calibrated.npz"
    )
    active_gallery_pointer_path: Path = Path(
        "data/generated/active_gallery.json"
    )
    gallery_versions_root: Path = Path(
        "data/generated/gallery_versions"
    )
    enrollment_staging_root: Path = Path(
        "data/enrollment_staging"
    )
    minimum_enrollment_images: int = 2

    def normalized(self) -> "AISettings":
        root = Path(self.project_root).resolve()

        def resolve_optional(path: Optional[Path]) -> Optional[Path]:
            if path is None:
                return None
            value = Path(path)
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        def resolve(path: Path) -> Path:
            value = Path(path)
            return value.resolve() if value.is_absolute() else (root / value).resolve()

        minimum = int(self.minimum_enrollment_images)
        if minimum < 1:
            raise ValueError("minimum_enrollment_images must be at least 1.")

        return AISettings(
            project_root=root,
            ai_config_path=resolve_optional(self.ai_config_path),
            legacy_gallery_path=resolve(self.legacy_gallery_path),
            active_gallery_pointer_path=resolve(
                self.active_gallery_pointer_path
            ),
            gallery_versions_root=resolve(self.gallery_versions_root),
            enrollment_staging_root=resolve(self.enrollment_staging_root),
            minimum_enrollment_images=minimum,
        )


@dataclass(frozen=True)
class EnrollmentImageAIResult(AIContractMixin):
    """AI validation result for one submitted enrollment image."""

    source_path: Path
    original_filename: str
    accepted: bool
    reason: str
    image_sha256: str
    embedding_index: Optional[int] = None
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

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "source_path": self.source_path,
            "original_filename": self.original_filename,
            "accepted": self.accepted,
            "reason": self.reason,
            "image_sha256": self.image_sha256,
            "embedding_index": self.embedding_index,
            "stored_original_path": self.stored_original_path,
            "stored_aligned_path": self.stored_aligned_path,
            "detection_score": self.detection_score,
            "face_width": self.face_width,
            "face_height": self.face_height,
            "blur_score": self.blur_score,
            "brightness": self.brightness,
            "contrast": self.contrast,
            "dark_ratio": self.dark_ratio,
            "bright_ratio": self.bright_ratio,
        })


@dataclass
class EnrollmentAIResult(AIContractMixin):
    """Stable backend-facing output of one batched student enrollment."""

    enrollment_id: str
    student_id: str
    full_name: str
    mode: str
    status: str
    submitted_images: int
    accepted_images: int
    rejected_images: int
    image_results: List[EnrollmentImageAIResult]
    embeddings: np.ndarray
    student_template: np.ndarray
    model_name: str
    model_sha256: str
    embedding_dimension: int
    identity_check: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    package_directory: Optional[Path] = None
    package_npz: Optional[Path] = None
    manifest_json: Optional[Path] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.student_template = np.asarray(
            self.student_template, dtype=np.float32
        ).reshape(-1)

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must have shape (N, D).")
        if self.embeddings.shape[1] != int(self.embedding_dimension):
            raise ValueError("embedding_dimension does not match embeddings.")
        if self.student_template.shape != (int(self.embedding_dimension),):
            raise ValueError("student_template has an invalid shape.")
        if self.embeddings.shape[0] != int(self.accepted_images):
            raise ValueError("accepted_images does not match embeddings count.")

    @property
    def ready(self) -> bool:
        return self.status in {"READY", "READY_FOR_REVIEW"}

    def iter_embedding_records(self) -> Iterator[Dict[str, Any]]:
        accepted = [item for item in self.image_results if item.accepted]
        if len(accepted) != len(self.embeddings):
            raise RuntimeError(
                "Accepted image metadata does not match embedding count."
            )

        for item in accepted:
            if item.embedding_index is None:
                raise RuntimeError(
                    "Accepted image is missing its embedding index."
                )
            index = int(item.embedding_index)
            yield {
                "student_id": self.student_id,
                "full_name": self.full_name,
                "enrollment_id": self.enrollment_id,
                "image_path": str(item.source_path),
                "stored_original_path": (
                    str(item.stored_original_path)
                    if item.stored_original_path
                    else None
                ),
                "stored_aligned_path": (
                    str(item.stored_aligned_path)
                    if item.stored_aligned_path
                    else None
                ),
                "image_sha256": item.image_sha256,
                "embedding": self.embeddings[index].copy(),
                "embedding_dimension": self.embedding_dimension,
                "model_name": self.model_name,
                "model_sha256": self.model_sha256,
                "detection_score": item.detection_score,
                "blur_score": item.blur_score,
                "brightness": item.brightness,
                "contrast": item.contrast,
            }

    def to_dict(
        self,
        *,
        include_vectors: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enrollment_id": self.enrollment_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "mode": self.mode,
            "status": self.status,
            "ready": self.ready,
            "submitted_images": self.submitted_images,
            "accepted_images": self.accepted_images,
            "rejected_images": self.rejected_images,
            "embedding_shape": list(self.embeddings.shape),
            "template_shape": list(self.student_template.shape),
            "embedding_dimension": self.embedding_dimension,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
            "identity_check": dict(self.identity_check),
            "warnings": list(self.warnings),
            "package_directory": self.package_directory,
            "package_npz": self.package_npz,
            "manifest_json": self.manifest_json,
            "created_at": self.created_at,
            "images": [item.to_dict() for item in self.image_results],
        }
        if include_vectors:
            payload["embeddings"] = self.embeddings.tolist()
            payload["student_template"] = self.student_template.tolist()
        return _json_safe(payload)

















@dataclass(frozen=True)
class GalleryCacheAIResult(AIContractMixin):
    """Public result of building one immutable gallery-cache version."""

    version_id: str
    status: str
    update_policy: str
    enrollment_id: str
    student_id: str
    full_name: str
    base_gallery_path: Path
    version_directory: Path
    gallery_path: Path
    manifest_json: Path
    base_gallery_sha256: str
    gallery_sha256: str
    base_student_count: int
    new_student_count: int
    base_embedding_count: int
    new_embedding_count: int
    removed_target_embeddings: int
    added_target_embeddings: int
    target_embedding_count: int
    target_template_similarity: float
    warnings: List[str] = field(default_factory=list)
    activated: bool = False
    active_gallery_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "version_id": self.version_id,
            "status": self.status,
            "update_policy": self.update_policy,
            "enrollment_id": self.enrollment_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "base_gallery_path": self.base_gallery_path,
            "version_directory": self.version_directory,
            "gallery_path": self.gallery_path,
            "manifest_json": self.manifest_json,
            "base_gallery_sha256": self.base_gallery_sha256,
            "gallery_sha256": self.gallery_sha256,
            "base_student_count": self.base_student_count,
            "new_student_count": self.new_student_count,
            "base_embedding_count": self.base_embedding_count,
            "new_embedding_count": self.new_embedding_count,
            "removed_target_embeddings": self.removed_target_embeddings,
            "added_target_embeddings": self.added_target_embeddings,
            "target_embedding_count": self.target_embedding_count,
            "target_template_similarity": self.target_template_similarity,
            "warnings": list(self.warnings),
            "activated": self.activated,
            "active_gallery_info": self.active_gallery_info,
            "created_at": self.created_at,
        })



#هذه الداله تعمل على حذف الطالب من الجاليري

@dataclass(frozen=True)
class GalleryStudentRemovalAIResult(
    AIContractMixin
):
    version_id: str
    status: str

    student_id: str
    full_name: str

    base_gallery_path: Path
    gallery_path: Path
    version_directory: Path
    manifest_json: Path

    base_gallery_sha256: str
    gallery_sha256: str

    base_student_count: int
    new_student_count: int

    base_embedding_count: int
    new_embedding_count: int

    removed_embedding_count: int

    activated: bool = False
    active_gallery_info: Optional[
        Dict[str, Any]
    ] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "version_id": self.version_id,
            "status": self.status,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "base_gallery_path": (
                self.base_gallery_path
            ),
            "gallery_path": self.gallery_path,
            "version_directory": (
                self.version_directory
            ),
            "manifest_json": self.manifest_json,
            "base_gallery_sha256": (
                self.base_gallery_sha256
            ),
            "gallery_sha256": (
                self.gallery_sha256
            ),
            "base_student_count": (
                self.base_student_count
            ),
            "new_student_count": (
                self.new_student_count
            ),
            "base_embedding_count": (
                self.base_embedding_count
            ),
            "new_embedding_count": (
                self.new_embedding_count
            ),
            "removed_embedding_count": (
                self.removed_embedding_count
            ),
            "activated": self.activated,
            "active_gallery_info": (
                self.active_gallery_info
            ),
            "created_at": self.created_at,
        })










@dataclass(frozen=True)
class GalleryRebuildAIResult(
    AIContractMixin
):
    version_id: str
    status: str
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

    activated: bool = False

    active_gallery_info: Optional[
        Dict[str, Any]
    ] = None

    created_at: Optional[
        datetime
    ] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "version_id": (
                self.version_id
            ),
            "status": self.status,
            "operation": (
                self.operation
            ),
            "model_name": (
                self.model_name
            ),
            "model_sha256": (
                self.model_sha256
            ),
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "version_directory": (
                self.version_directory
            ),
            "gallery_path": (
                self.gallery_path
            ),
            "gallery_sha256": (
                self.gallery_sha256
            ),
            "manifest_json": (
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
            "activated": (
                self.activated
            ),
            "active_gallery_info": (
                self.active_gallery_info
            ),
            "created_at": (
                self.created_at
            ),
        })





# ---------------------------------------------------------------------------
# Dynamic attendance input and output contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttendanceOutputAIOptions(AIContractMixin):
    """Optional file outputs. The backend may leave all values disabled."""

    save_csv: bool = False
    save_events: bool = False
    save_summary: bool = False
    save_video: bool = False
    output_directory: Optional[Path] = None
    output_prefix: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "save_csv": self.save_csv,
            "save_events": self.save_events,
            "save_summary": self.save_summary,
            "save_video": self.save_video,
            "output_directory": self.output_directory,
            "output_prefix": self.output_prefix,
        })


@dataclass
class AttendanceSessionAIRequest(AIContractMixin):
    """Dynamic request used to create one attendance session.

    ``metadata`` stores business context supplied by the backend. ``extensions``
    is reserved for future AI integration fields, so new optional behavior can
    be added without changing the required constructor fields.
    """

    session_id: Union[str,int]
    class_id: Union[str,int]
    source: SourceValue
    roster_student_ids: List[Union[str, int]]
    course_id: Optional[Union[str, int]] = None
    camera_id: Optional[Union[str, int]] = None
    gallery_path: Optional[Path] = None
    max_frames: int = 0
    max_seconds: float = 0.0
    output: AttendanceOutputAIOptions = field(
        default_factory=AttendanceOutputAIOptions
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = str(self.session_id).strip()
        self.class_id = str(self.class_id).strip()
        if not self.session_id:
            raise ValueError("session_id cannot be empty.")
        if not self.class_id:
            raise ValueError("class_id cannot be empty.")

        if self.course_id is not None:
            self.course_id = str(self.course_id).strip() or None
        if self.camera_id is not None:
            self.camera_id = str(self.camera_id).strip() or None
        if self.gallery_path is not None:
            self.gallery_path = Path(self.gallery_path)

        if isinstance(self.source, Path):
            pass
        elif isinstance(self.source, str):
            self.source = self.source.strip()
            if not self.source:
                raise ValueError("source cannot be empty.")
        elif isinstance(self.source, int):
            if self.source < 0:
                raise ValueError("Camera source index cannot be negative.")
        else:
            raise TypeError("source must be a path, URL, string, or camera index.")

        roster = []
        for student_id in self.roster_student_ids:
            normalized = str(student_id).strip()
            if not normalized:
                raise ValueError("roster_student_ids contains an empty value.")
            roster.append(normalized)
        if not roster:
            raise ValueError("roster_student_ids cannot be empty.")
        if len(roster) != len(set(roster)):
            raise ValueError("roster_student_ids contains duplicates.")
        self.roster_student_ids = roster

        self.max_frames = int(self.max_frames)
        self.max_seconds = float(self.max_seconds)
        if self.max_frames < 0:
            raise ValueError("max_frames cannot be negative.")
        if self.max_seconds < 0:
            raise ValueError("max_seconds cannot be negative.")

        if isinstance(self.output, Mapping):
            self.output = AttendanceOutputAIOptions(**dict(self.output))
        elif not isinstance(self.output, AttendanceOutputAIOptions):
            # Supports the existing internal OutputOptions without importing it.
            self.output = AttendanceOutputAIOptions(
                save_csv=bool(getattr(self.output, "save_csv", False)),
                save_events=bool(getattr(self.output, "save_events", False)),
                save_summary=bool(getattr(self.output, "save_summary", False)),
                save_video=bool(getattr(self.output, "save_video", False)),
                output_directory=getattr(self.output, "output_directory", None),
                output_prefix=getattr(self.output, "output_prefix", None),
            )

        self.metadata = dict(self.metadata or {})
        self.extensions = dict(self.extensions or {})

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "session_id": self.session_id,
            "class_id": self.class_id,
            "course_id": self.course_id,
            "camera_id": self.camera_id,
            "source": self.source,
            "roster_student_ids": list(self.roster_student_ids),
            "gallery_path": self.gallery_path,
            "max_frames": self.max_frames,
            "max_seconds": self.max_seconds,
            "output": self.output.to_dict(),
            "metadata": self.metadata,
            "extensions": self.extensions,
        })


@dataclass(frozen=True)
class AttendanceSessionSnapshotAIResult(AIContractMixin):
    """Live, non-final view of a background attendance session."""

    contract_version: str
    facade_version: str
    session_id: str
    class_id: str
    course_id: Optional[str]
    camera_id: Optional[str]
    status: str
    source_type: str
    source: Any
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    frames_processed: int
    latest_timestamp_seconds: float
    present_student_ids: List[str]
    roster_count: int
    gallery_version_id: Optional[str]
    gallery_path: Optional[Path]
    gallery_sha256: Optional[str]
    error_message: Optional[str]
    result_available: bool
    stop_requested: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @property
    def present_count(self) -> int:
        return len(self.present_student_ids)

    @property
    def terminal(self) -> bool:
        return self.status in {"COMPLETED", "FAILED", "CANCELLED"}

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "contract_version": self.contract_version,
            "facade_version": self.facade_version,
            "session_id": self.session_id,
            "class_id": self.class_id,
            "course_id": self.course_id,
            "camera_id": self.camera_id,
            "status": self.status,
            "terminal": self.terminal,
            "source_type": self.source_type,
            "source": self.source,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "frames_processed": self.frames_processed,
            "latest_timestamp_seconds": self.latest_timestamp_seconds,
            "present_count": self.present_count,
            "present_student_ids": list(self.present_student_ids),
            "roster_count": self.roster_count,
            "gallery_version_id": self.gallery_version_id,
            "gallery_path": self.gallery_path,
            "gallery_sha256": self.gallery_sha256,
            "error_message": self.error_message,
            "result_available": self.result_available,
            "stop_requested": self.stop_requested,
            "metadata": self.metadata,
            "extensions": self.extensions,
        })


@dataclass(frozen=True)
class AttendanceRecordAIResult(AIContractMixin):
    """Final AI attendance decision for one roster student."""

    session_id: str
    student_id: str
    full_name: str
    status: str
    first_seen_seconds: Optional[float]
    confirmed_at_seconds: Optional[float]
    last_detected_seconds: Optional[float]
    last_recognized_seconds: Optional[float]
    best_score: Optional[float]
    best_margin: Optional[float]
    observation_count: int
    track_ids: List[int]
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "session_id": self.session_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "status": self.status,
            "first_seen_seconds": self.first_seen_seconds,
            "confirmed_at_seconds": self.confirmed_at_seconds,
            "last_detected_seconds": self.last_detected_seconds,
            "last_recognized_seconds": self.last_recognized_seconds,
            "best_score": self.best_score,
            "best_margin": self.best_margin,
            "observation_count": self.observation_count,
            "track_ids": list(self.track_ids),
            "extensions": self.extensions,
        })


@dataclass(frozen=True)
class RecognitionEventAIResult(AIContractMixin):
    """One recognition/attendance event generated by the AI engine."""

    event_id: str
    event_type: str
    session_id: str
    student_id: str
    full_name: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    score: float
    margin: float
    created_at: datetime
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "event_id": self.event_id,
            "event_type": self.event_type,
            "session_id": self.session_id,
            "student_id": self.student_id,
            "full_name": self.full_name,
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp_seconds": self.timestamp_seconds,
            "score": self.score,
            "margin": self.margin,
            "created_at": self.created_at,
            "extensions": self.extensions,
        })


@dataclass(frozen=True)
class AttendanceMetricsAIResult(AIContractMixin):
    """Processing metrics returned with the final video result."""

    frames_processed: int
    detection_calls: int
    recognition_calls: int
    recognition_faces: int
    tracks_created: int
    source_fps: float
    processing_seconds: float
    processing_fps: float
    average_detection_ms: float
    average_recognition_ms: float
    extensions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "frames_processed": self.frames_processed,
            "detection_calls": self.detection_calls,
            "recognition_calls": self.recognition_calls,
            "recognition_faces": self.recognition_faces,
            "tracks_created": self.tracks_created,
            "source_fps": self.source_fps,
            "processing_seconds": self.processing_seconds,
            "processing_fps": self.processing_fps,
            "average_detection_ms": self.average_detection_ms,
            "average_recognition_ms": self.average_recognition_ms,
            "extensions": self.extensions,
        })


@dataclass(frozen=True)
class AttendanceSessionAIResult(AIContractMixin):
    """Final backend-facing result for one processed video/stream session."""

    contract_version: str
    facade_version: str
    session_id: str
    class_id: str
    course_id: Optional[str]
    camera_id: Optional[str]
    status: str
    source_type: str
    source: Any
    roster_student_ids: List[str]
    present_student_ids: List[str]
    absent_student_ids: List[str]
    attendance: List[AttendanceRecordAIResult]
    events: List[RecognitionEventAIResult]
    metrics: AttendanceMetricsAIResult
    gallery_version_id: Optional[str]
    gallery_path: Optional[Path]
    gallery_sha256: Optional[str]
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @property
    def present_count(self) -> int:
        return len(self.present_student_ids)

    @property
    def absent_count(self) -> int:
        return len(self.absent_student_ids)

    @property
    def succeeded(self) -> bool:
        return self.status == "COMPLETED" and self.error_message is None

    def get_student(self, student_id: str) -> Optional[AttendanceRecordAIResult]:
        normalized = str(student_id).strip()
        for record in self.attendance:
            if record.student_id == normalized:
                return record
        return None

    def iter_attendance_records(self) -> Iterator[Dict[str, Any]]:
        """Yield one DB-friendly dictionary per roster student."""
        for record in self.attendance:
            payload = record.to_dict()
            payload.update({
                "class_id": self.class_id,
                "course_id": self.course_id,
                "camera_id": self.camera_id,
                "gallery_version_id": self.gallery_version_id,
                "gallery_sha256": self.gallery_sha256,
            })
            yield payload

    def iter_event_records(self) -> Iterator[Dict[str, Any]]:
        """Yield one DB-friendly dictionary per recognition event."""
        for event in self.events:
            payload = event.to_dict()
            payload.update({
                "class_id": self.class_id,
                "course_id": self.course_id,
                "camera_id": self.camera_id,
                "gallery_version_id": self.gallery_version_id,
            })
            yield payload

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe({
            "contract_version": self.contract_version,
            "facade_version": self.facade_version,
            "session_id": self.session_id,
            "class_id": self.class_id,
            "course_id": self.course_id,
            "camera_id": self.camera_id,
            "status": self.status,
            "succeeded": self.succeeded,
            "source_type": self.source_type,
            "source": self.source,
            "roster_student_ids": list(self.roster_student_ids),
            "roster_count": len(self.roster_student_ids),
            "present_count": self.present_count,
            "absent_count": self.absent_count,
            "present_student_ids": list(self.present_student_ids),
            "absent_student_ids": list(self.absent_student_ids),
            "attendance": [item.to_dict() for item in self.attendance],
            "events": [item.to_dict() for item in self.events],
            "metrics": self.metrics.to_dict(),
            "gallery_version_id": self.gallery_version_id,
            "gallery_path": self.gallery_path,
            "gallery_sha256": self.gallery_sha256,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "extensions": self.extensions,
        })