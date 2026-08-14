"""Stable integration facade for the Smart Attendance AI subsystem.

The backend should call this class instead of importing the detector,
recognizer, enrollment service, gallery service, or session manager directly.
There is deliberately no FastAPI or database dependency in this module.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import numpy as np

from smart_attendance_ai.ai_contracts import (
    AISettings,
    AttendanceMetricsAIResult,
    AttendanceOutputAIOptions,
    AttendanceRecordAIResult,
    AttendanceSessionAIRequest,
    AttendanceSessionAIResult,
    AttendanceSessionSnapshotAIResult,
    EnrollmentAIResult,
    EnrollmentImageAIResult,
    GalleryCacheAIResult,
    RecognitionEventAIResult,
    GalleryStudentRemovalAIResult,
    GalleryRebuildAIResult,
)
from smart_attendance_ai.attendance_schemas import (
    OutputOptions,
    SessionConfig,
    SessionResult,
)
from smart_attendance_ai.attendance_service import AttendanceService
from smart_attendance_ai.enrollment_schemas import EnrollmentMode
from smart_attendance_ai.enrollment_service import EnrollmentService
from smart_attendance_ai.gallery_schemas import GalleryUpdatePolicy
from smart_attendance_ai.gallery_version_service import GalleryVersionService
from smart_attendance_ai.session_manager import AttendanceSessionManager


Source = Union[str, int, Path]
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


class AIFacadeError(RuntimeError):
    """Base error raised by the stable AI integration layer."""

class AIOperationConflictError(AIFacadeError):
    """Raised when two GPU operations are not safe to run together."""


class SmartAttendanceAI:
    """
    Backend-facing facade for enrollment, gallery caches, and attendance.

    Design goals:
    - one stable public integration point;
    - no FastAPI or database dependency;
    - detector and recognizer models are loaded once and shared;
    - implementation services can be replaced by dependency injection;
    - NPZ remains a runtime cache, while the caller may store vectors in DB.
    """

    FACADE_VERSION = "1.1"
    ATTENDANCE_CONTRACT_VERSION = "1.0"

    def __init__(
        self,
        project_root: Optional[Path] = None,
        *,
        settings: Optional[AISettings] = None,
        ai_config_path: Optional[Path] = None,
        attendance_service: Optional[AttendanceService] = None,
        enrollment_service: Optional[EnrollmentService] = None,
        gallery_service: Optional[GalleryVersionService] = None,
        session_manager: Optional[AttendanceSessionManager] = None,
    ) -> None:
        if settings is None:
            if project_root is None:
                raise ValueError("project_root or settings is required.")
            settings = AISettings(
                project_root=Path(project_root),
                ai_config_path=ai_config_path,
            )
        elif project_root is not None or ai_config_path is not None:
            raise ValueError(
                "Use either settings or project_root/ai_config_path, not both."
            )

        self.settings = settings.normalized()
        self.root = self.settings.project_root
        self._operation_lock = threading.RLock()
        self._closed = False
        self._attendance_contexts: Dict[str, Dict[str, Any]] = {}

        # Reuse the service already owned by an injected manager when possible.
        if attendance_service is None and session_manager is not None:
            attendance_service = session_manager.service

        self.attendance_service = attendance_service or AttendanceService(
            root=self.root,
            ai_config_path=self.settings.ai_config_path,
        )

        # Enrollment reuses the exact same long-lived engine/models.
        self.enrollment_service = enrollment_service or EnrollmentService(
            root=self.root,
            engine=self.attendance_service.engine,
            staging_root=self.settings.enrollment_staging_root,
        )

        self.gallery_service = gallery_service or GalleryVersionService(
            root=self.root,
            versions_root=self.settings.gallery_versions_root,
            active_pointer_path=self.settings.active_gallery_pointer_path,
        )

        self.session_manager = session_manager or AttendanceSessionManager(
            root=self.root,
            service=self.attendance_service,
        )

    # ------------------------------------------------------------------
    # Runtime and active-gallery information
    # ------------------------------------------------------------------

    def get_health(
        self,
    ) -> Dict[str, Any]:
        """
        Return AI runtime health without requiring
        a Gallery to already exist.
        """

        self._ensure_open()

        runtime = dict(
            self.attendance_service
            .engine
            .get_runtime_info()
        )

        pointer_exists = (
            self.settings
            .active_gallery_pointer_path
            .is_file()
        )

        if pointer_exists:
            # If a pointer exists but is corrupt,
            # propagate the error. Do not hide it.
            active_gallery = (
                self.get_active_gallery_info()
            )

        else:
            try:
                active_gallery = (
                    self.get_active_gallery_info()
                )
            except FileNotFoundError:
                active_gallery = None

        return {
            "facade_version": (
                self.FACADE_VERSION
            ),
            "status": (
                "READY"
                if runtime.get(
                    "models_loaded"
                )
                else "NOT_READY"
            ),
            "gallery_ready": (
                active_gallery is not None
            ),
            "active_session_id": (
                self.session_manager
                .active_session_id
            ),
            "active_gallery": (
                active_gallery
            ),
            "runtime": runtime,
        }
    
    def get_active_gallery_path(self) -> Path:
        """Resolve the versioned active Gallery or the legacy fallback."""
        self._ensure_open()
        return self.gallery_service.get_active_gallery_path(
            fallback_path=self.settings.legacy_gallery_path
        )

    def get_active_gallery_info(self) -> Dict[str, Any]:
        """Return the validated active Gallery pointer and file hash."""
        self._ensure_open()
        gallery_path = self.get_active_gallery_path()
        actual_hash = self._sha256_file(gallery_path)
        pointer = self.settings.active_gallery_pointer_path

        if pointer.is_file():
            with pointer.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            return {
                "source": "VERSIONED_CACHE",
                "active_version_id": payload.get("active_version_id"),
                "gallery_path": str(gallery_path),
                "gallery_sha256": actual_hash,
                "activated_at": payload.get("activated_at"),
            }

        return {
            "source": "LEGACY_FALLBACK",
            "active_version_id": None,
            "gallery_path": str(gallery_path),
            "gallery_sha256": actual_hash,
            "activated_at": None,
        }

    # ------------------------------------------------------------------
    # Batched student enrollment
    # ------------------------------------------------------------------

    def prepare_student_enrollment(
        self,
        *,
        student_id: str,
        full_name: str,
        image_paths: Iterable[Union[str, Path]],
        mode: Union[str, EnrollmentMode] = EnrollmentMode.NEW,
        enrollment_id: Optional[str] = None,
        minimum_accepted_images: Optional[int] = None,
        active_gallery_path: Optional[Path] = None,
    ) -> EnrollmentAIResult:
        """
        Process all student images in one call and return DB-ready vectors.

        This method never changes or activates a Gallery cache.
        """
        self._ensure_open()
        normalized_student_id = self._validate_id(student_id, "student_id")
        normalized_name = str(full_name).strip()
        if not normalized_name:
            raise ValueError("full_name cannot be empty.")

        normalized_mode = (
            mode if isinstance(mode, EnrollmentMode) else EnrollmentMode(str(mode).upper())
        )
        normalized_enrollment_id = self._validate_id(
            enrollment_id or self._new_enrollment_id(normalized_student_id),
            "enrollment_id",
        )
        paths = [Path(value) for value in image_paths]
        if not paths:
            raise ValueError("image_paths cannot be empty.")

        minimum = int(
            minimum_accepted_images
            if minimum_accepted_images is not None
            else self.settings.minimum_enrollment_images
        )
        if minimum < 1:
            raise ValueError("minimum_accepted_images must be at least 1.")

        with self._operation_lock:
            self._assert_no_active_attendance("student enrollment")
            gallery_path = (
                self._resolve_user_path(active_gallery_path)
                if active_gallery_path is not None
                else self.get_active_gallery_path()
            )
            internal_result = self.enrollment_service.prepare_enrollment(
                enrollment_id=normalized_enrollment_id,
                student_id=normalized_student_id,
                full_name=normalized_name,
                image_paths=paths,
                active_gallery_path=gallery_path,
                mode=normalized_mode,
                minimum_accepted_images=minimum,
            )
            return self._load_enrollment_contract(internal_result)

    # A shorter alias is convenient for a backend service.
    enroll_student = prepare_student_enrollment

    # ------------------------------------------------------------------
    # Versioned NPZ runtime cache
    # ------------------------------------------------------------------

    def build_gallery_version(
        self,
        *,
        enrollment: Union[EnrollmentAIResult, str, Path],
        update_policy: Union[str, GalleryUpdatePolicy] = GalleryUpdatePolicy.APPEND,
        version_id: Optional[str] = None,
        base_gallery_path: Optional[Path] = None,
        approve_reviewed_package: bool = False,
        activate_after_build: bool = False,
        activate_reviewed_version: bool = False,
    ) -> GalleryCacheAIResult:
        """Build an immutable Gallery cache version from an enrollment package."""
        self._ensure_open()
        package_path = self._package_path(enrollment)
        policy = (
            update_policy
            if isinstance(update_policy, GalleryUpdatePolicy)
            else GalleryUpdatePolicy(str(update_policy).upper())
        )

        package_metadata = self._read_package_metadata(package_path)
        normalized_version_id = self._validate_id(
            version_id
            or self._new_gallery_version_id(package_metadata["student_id"]),
            "version_id",
        )
        base_path = (
            self._resolve_user_path(base_gallery_path)
            if base_gallery_path is not None
            else self.get_active_gallery_path()
        )

        with self._operation_lock:
            internal = self.gallery_service.build_version_from_package(
                version_id=normalized_version_id,
                base_gallery_path=base_path,
                enrollment_package_path=package_path,
                update_policy=policy,
                approve_reviewed_package=approve_reviewed_package,
            )

            activated = False
            active_info: Optional[Dict[str, Any]] = None
            if activate_after_build:
                has_review_warning = str(internal.status.value) == "READY_FOR_REVIEW"
                if has_review_warning and not activate_reviewed_version:
                    raise AIFacadeError(
                        "Gallery version was built but not activated because it "
                        "requires review. Pass activate_reviewed_version=True only "
                        "after regression testing."
                    )
                active_info = self.activate_gallery_version(internal.version_id)
                activated = True

            return GalleryCacheAIResult(
                version_id=internal.version_id,
                status=internal.status.value,
                update_policy=internal.update_policy.value,
                enrollment_id=internal.enrollment_id,
                student_id=internal.student_id,
                full_name=internal.full_name,
                base_gallery_path=Path(internal.base_gallery_path),
                version_directory=Path(internal.version_directory),
                gallery_path=Path(internal.gallery_path),
                manifest_json=Path(internal.manifest_json),
                base_gallery_sha256=internal.base_gallery_sha256,
                gallery_sha256=internal.gallery_sha256,
                base_student_count=internal.base_student_count,
                new_student_count=internal.new_student_count,
                base_embedding_count=internal.base_embedding_count,
                new_embedding_count=internal.new_embedding_count,
                removed_target_embeddings=internal.removed_target_embeddings,
                added_target_embeddings=internal.added_target_embeddings,
                target_embedding_count=internal.target_embedding_count,
                target_template_similarity=internal.target_template_similarity,
                warnings=list(internal.warnings),
                activated=activated,
                active_gallery_info=active_info,
                created_at=internal.created_at,
            )

    publish_student_to_gallery = build_gallery_version

    def activate_gallery_version(self, version_id: str) -> Dict[str, Any]:
        """Atomically activate a tested Gallery cache version."""
        self._ensure_open()
        normalized = self._validate_id(version_id, "version_id")
        with self._operation_lock:
            self._assert_no_active_attendance("Gallery activation")
            return dict(self.gallery_service.activate_version(normalized))


    def remove_student_from_gallery(
        self,
        *,
        student_id: Union[str, int],
        version_id: Optional[str] = None,
        base_gallery_path: Optional[Path] = None,
        activate_after_build: bool = False,
    ) -> GalleryStudentRemovalAIResult:
        """
        Build a new Gallery version without one student.

        The active Gallery remains unchanged unless
        activate_after_build=True.
        """

        self._ensure_open()

        normalized_student_id = self._validate_id(
            student_id,
            "student_id",
        )

        if version_id is None:
            timestamp = datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%S"
            )

            version_id = (
                f"GAL_REMOVE_"
                f"{normalized_student_id}_"
                f"{timestamp}_"
                f"{uuid.uuid4().hex[:8]}"
            )

        normalized_version_id = self._validate_id(
            version_id,
            "version_id",
        )

        resolved_base = (
            self._resolve_user_path(
                base_gallery_path
            )
            if base_gallery_path is not None
            else self.get_active_gallery_path()
        )

        with self._operation_lock:
            self._assert_no_active_attendance(
                "student gallery removal"
            )

            internal = (
                self.gallery_service
                .build_version_without_student(
                    version_id=(
                        normalized_version_id
                    ),
                    base_gallery_path=(
                        resolved_base
                    ),
                    student_id=(
                        normalized_student_id
                    ),
                )
            )

            activated = False
            active_info = None

            if activate_after_build:
                active_info = (
                    self.activate_gallery_version(
                        internal.version_id
                    )
                )

                activated = True

            return GalleryStudentRemovalAIResult(
                version_id=internal.version_id,
                status=internal.status.value,
                student_id=internal.student_id,
                full_name=internal.full_name,
                base_gallery_path=(
                    internal.base_gallery_path
                ),
                gallery_path=internal.gallery_path,
                version_directory=(
                    internal.version_directory
                ),
                manifest_json=(
                    internal.manifest_json
                ),
                base_gallery_sha256=(
                    internal.base_gallery_sha256
                ),
                gallery_sha256=(
                    internal.gallery_sha256
                ),
                base_student_count=(
                    internal.base_student_count
                ),
                new_student_count=(
                    internal.new_student_count
                ),
                base_embedding_count=(
                    internal.base_embedding_count
                ),
                new_embedding_count=(
                    internal.new_embedding_count
                ),
                removed_embedding_count=(
                    internal.removed_embedding_count
                ),
                activated=activated,
                active_gallery_info=active_info,
                created_at=internal.created_at,
            )


    def rebuild_gallery_from_records(
        self,
        *,
        records: Iterable[
            Mapping[str, Any]
        ],
        version_id: Optional[
            str
        ] = None,
        activate_after_build: bool = False,
    ) -> GalleryRebuildAIResult:
        """
        Rebuild the entire runtime Gallery from
        caller-supplied embedding records.

        The caller may use PostgreSQL as the
        source of truth. This method has no
        database dependency.
        """

        self._ensure_open()

        source_records = list(
            records
        )

        if version_id is None:
            timestamp = datetime.now(
                timezone.utc
            ).strftime(
                "%Y%m%dT%H%M%S"
            )

            operation_name = (
                "BOOTSTRAP"
                if not source_records
                else "REBUILD"
            )

            version_id = (
                f"GAL_{operation_name}_"
                f"{timestamp}_"
                f"{uuid.uuid4().hex[:8]}"
            )

        normalized_version_id = (
            self._validate_id(
                version_id,
                "version_id",
            )
        )

        with self._operation_lock:
            self._assert_no_active_attendance(
                "Gallery rebuild"
            )

            metadata = (
                self._recognizer_gallery_metadata()
            )

            internal = (
                self.gallery_service
                .build_version_from_records(
                    version_id=(
                        normalized_version_id
                    ),
                    records=source_records,
                    model_name=(
                        metadata["model_name"]
                    ),
                    model_sha256=(
                        metadata[
                            "model_sha256"
                        ]
                    ),
                    embedding_dimension=(
                        metadata[
                            "embedding_dimension"
                        ]
                    ),
                )
            )

            activated = False
            active_info = None

            if activate_after_build:
                active_info = dict(
                    self.gallery_service
                    .activate_version(
                        internal.version_id
                    )
                )

                activated = True

            return GalleryRebuildAIResult(
                version_id=(
                    internal.version_id
                ),
                status=(
                    internal.status.value
                ),
                operation=(
                    internal.operation
                ),
                model_name=(
                    internal.model_name
                ),
                model_sha256=(
                    internal.model_sha256
                ),
                embedding_dimension=(
                    internal.embedding_dimension
                ),
                version_directory=(
                    internal.version_directory
                ),
                gallery_path=(
                    internal.gallery_path
                ),
                gallery_sha256=(
                    internal.gallery_sha256
                ),
                manifest_json=(
                    internal.manifest_json
                ),
                student_count=(
                    internal.student_count
                ),
                embedding_count=(
                    internal.embedding_count
                ),
                warnings=list(
                    internal.warnings
                ),
                activated=activated,
                active_gallery_info=(
                    active_info
                ),
                created_at=(
                    internal.created_at
                ),
            )

    def bootstrap_empty_gallery(
        self,
        *,
        version_id: Optional[
            str
        ] = None,
        activate_after_build: bool = True,
    ) -> GalleryRebuildAIResult:
        """
        Create the first valid zero-student Gallery.

        Safety rule:
        bootstrap is allowed only when no active
        or legacy Gallery currently resolves.
        """

        self._ensure_open()

        with self._operation_lock:
            self._assert_no_active_attendance(
                "Gallery bootstrap"
            )

            try:
                existing_gallery = (
                    self.get_active_gallery_path()
                )
            except FileNotFoundError:
                existing_gallery = None

            if existing_gallery is not None:
                raise AIFacadeError(
                    "Gallery bootstrap refused because "
                    "a Gallery already exists: "
                    f"{existing_gallery}. "
                    "Use rebuild_gallery_from_records() "
                    "when replacing an existing Gallery."
                )

            return (
                self.rebuild_gallery_from_records(
                    records=[],
                    version_id=version_id,
                    activate_after_build=(
                        activate_after_build
                    ),
                )
            )

    def _recognizer_gallery_metadata(
        self,
    ) -> Dict[str, Any]:
        engine = (
            self.attendance_service.engine
        )

        engine.load_models()

        if engine.recognizer is None:
            raise AIFacadeError(
                "Recognizer is not loaded."
            )

        model_path = (
            self._resolve_user_path(
                engine.config[
                    "models"
                ][
                    "recognizer"
                ][
                    "path"
                ]
            )
        )

        if not model_path.is_file():
            raise FileNotFoundError(
                "Recognizer model does not "
                f"exist: {model_path}"
            )

        expected_hash = str(
            engine.config[
                "models"
            ][
                "recognizer"
            ][
                "sha256"
            ]
        ).upper()

        actual_hash = (
            self._sha256_file(
                model_path
            ).upper()
        )

        if (
            expected_hash
            != actual_hash
        ):
            raise AIFacadeError(
                "Recognizer model SHA256 "
                "does not match ai.yaml."
            )

        model_name = str(
            engine.config[
                "recognizer"
            ][
                "model_name"
            ]
        ).strip()

        if not model_name:
            raise AIFacadeError(
                "Recognizer model_name "
                "is empty."
            )

        dimension = int(
            engine.recognizer
            .embedding_size
        )

        if dimension < 1:
            raise AIFacadeError(
                "Invalid recognizer "
                "embedding dimension."
            )

        return {
            "model_name": model_name,
            "model_sha256": (
                actual_hash
            ),
            "embedding_dimension": (
                dimension
            ),
            "model_path": model_path,
        }
    # ------------------------------------------------------------------
    # Background attendance API
    # ------------------------------------------------------------------

    def create_attendance_session(
        self,
        request: Optional[AttendanceSessionAIRequest] = None,
        **kwargs: Any,
    ) -> AttendanceSessionSnapshotAIResult:
        """Register a fully dynamic attendance session without starting it.

        Preferred usage passes ``AttendanceSessionAIRequest``. Keyword arguments
        remain supported for backwards compatibility with facade version 1.0.
        """
        self._ensure_open()
        public_request = self._coerce_attendance_request(request, kwargs)

        resolved_gallery = (
            self._resolve_user_path(public_request.gallery_path)
            if public_request.gallery_path is not None
            else self.get_active_gallery_path()
        )

        normalized_source: Union[str, int]
        if isinstance(public_request.source, Path):
            normalized_source = str(public_request.source)
        else:
            normalized_source = public_request.source

        output_directory = (
            public_request.output.output_directory
            if public_request.output.output_directory is not None
            else self.root / "outputs"
        )
        output = OutputOptions(
            save_csv=public_request.output.save_csv,
            save_events=public_request.output.save_events,
            save_summary=public_request.output.save_summary,
            save_video=public_request.output.save_video,
            output_directory=self._resolve_user_path(output_directory),
            output_prefix=(
                public_request.output.output_prefix
                or public_request.session_id
            ),
        )

        config = SessionConfig(
            session_id=public_request.session_id,
            class_id=public_request.class_id,
            course_id=public_request.course_id,
            camera_id=public_request.camera_id,
            source=normalized_source,
            gallery_path=resolved_gallery,
            roster_student_ids=list(public_request.roster_student_ids),
            output=output,
            metadata=dict(public_request.metadata),
        )

        snapshot = self.session_manager.create_session(
            config,
            max_frames=public_request.max_frames,
            max_seconds=public_request.max_seconds,
        )

        gallery_info = self._describe_gallery(resolved_gallery)
        self._attendance_contexts[public_request.session_id] = {
            "request": public_request,
            "source_type": self._source_type(public_request.source),
            "public_source": self._public_source(public_request.source),
            "gallery": gallery_info,
            "metadata": dict(public_request.metadata),
            "extensions": dict(public_request.extensions),
        }
        return self._snapshot_contract(snapshot)

    def start_registered_attendance_session(
        self,
        session_id: str,
    ) -> AttendanceSessionSnapshotAIResult:
        """Start a previously created session and return immediately."""
        self._ensure_open()
        with self._operation_lock:
            snapshot = self.session_manager.start_session(session_id)
        return self._snapshot_contract(snapshot)

    def start_attendance(
        self,
        request: Optional[AttendanceSessionAIRequest] = None,
        **kwargs: Any,
    ) -> AttendanceSessionSnapshotAIResult:
        """Create and immediately start one background attendance session."""
        created = self.create_attendance_session(request, **kwargs)
        return self.start_registered_attendance_session(created.session_id)

    def get_attendance_status(
        self,
        session_id: str,
    ) -> AttendanceSessionSnapshotAIResult:
        self._ensure_open()
        return self._snapshot_contract(
            self.session_manager.get_session(session_id)
        )

    def list_attendance_sessions(self) -> List[AttendanceSessionSnapshotAIResult]:
        self._ensure_open()
        return [
            self._snapshot_contract(item)
            for item in self.session_manager.list_sessions()
        ]

    def stop_attendance(
        self,
        session_id: str,
    ) -> AttendanceSessionSnapshotAIResult:
        self._ensure_open()
        return self._snapshot_contract(
            self.session_manager.stop_session(session_id)
        )

    def wait_for_attendance(
        self,
        session_id: str,
        *,
        timeout: Optional[float] = None,
    ) -> AttendanceSessionSnapshotAIResult:
        """CLI/testing helper; production callers normally poll status."""
        self._ensure_open()
        return self._snapshot_contract(
            self.session_manager.wait_for_terminal(
                session_id,
                timeout=timeout,
            )
        )

    def get_attendance_result(
        self,
        session_id: str,
    ) -> Optional[AttendanceSessionAIResult]:
        """Return the final typed result, or ``None`` before completion."""
        self._ensure_open()
        result = self.session_manager.get_result(session_id)
        if result is None:
            return None
        return self._session_result_to_contract(result)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        self.session_manager.shutdown(wait=wait)

    def __enter__(self) -> "SmartAttendanceAI":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Internal adapters. Backend code should not call these methods.
    # ------------------------------------------------------------------

    def _load_enrollment_contract(self, result: Any) -> EnrollmentAIResult:
        package_path = Path(result.package_npz)
        with np.load(package_path, allow_pickle=False) as package:
            embeddings = package["image_embeddings"].astype(np.float32).copy()
            template = package["student_template"].astype(np.float32).copy()
            model_name = self._np_scalar(package["model_name"])
            model_hash = self._np_scalar(package["model_sha256"]).upper()
            dimension = int(np.asarray(package["embedding_dimension"]).item())

        public_images: List[EnrollmentImageAIResult] = []
        embedding_index = 0
        for item in result.image_results:
            current_index: Optional[int] = None
            if item.accepted:
                current_index = embedding_index
                embedding_index += 1
            public_images.append(
                EnrollmentImageAIResult(
                    source_path=Path(item.source_path),
                    original_filename=item.original_filename,
                    accepted=bool(item.accepted),
                    reason=item.reason,
                    image_sha256=item.image_sha256,
                    embedding_index=current_index,
                    stored_original_path=(
                        Path(item.stored_original_path)
                        if item.stored_original_path
                        else None
                    ),
                    stored_aligned_path=(
                        Path(item.stored_aligned_path)
                        if item.stored_aligned_path
                        else None
                    ),
                    detection_score=item.detection_score,
                    face_width=item.face_width,
                    face_height=item.face_height,
                    blur_score=item.blur_score,
                    brightness=item.brightness,
                    contrast=item.contrast,
                    dark_ratio=item.dark_ratio,
                    bright_ratio=item.bright_ratio,
                )
            )

        if embedding_index != len(embeddings):
            raise AIFacadeError(
                "Enrollment package embeddings do not match accepted images."
            )

        return EnrollmentAIResult(
            enrollment_id=result.enrollment_id,
            student_id=result.student_id,
            full_name=result.full_name,
            mode=result.mode.value,
            status=result.status.value,
            submitted_images=result.submitted_images,
            accepted_images=result.accepted_images,
            rejected_images=result.rejected_images,
            image_results=public_images,
            embeddings=embeddings,
            student_template=template,
            model_name=model_name,
            model_sha256=model_hash,
            embedding_dimension=dimension,
            identity_check=result.identity_check.to_dict(),
            warnings=list(result.warnings),
            package_directory=Path(result.package_directory),
            package_npz=package_path,
            manifest_json=Path(result.manifest_json),
            created_at=result.created_at,
        )

    def _snapshot_contract(self, snapshot: Any) -> AttendanceSessionSnapshotAIResult:
        context = self._attendance_context(snapshot.session_id, snapshot)
        gallery = context["gallery"]
        request = context.get("request")
        roster_count = (
            len(request.roster_student_ids)
            if isinstance(request, AttendanceSessionAIRequest)
            else 0
        )
        return AttendanceSessionSnapshotAIResult(
            contract_version=self.ATTENDANCE_CONTRACT_VERSION,
            facade_version=self.FACADE_VERSION,
            session_id=snapshot.session_id,
            class_id=snapshot.class_id,
            course_id=snapshot.course_id,
            camera_id=snapshot.camera_id,
            status=snapshot.status.value,
            source_type=context["source_type"],
            source=context["public_source"],
            created_at=snapshot.created_at,
            started_at=snapshot.started_at,
            finished_at=snapshot.finished_at,
            frames_processed=snapshot.frames_processed,
            latest_timestamp_seconds=snapshot.latest_timestamp_seconds,
            present_student_ids=list(snapshot.present_student_ids),
            roster_count=roster_count,
            gallery_version_id=gallery.get("active_version_id"),
            gallery_path=(
                Path(gallery["gallery_path"])
                if gallery.get("gallery_path")
                else None
            ),
            gallery_sha256=gallery.get("gallery_sha256"),
            error_message=snapshot.error_message,
            result_available=snapshot.result_available,
            stop_requested=snapshot.stop_requested,
            metadata=dict(context.get("metadata", {})),
            extensions=dict(context.get("extensions", {})),
        )

    def _session_result_to_contract(
        self,
        result: SessionResult,
    ) -> AttendanceSessionAIResult:
        context = self._attendance_context(result.session_id)
        gallery = context["gallery"]
        request = context.get("request")

        attendance = [
            AttendanceRecordAIResult(
                session_id=record.session_id,
                student_id=record.student_id,
                full_name=record.full_name,
                status=record.status.value,
                first_seen_seconds=record.first_seen_seconds,
                confirmed_at_seconds=record.confirmed_at_seconds,
                last_detected_seconds=record.last_detected_seconds,
                last_recognized_seconds=record.last_recognized_seconds,
                best_score=record.best_score,
                best_margin=record.best_margin,
                observation_count=record.observation_count,
                track_ids=sorted(record.track_ids),
            )
            for record in result.attendance
        ]

        events = [
            RecognitionEventAIResult(
                event_id=event.event_id,
                event_type=event.event_type,
                session_id=event.session_id,
                student_id=event.student_id,
                full_name=event.full_name,
                track_id=event.track_id,
                frame_index=event.frame_index,
                timestamp_seconds=event.timestamp_seconds,
                score=event.score,
                margin=event.margin,
                created_at=event.created_at,
            )
            for event in result.events
        ]

        metrics = AttendanceMetricsAIResult(
            frames_processed=result.metrics.frames_processed,
            detection_calls=result.metrics.detection_calls,
            recognition_calls=result.metrics.recognition_calls,
            recognition_faces=result.metrics.recognition_faces,
            tracks_created=result.metrics.tracks_created,
            source_fps=result.metrics.source_fps,
            processing_seconds=result.metrics.processing_seconds,
            processing_fps=result.metrics.processing_fps,
            average_detection_ms=result.metrics.average_detection_ms,
            average_recognition_ms=result.metrics.average_recognition_ms,
        )

        if isinstance(request, AttendanceSessionAIRequest):
            roster_student_ids = list(request.roster_student_ids)
            course_id = request.course_id
            camera_id = request.camera_id
        else:
            roster_student_ids = [record.student_id for record in attendance]
            course_id = None
            camera_id = None

        return AttendanceSessionAIResult(
            contract_version=self.ATTENDANCE_CONTRACT_VERSION,
            facade_version=self.FACADE_VERSION,
            session_id=result.session_id,
            class_id=result.class_id,
            course_id=course_id,
            camera_id=camera_id,
            status=result.status.value,
            source_type=context["source_type"],
            source=context["public_source"],
            roster_student_ids=roster_student_ids,
            present_student_ids=list(result.present_student_ids),
            absent_student_ids=list(result.absent_student_ids),
            attendance=attendance,
            events=events,
            metrics=metrics,
            gallery_version_id=gallery.get("active_version_id"),
            gallery_path=(
                Path(gallery["gallery_path"])
                if gallery.get("gallery_path")
                else None
            ),
            gallery_sha256=gallery.get("gallery_sha256"),
            started_at=result.started_at,
            finished_at=result.finished_at,
            error_message=result.error_message,
            metadata=dict(context.get("metadata", {})),
            extensions=dict(context.get("extensions", {})),
        )

    def _coerce_attendance_request(
        self,
        request: Optional[AttendanceSessionAIRequest],
        kwargs: Dict[str, Any],
    ) -> AttendanceSessionAIRequest:
        if request is not None and kwargs:
            raise ValueError(
                "Pass either AttendanceSessionAIRequest or keyword arguments, not both."
            )
        if isinstance(request, AttendanceSessionAIRequest):
            return request
        if isinstance(request, dict):
            return AttendanceSessionAIRequest(**dict(request))
        if request is not None:
            raise TypeError(
                "request must be AttendanceSessionAIRequest, dict, or None."
            )

        values = dict(kwargs)
        if "output_options" in values and "output" not in values:
            values["output"] = values.pop("output_options")
        return AttendanceSessionAIRequest(**values)

    def _attendance_context(
        self,
        session_id: str,
        snapshot: Optional[Any] = None,
    ) -> Dict[str, Any]:
        existing = self._attendance_contexts.get(str(session_id))
        if existing is not None:
            return existing

        source = snapshot.source if snapshot is not None else ""
        fallback_gallery = self.get_active_gallery_info()
        return {
            "request": None,
            "source_type": self._source_type(source),
            "public_source": self._public_source(source),
            "gallery": fallback_gallery,
            "metadata": {},
            "extensions": {},
        }

    def _describe_gallery(self, gallery_path: Path) -> Dict[str, Any]:
        resolved = Path(gallery_path).resolve()
        gallery_hash = self._sha256_file(resolved)
        active = self.get_active_gallery_info()
        active_path = Path(active["gallery_path"]).resolve()
        if resolved == active_path:
            return dict(active)

        version_id: Optional[str] = None
        try:
            with np.load(resolved, allow_pickle=False) as gallery:
                if "gallery_version_id" in gallery.files:
                    version_id = self._np_scalar(gallery["gallery_version_id"])
        except Exception:
            version_id = None

        return {
            "source": "CUSTOM_GALLERY",
            "active_version_id": version_id,
            "gallery_path": str(resolved),
            "gallery_sha256": gallery_hash,
            "activated_at": None,
        }

    @staticmethod
    def _source_type(source: Any) -> str:
        if isinstance(source, int):
            return "CAMERA_INDEX"
        value = str(source).strip()
        lowered = value.lower()
        if lowered.startswith("rtsp://"):
            return "RTSP_STREAM"
        if lowered.startswith("http://") or lowered.startswith("https://"):
            return "HTTP_STREAM"
        return "VIDEO_FILE"

    @staticmethod
    def _public_source(source: Any) -> Any:
        if isinstance(source, int):
            return source
        value = str(source)
        lowered = value.lower()
        if not (
            lowered.startswith("rtsp://")
            or lowered.startswith("http://")
            or lowered.startswith("https://")
        ):
            return value

        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit(
            (parsed.scheme, host, parsed.path, parsed.query, parsed.fragment)
        )

    def _package_path(self, enrollment: Union[EnrollmentAIResult, str, Path]) -> Path:
        if isinstance(enrollment, EnrollmentAIResult):
            if enrollment.package_npz is None:
                raise ValueError("Enrollment result has no package_npz path.")
            path = Path(enrollment.package_npz)
        else:
            path = Path(enrollment)
        path = self._resolve_user_path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Enrollment package not found: {path}")
        return path

    @staticmethod
    def _read_package_metadata(package_path: Path) -> Dict[str, str]:
        with np.load(package_path, allow_pickle=False) as package:
            return {
                "enrollment_id": SmartAttendanceAI._np_scalar(
                    package["enrollment_id"]
                ),
                "student_id": SmartAttendanceAI._np_scalar(package["student_id"]),
            }


    def _resolve_user_path(self, value: Union[str, Path]) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _assert_no_active_attendance(self, operation: str) -> None:
        active = self.session_manager.active_session_id
        if active is not None:
            raise AIOperationConflictError(
                f"Cannot run {operation} while attendance session {active} is active."
            )

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIFacadeError("SmartAttendanceAI is shut down.")

    @staticmethod
    def _validate_id(value: str, field_name: str) -> str:
        normalized = str(value).strip()
        if not _SAFE_ID.fullmatch(normalized):
            raise ValueError(f"Invalid {field_name}: {normalized!r}")
        return normalized

    @staticmethod
    def _new_enrollment_id(student_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"ENR_{student_id}_{timestamp}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _new_gallery_version_id(student_id: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"GAL_{student_id}_{timestamp}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _np_scalar(value: np.ndarray) -> str:
        array = np.asarray(value)
        if array.size != 1:
            raise AIFacadeError("Expected scalar NPZ metadata.")
        return str(array.reshape(-1)[0])

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()