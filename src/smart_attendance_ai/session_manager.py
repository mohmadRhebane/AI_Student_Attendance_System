"""Thread-safe background manager for attendance sessions."""

from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import threading
from typing import Dict, List, Optional, Set

from smart_attendance_ai.attendance_schemas import (
    FrameResult,
    SessionConfig,
    SessionResult,
    SessionStatus,
)
from smart_attendance_ai.attendance_service import AttendanceService


TERMINAL_STATUSES = {
    SessionStatus.COMPLETED,
    SessionStatus.FAILED,
    SessionStatus.CANCELLED,
}


@dataclass(frozen=True)
class SessionSnapshot:
    """Serializable status view suitable for a future API response."""

    session_id: str
    class_id: str
    course_id: Optional[str]
    camera_id: Optional[str]
    status: SessionStatus
    source: object
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    frames_processed: int
    latest_timestamp_seconds: float
    present_student_ids: List[str]
    error_message: Optional[str]
    result_available: bool
    stop_requested: bool

    @property
    def present_count(self) -> int:
        return len(self.present_student_ids)

    def to_dict(self) -> dict:
        """Return JSON-friendly primitives for FastAPI later."""
        return {
            "session_id": self.session_id,
            "class_id": self.class_id,
            "course_id": self.course_id,
            "camera_id": self.camera_id,
            "status": self.status.value,
            "source": self.source,
            "created_at": self.created_at.isoformat(),
            "started_at": (
                self.started_at.isoformat() if self.started_at else None
            ),
            "finished_at": (
                self.finished_at.isoformat() if self.finished_at else None
            ),
            "frames_processed": self.frames_processed,
            "latest_timestamp_seconds": round(
                self.latest_timestamp_seconds, 3
            ),
            "present_count": self.present_count,
            "present_student_ids": list(self.present_student_ids),
            "error_message": self.error_message,
            "result_available": self.result_available,
            "stop_requested": self.stop_requested,
        }


@dataclass
class _ManagedSession:
    config: SessionConfig
    max_frames: int = 0
    max_seconds: float = 0.0
    status: SessionStatus = SessionStatus.CREATED
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    frames_processed: int = 0
    latest_timestamp_seconds: float = 0.0
    present_student_ids: Set[str] = field(default_factory=set)
    error_message: Optional[str] = None
    result: Optional[SessionResult] = None
    future: Optional[Future] = None
    stop_requested: bool = False


class AttendanceSessionManager:
    """
    Start and monitor AttendanceService sessions in one background worker.

    The manager intentionally permits one active GPU session at a time. It has
    no FastAPI dependency; future endpoints will call these methods directly.
    """

    def __init__(
        self,
        root: Path,
        service: Optional[AttendanceService] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.service = service or AttendanceService(root=self.root)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="attendance-worker",
        )
        self._lock = threading.RLock()
        self._sessions: Dict[str, _ManagedSession] = {}
        self._active_session_id: Optional[str] = None
        self._closed = False

    @property
    def active_session_id(self) -> Optional[str]:
        with self._lock:
            return self._active_session_id

    def create_session(
        self,
        config: SessionConfig,
        *,
        max_frames: int = 0,
        max_seconds: float = 0.0,
    ) -> SessionSnapshot:
        """Register a session without starting frame processing."""
        if max_frames < 0:
            raise ValueError("max_frames cannot be negative.")
        if max_seconds < 0:
            raise ValueError("max_seconds cannot be negative.")

        with self._lock:
            self._ensure_open()
            if config.session_id in self._sessions:
                raise ValueError(
                    f"Session ID already exists: {config.session_id}"
                )

            managed = _ManagedSession(
                config=config,
                max_frames=int(max_frames),
                max_seconds=float(max_seconds),
            )
            self._sessions[config.session_id] = managed
            return self._snapshot(managed)

    def start_session(self, session_id: str) -> SessionSnapshot:
        """Submit a created session and return immediately."""
        with self._lock:
            self._ensure_open()
            managed = self._require_session(session_id)

            if managed.status != SessionStatus.CREATED:
                raise RuntimeError(
                    "Only a CREATED session can be started; "
                    f"current status is {managed.status.value}."
                )

            if self._active_session_id is not None:
                raise RuntimeError(
                    "Another attendance session is already active: "
                    f"{self._active_session_id}"
                )

            self._active_session_id = session_id
            managed.status = SessionStatus.STARTING
            managed.started_at = datetime.now(timezone.utc)
            managed.future = self._executor.submit(
                self._run_managed_session,
                session_id,
            )
            return self._snapshot(managed)

    def stop_session(self, session_id: str) -> SessionSnapshot:
        """Request cooperative cancellation and return current state."""
        with self._lock:
            managed = self._require_session(session_id)

            if managed.status in TERMINAL_STATUSES:
                return self._snapshot(managed)

            if managed.status == SessionStatus.CREATED:
                managed.stop_requested = True
                managed.status = SessionStatus.CANCELLED
                managed.finished_at = datetime.now(timezone.utc)
                return self._snapshot(managed)

            managed.stop_requested = True
            managed.status = SessionStatus.STOPPING

        # The service may still be between worker startup and run_session().
        # The worker also checks stop_requested before and during callbacks.
        self.service.request_stop(session_id)
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> SessionSnapshot:
        with self._lock:
            return self._snapshot(self._require_session(session_id))

    def list_sessions(self) -> List[SessionSnapshot]:
        with self._lock:
            return [
                self._snapshot(item)
                for item in sorted(
                    self._sessions.values(),
                    key=lambda value: value.created_at,
                )
            ]

    def get_result(self, session_id: str) -> Optional[SessionResult]:
        with self._lock:
            return self._require_session(session_id).result

    def wait_for_terminal(
        self,
        session_id: str,
        timeout: Optional[float] = None,
    ) -> SessionSnapshot:
        """Testing/CLI helper; FastAPI endpoints should poll get_session()."""
        with self._lock:
            managed = self._require_session(session_id)
            future = managed.future

        if future is not None:
            try:
                future.result(timeout=timeout)
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Session did not finish within {timeout} seconds."
                ) from exc

        return self.get_session(session_id)

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            active_id = self._active_session_id
            self._closed = True

        if active_id is not None:
            self.stop_session(active_id)

        self._executor.shutdown(wait=wait)

    def _run_managed_session(self, session_id: str) -> None:
        with self._lock:
            managed = self._require_session(session_id)
            if managed.stop_requested:
                managed.status = SessionStatus.CANCELLED
                managed.finished_at = datetime.now(timezone.utc)
                self._active_session_id = None
                return

        def on_frame(frame_result: FrameResult) -> None:
            should_stop = False
            with self._lock:
                current = self._require_session(session_id)
                if current.status == SessionStatus.STARTING:
                    current.status = SessionStatus.RUNNING
                current.frames_processed = frame_result.frame_index + 1
                current.latest_timestamp_seconds = (
                    frame_result.timestamp_seconds
                )
                current.present_student_ids.update(
                    frame_result.newly_confirmed_student_ids
                )
                should_stop = current.stop_requested

            if should_stop:
                self.service.request_stop(session_id)

        try:
            result = self.service.run_session(
                managed.config,
                max_frames=managed.max_frames,
                max_seconds=managed.max_seconds,
                on_frame=on_frame,
            )

            with self._lock:
                current = self._require_session(session_id)
                current.result = result
                current.status = result.status
                current.frames_processed = result.metrics.frames_processed
                current.present_student_ids = set(
                    result.present_student_ids
                )
                current.error_message = result.error_message
                current.started_at = result.started_at or current.started_at
                current.finished_at = (
                    result.finished_at or datetime.now(timezone.utc)
                )

        except Exception as exc:
            with self._lock:
                current = self._require_session(session_id)
                current.status = SessionStatus.FAILED
                current.error_message = f"{type(exc).__name__}: {exc}"
                current.finished_at = datetime.now(timezone.utc)

        finally:
            with self._lock:
                if self._active_session_id == session_id:
                    self._active_session_id = None

    def _snapshot(self, managed: _ManagedSession) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=managed.config.session_id,
            class_id=managed.config.class_id,
            course_id=managed.config.course_id,
            camera_id=managed.config.camera_id,
            status=managed.status,
            source=managed.config.source,
            created_at=managed.created_at,
            started_at=managed.started_at,
            finished_at=managed.finished_at,
            frames_processed=managed.frames_processed,
            latest_timestamp_seconds=managed.latest_timestamp_seconds,
            present_student_ids=sorted(managed.present_student_ids),
            error_message=managed.error_message,
            result_available=managed.result is not None,
            stop_requested=managed.stop_requested,
        )

    def _require_session(self, session_id: str) -> _ManagedSession:
        normalized_id = str(session_id).strip()
        if normalized_id not in self._sessions:
            raise KeyError(f"Unknown session ID: {normalized_id}")
        return self._sessions[normalized_id]

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("AttendanceSessionManager is shut down.")