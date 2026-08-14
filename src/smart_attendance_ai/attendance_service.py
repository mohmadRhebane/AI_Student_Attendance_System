"""Internal orchestration service for attendance sessions."""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Union

import cv2
import numpy as np

from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.attendance_schemas import (
    AttendanceRecord,
    AttendanceStatus,
    FrameResult,
    ProcessingMetrics,
    SessionConfig,
    SessionResult,
    SessionStatus,
)


CaptureSource = Union[str, int]
FrameCallback = Callable[[FrameResult], None]


class AttendanceService:
    """
    Run attendance sessions by calling AttendanceEngine directly.

    This class has no FastAPI dependency. FastAPI will call this service later.
    One service instance owns one long-lived detector/recognizer pair and permits
    one active GPU session at a time.
    """

    def __init__(
        self,
        root: Path,
        ai_config_path: Optional[Path] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.engine = AttendanceEngine(
            root=self.root,
            ai_config_path=ai_config_path,
        )
        self.engine.load_models()

        self._run_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active_session_id: Optional[str] = None
        self._last_result: Optional[SessionResult] = None

    @property
    def active_session_id(self) -> Optional[str]:
        return self._active_session_id

    @property
    def last_result(self) -> Optional[SessionResult]:
        return self._last_result

    @property
    def is_running(self) -> bool:
        return self._active_session_id is not None

    def request_stop(self, session_id: str) -> bool:
        """Request cancellation of the currently running session."""
        if self._active_session_id != session_id:
            return False
        self._stop_event.set()
        return True

    def _resolve_capture_source(
        self,
        source: Union[str, int],
    ) -> CaptureSource:
        if isinstance(source, int):
            return source

        source_text = str(source).strip()
        if not source_text:
            raise ValueError("Session source cannot be empty.")

        if source_text.lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        ):
            return source_text

        source_path = Path(source_text)
        if not source_path.is_absolute():
            source_path = self.root / source_path
        source_path = source_path.resolve()

        if not source_path.exists():
            raise FileNotFoundError(
                f"Session source does not exist: {source_path}"
            )

        return str(source_path)

    @staticmethod
    def _is_live_source(source: CaptureSource) -> bool:
        if isinstance(source, int):
            return True
        return str(source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )

    def _build_attendance_records(
        self,
        session_config: SessionConfig,
    ) -> list[AttendanceRecord]:
        records = []

        for student_id in session_config.roster_student_ids:
            present_record = self.engine.attendance.get(student_id)
            if present_record is not None:
                records.append(present_record)
                continue

            records.append(
                AttendanceRecord(
                    session_id=session_config.session_id,
                    student_id=student_id,
                    full_name=self.engine.student_name_map.get(student_id, ""),
                    status=AttendanceStatus.ABSENT,
                )
            )

        return records

    def run_session(
        self,
        session_config: SessionConfig,
        *,
        max_frames: int = 0,
        max_seconds: float = 0.0,
        on_frame: Optional[FrameCallback] = None,
    ) -> SessionResult:
        """
        Run one session synchronously and return structured Python data.

        A future SessionManager will execute this blocking method in a worker
        thread. FastAPI endpoints must call the manager, not process frames.
        """
        if max_frames < 0:
            raise ValueError("max_frames cannot be negative.")
        if max_seconds < 0:
            raise ValueError("max_seconds cannot be negative.")

        if not self._run_lock.acquire(blocking=False):
            raise RuntimeError(
                "Another attendance session is already using this service."
            )

        capture = None
        captured_source_fps = 0.0
        started_at = datetime.now(timezone.utc)
        processing_started = time.perf_counter()
        final_status = SessionStatus.STARTING
        error_message = None
        result: Optional[SessionResult] = None

        self._active_session_id = session_config.session_id
        self._stop_event.clear()

        try:
            self.engine.start_session(session_config)
            capture_source = self._resolve_capture_source(
                session_config.source
            )
            is_live = self._is_live_source(capture_source)

            capture = cv2.VideoCapture(capture_source)
            if not capture.isOpened():
                raise RuntimeError(
                    f"Cannot open session source: {capture_source}"
                )

            captured_source_fps = float(
                capture.get(cv2.CAP_PROP_FPS)
            )
            timestamp_fps = (
                captured_source_fps
                if captured_source_fps > 0
                else 30.0
            )

            success, first_frame = capture.read()
            if not success:
                raise RuntimeError(
                    "Cannot read the first frame from the session source."
                )

            self.engine.warm_up(first_frame)

            pending_frame: Optional[np.ndarray] = first_frame
            if not is_live:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                pending_frame = None

            frame_index = 0
            final_status = SessionStatus.RUNNING

            while not self._stop_event.is_set():
                if pending_frame is not None:
                    frame = pending_frame
                    pending_frame = None
                    success = True
                else:
                    success, frame = capture.read()

                if not success:
                    break

                if is_live:
                    timestamp_seconds = (
                        time.perf_counter() - processing_started
                    )
                else:
                    timestamp_seconds = frame_index / timestamp_fps

                if max_frames and frame_index >= max_frames:
                    break
                if max_seconds and timestamp_seconds >= max_seconds:
                    break

                frame_result = self.engine.process_frame(
                    frame=frame,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                )

                if on_frame is not None:
                    on_frame(frame_result)

                frame_index += 1

            final_status = (
                SessionStatus.CANCELLED
                if self._stop_event.is_set()
                else SessionStatus.COMPLETED
            )

        except Exception as exc:
            final_status = SessionStatus.FAILED
            error_message = f"{type(exc).__name__}: {exc}"

        finally:
            if capture is not None:
                capture.release()

            processing_seconds = time.perf_counter() - processing_started
            runtime = self.engine.get_runtime_info()
            frames_processed = int(runtime["frames_processed"])
            processing_fps = (
                frames_processed / processing_seconds
                if processing_seconds > 0
                else 0.0
            )

            average_detection_ms = (
                sum(self.engine.detector_times)
                / len(self.engine.detector_times)
                if self.engine.detector_times
                else 0.0
            )
            average_recognition_ms = (
                sum(self.engine.recognizer_times)
                / len(self.engine.recognizer_times)
                if self.engine.recognizer_times
                else 0.0
            )

            attendance = self._build_attendance_records(session_config)
            events = list(self.engine.events)

            result = SessionResult(
                session_id=session_config.session_id,
                class_id=session_config.class_id,
                status=final_status,
                attendance=attendance,
                events=events,
                metrics=ProcessingMetrics(
                    frames_processed=frames_processed,
                    detection_calls=int(runtime["detection_calls"]),
                    recognition_calls=int(runtime["recognition_calls"]),
                    recognition_faces=int(runtime["recognition_faces"]),
                    tracks_created=(
                        int(self.engine.tracker.total_created_tracks)
                        if self.engine.tracker is not None
                        else 0
                    ),
                    source_fps=captured_source_fps,
                    processing_seconds=processing_seconds,
                    processing_fps=processing_fps,
                    average_detection_ms=average_detection_ms,
                    average_recognition_ms=average_recognition_ms,
                ),
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                error_message=error_message,
            )

            self._last_result = result
            self._active_session_id = None
            self._stop_event.clear()
            self.engine.clear_session()
            self._run_lock.release()

        return result