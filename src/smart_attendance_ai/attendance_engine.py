"""Core engine for dynamic attendance sessions."""

import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from smart_attendance_ai.arcface_recognizer import ArcFaceRecognizer
from smart_attendance_ai.attendance_schemas import (
    AttendanceRecord,
    AttendanceStatus,
    FrameMetrics,
    FrameResult,
    RecognitionEvent,
    RecognitionStatus,
    SessionConfig,
    TrackResult,
)
from smart_attendance_ai.face_alignment import (
    align_face,
    calculate_face_quality,
)
from smart_attendance_ai.face_matcher import FaceGalleryMatcher, MatchResult
from smart_attendance_ai.scrfd_detector import SCRFDDetector
from smart_attendance_ai.simple_tracker import SimpleFaceTracker
from smart_attendance_ai.temporal_identity import TemporalIdentityResolver


class AttendanceEngine:
    """Process frames while keeping AI models independent of FastAPI and I/O."""

    def __init__(
        self,
        root: Path,
        ai_config_path: Optional[Path] = None,
    ):
        self.root = Path(root).resolve()
        self.ai_config_path = Path(
            ai_config_path
            or self.root / "configs" / "ai.yaml"
        ).resolve()
        self.config = self._load_ai_config()

        self.detector = None
        self.recognizer = None
        self._models_warmed_up = False

        self.matcher = None
        self.tracker = None
        self.resolver = None
        self.session_config = None
        self.student_name_map: Dict[str, str] = {}
        self._roster_ids = set()

        self.last_recognition_frame: Dict[int, int] = {}
        self.attendance: Dict[str, AttendanceRecord] = {}
        self.events = []

        self.frames_processed = 0
        self.detection_calls = 0
        self.recognition_calls = 0
        self.recognition_faces = 0
        self.alignment_failures = 0
        self.detector_times = []
        self.recognizer_times = []

    def _load_ai_config(self) -> dict:
        if not self.ai_config_path.exists():
            raise FileNotFoundError(
                f"AI configuration file does not exist: {self.ai_config_path}"
            )

        with self.ai_config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("ai.yaml must contain a dictionary.")

        required_sections = [
            "models",
            "detector",
            "recognizer",
            "matcher",
            "tracking",
            "video_recognition",
            "attendance",
        ]
        missing_sections = [
            section for section in required_sections if section not in config
        ]
        if missing_sections:
            raise ValueError(
                "Missing ai.yaml sections: " + ", ".join(missing_sections)
            )

        return config

    def _resolve_project_path(self, path_value) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path.resolve()
        return (self.root / path).resolve()

    @property
    def models_loaded(self) -> bool:
        return self.detector is not None and self.recognizer is not None

    @property
    def session_active(self) -> bool:
        return all(
            component is not None
            for component in (
                self.session_config,
                self.matcher,
                self.tracker,
                self.resolver,
            )
        )

    @property
    def ready_for_frames(self) -> bool:
        return self.models_loaded and self.session_active

    def load_models(self) -> None:
        if self.models_loaded:
            return

        detector_config = self.config["detector"]
        recognizer_config = self.config["recognizer"]
        runtime_config = self.config.get(
            "runtime",
            {},
        )

        provider = str(
            runtime_config.get(
                "provider",
                "cuda",
            )
        ).strip().lower()

        allow_cpu_fallback = bool(
            runtime_config.get(
                "allow_cpu_fallback",
                False,
            )
        )

        device_id = int(
            runtime_config.get(
                "device_id",
                0,
            )
        )

        detector_model_path = self._resolve_project_path(
            self.config["models"]["detector"]["path"]
        )
        recognizer_model_path = self._resolve_project_path(
            self.config["models"]["recognizer"]["path"]
        )

        if not detector_model_path.exists():
            raise FileNotFoundError(
                f"Detector model does not exist: {detector_model_path}"
            )
        if not recognizer_model_path.exists():
            raise FileNotFoundError(
                f"Recognizer model does not exist: {recognizer_model_path}"
            )

        self.detector = SCRFDDetector(
            model_path=detector_model_path,
            input_size=(
                detector_config["input_width"],
                detector_config["input_height"],
            ),
            score_threshold=detector_config[
                "score_threshold"
            ],
            nms_threshold=detector_config[
                "nms_threshold"
            ],
            provider=provider,
            allow_cpu_fallback=(
                allow_cpu_fallback
            ),
            device_id=device_id,
        )
        
        self.recognizer = ArcFaceRecognizer(
            model_path=recognizer_model_path,
            input_size=(
                recognizer_config["input_width"],
                recognizer_config["input_height"],
            ),
            embedding_size=recognizer_config[
                "embedding_size"
            ],
            batch_size=recognizer_config[
                "batch_size"
            ],
            normalize=True,
            provider=provider,
            allow_cpu_fallback=(
                allow_cpu_fallback
            ),
            device_id=device_id,
        )

    def warm_up(self, frame: np.ndarray) -> None:
        """Warm up long-lived GPU models once using a real frame."""
        self._validate_frame(frame)
        self.load_models()

        if self._models_warmed_up:
            return

        self.detector.detect(frame)
        self.recognizer.warm_up()
        self._models_warmed_up = True

    def start_session(self, session_config: SessionConfig) -> None:
        self.load_models()
        self.clear_session()

        matcher_config = self.config["matcher"]
        tracking_config = self.config["tracking"]
        attendance_config = self.config["attendance"]
        gallery_path = self._resolve_project_path(session_config.gallery_path)

        if not gallery_path.exists():
            raise FileNotFoundError(
                f"Face gallery does not exist: {gallery_path}"
            )

        self.matcher = FaceGalleryMatcher(
            gallery_path=gallery_path,
            top_k=matcher_config["top_k"],
            min_score=matcher_config["min_score"],
            min_margin=matcher_config["min_margin"],
            max_candidates=matcher_config["max_candidates"],
        )
        self.student_name_map = dict(self.matcher.student_names)
        self._roster_ids = set(session_config.roster_student_ids)

        missing_students = sorted(
            self._roster_ids - set(self.student_name_map.keys())
        )
        if missing_students:
            raise ValueError(
                "The following roster students are missing from the face "
                "gallery: " + ", ".join(missing_students)
            )

        self.tracker = SimpleFaceTracker(
            minimum_iou=tracking_config["minimum_iou"],
            maximum_center_distance=tracking_config[
                "maximum_center_distance"
            ],
            maximum_age_frames=tracking_config["maximum_age_frames"],
            bbox_smoothing=tracking_config["bbox_smoothing"],
        )
        self.resolver = TemporalIdentityResolver(
            history_size=attendance_config["identity_history_size"],
            minimum_observations=attendance_config[
                "minimum_identity_observations"
            ],
            minimum_support_ratio=attendance_config[
                "minimum_support_ratio"
            ],
            minimum_average_score=attendance_config[
                "minimum_average_score"
            ],
            minimum_average_margin=attendance_config[
                "minimum_average_margin"
            ],
        )
        self.session_config = session_config

    def _validate_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError("frame must be a NumPy array.")
        if frame.size == 0:
            raise ValueError("frame cannot be empty.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Expected BGR frame shape (H, W, 3), got {frame.shape}."
            )

    def _match_for_roster(self, embedding: np.ndarray) -> MatchResult:
        """Rank only students belonging to the current session roster."""
        candidates = [
            candidate
            for candidate in self.matcher.rank_candidates(embedding)
            if candidate.student_id in self._roster_ids
        ]

        if not candidates:
            return MatchResult(
                status="UNKNOWN",
                student_id="",
                full_name="",
                score=0.0,
                second_score=0.0,
                margin=0.0,
                candidates=[],
            )

        best = candidates[0]
        second_score = candidates[1].score if len(candidates) > 1 else -1.0
        margin = best.score - second_score

        if best.score < self.matcher.min_score:
            status = "UNKNOWN"
        elif margin < self.matcher.min_margin:
            status = "AMBIGUOUS"
        else:
            status = "MATCH"

        return MatchResult(
            status=status,
            student_id=best.student_id,
            full_name=best.full_name,
            score=best.score,
            second_score=second_score,
            margin=margin,
            candidates=candidates[: self.matcher.max_candidates],
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
    ) -> FrameResult:
        """Process one frame and return data without drawing or writing files."""
        if not self.ready_for_frames:
            raise RuntimeError("Start a session before processing frames.")

        self._validate_frame(frame)
        frame_index = int(frame_index)
        timestamp_seconds = float(timestamp_seconds)
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative.")
        if timestamp_seconds < 0:
            raise ValueError("timestamp_seconds cannot be negative.")

        tracking_config = self.config["tracking"]
        quality_config = self.config["video_recognition"]
        attendance_config = self.config["attendance"]

        detection_interval = int(
            tracking_config["detection_interval_frames"]
        )
        recognition_interval = int(
            attendance_config["recognition_interval_frames"]
        )
        minimum_track_hits = int(attendance_config["minimum_track_hits"])

        detection_performed = frame_index % detection_interval == 0
        detections = []
        recognition_tracks = []
        aligned_faces = []
        frame_events = []
        newly_confirmed = []
        metrics = FrameMetrics()

        if detection_performed:
            started = time.perf_counter()
            detections = self.detector.detect(frame)
            metrics.detection_ms = (time.perf_counter() - started) * 1000.0
            self.detector_times.append(metrics.detection_ms)
            self.detection_calls += 1

            tracks = self.tracker.update(detections, frame_index)

            alignment_started = time.perf_counter()
            for track in tracks:
                if track.last_seen_frame != frame_index:
                    continue
                if track.hits < minimum_track_hits:
                    continue

                last_frame = self.last_recognition_frame.get(
                    track.track_id,
                    -recognition_interval,
                )
                if frame_index - last_frame < recognition_interval:
                    continue

                minimum_side = min(track.width, track.height)
                if (
                    track.detection_score
                    < quality_config["min_detection_score"]
                ):
                    continue
                if minimum_side < quality_config["min_face_size_original"]:
                    continue

                try:
                    aligned, _ = align_face(
                        frame,
                        track.landmarks,
                        output_size=112,
                    )
                    quality = calculate_face_quality(aligned)
                except Exception:
                    self.alignment_failures += 1
                    continue

                if quality["blur_score"] < quality_config["min_blur_score"]:
                    continue
                if not (
                    quality_config["min_brightness"]
                    <= quality["brightness"]
                    <= quality_config["max_brightness"]
                ):
                    continue

                recognition_tracks.append(track)
                aligned_faces.append(aligned)

            metrics.alignment_ms = (
                time.perf_counter() - alignment_started
            ) * 1000.0

            if aligned_faces:
                started = time.perf_counter()
                embeddings = self.recognizer.embed_faces(aligned_faces)
                metrics.recognition_ms = (
                    time.perf_counter() - started
                ) * 1000.0
                self.recognizer_times.append(metrics.recognition_ms)
                self.recognition_calls += 1
                self.recognition_faces += len(aligned_faces)

                matching_started = time.perf_counter()
                for track, embedding in zip(recognition_tracks, embeddings):
                    self.last_recognition_frame[track.track_id] = frame_index
                    match_result = self._match_for_roster(embedding)
                    # if match_result.student_id in {"5", "10", "15"}:
                        # print(
                        #     f"[MATCH DEBUG] frame={frame_index} "
                        #     f"track={track.track_id} "
                        #     f"student={match_result.student_id} "
                        #     f"status={match_result.status} "
                        #     f"score={match_result.score:.4f} "
                        #     f"second={match_result.second_score:.4f} "
                        #     f"margin={match_result.margin:.4f}"
                        # )
                    state, confirmed_now = self.resolver.add_result(
                        track_id=track.track_id,
                        frame_index=frame_index,
                        timestamp_seconds=timestamp_seconds,
                        match_result=match_result,
                    )

                    if confirmed_now:
                        student_id = state.confirmed_student_id
                        if student_id not in self.attendance:
                            matching_history = [
                                item.timestamp_seconds
                                for item in state.history
                                if item.student_id == student_id
                            ]
                            record = AttendanceRecord(
                                session_id=self.session_config.session_id,
                                student_id=student_id,
                                full_name=self.student_name_map.get(
                                    student_id, ""
                                ),
                                status=AttendanceStatus.PRESENT,
                                first_seen_seconds=min(matching_history),
                                confirmed_at_seconds=timestamp_seconds,
                                last_detected_seconds=timestamp_seconds,
                                last_recognized_seconds=timestamp_seconds,
                                best_score=state.average_score,
                                best_margin=state.average_margin,
                                observation_count=state.winner_observations,
                                track_ids={track.track_id},
                            )
                            self.attendance[student_id] = record

                            event = RecognitionEvent(
                                event_id=str(uuid.uuid4()),
                                event_type="attendance_confirmed",
                                session_id=self.session_config.session_id,
                                student_id=student_id,
                                full_name=record.full_name,
                                track_id=track.track_id,
                                frame_index=frame_index,
                                timestamp_seconds=round(
                                    timestamp_seconds, 3
                                ),
                                score=round(state.average_score, 6),
                                margin=round(state.average_margin, 6),
                            )
                            self.events.append(event)
                            frame_events.append(event)
                            newly_confirmed.append(student_id)

                    if state.confirmed_student_id:
                        student_id = state.confirmed_student_id
                        record = self.attendance.get(student_id)
                        if record is not None:
                            record.last_recognized_seconds = timestamp_seconds
                            record.best_score = max(
                                record.best_score or 0.0,
                                state.average_score,
                            )
                            record.best_margin = max(
                                record.best_margin or 0.0,
                                state.average_margin,
                            )
                            record.observation_count = max(
                                record.observation_count,
                                state.winner_observations,
                            )
                            record.track_ids.add(track.track_id)

                metrics.matching_ms = (
                    time.perf_counter() - matching_started
                ) * 1000.0
        else:
            tracks = self.tracker.get_active_tracks(frame_index)

        visible_tracks = []
        for track in tracks:
            stale_frames = frame_index - track.last_seen_frame
            if stale_frames > detection_interval * 2:
                continue

            state = self.resolver.get_state(track.track_id)
            if state.confirmed_student_id:
                status = RecognitionStatus.CONFIRMED
                student_id = state.confirmed_student_id
            elif state.latest_status == "AMBIGUOUS":
                status = RecognitionStatus.AMBIGUOUS
                student_id = state.latest_student_id
            elif state.latest_status == "UNKNOWN":
                status = RecognitionStatus.UNKNOWN
                student_id = ""
            elif state.latest_status == "MATCH":
                status = RecognitionStatus.MATCH
                student_id = state.latest_student_id
            else:
                status = RecognitionStatus.PENDING
                student_id = ""

            if (
                state.confirmed_student_id
                and track.last_seen_frame == frame_index
            ):
                record = self.attendance.get(state.confirmed_student_id)
                if record is not None:
                    record.last_detected_seconds = timestamp_seconds
                    record.track_ids.add(track.track_id)

            visible_tracks.append(
                TrackResult(
                    track_id=track.track_id,
                    bbox=[float(value) for value in track.bbox],
                    detection_score=float(track.detection_score),
                    hits=int(track.hits),
                    last_seen_frame=int(track.last_seen_frame),
                    status=status,
                    student_id=student_id,
                    full_name=self.student_name_map.get(student_id, ""),
                    score=float(state.average_score),
                    margin=float(state.average_margin),
                    observations=int(state.winner_observations),
                )
            )

        self.frames_processed += 1

        return FrameResult(
            session_id=self.session_config.session_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            detection_performed=detection_performed,
            detected_faces=len(detections),
            active_tracks=len(visible_tracks),
            recognized_faces=len(aligned_faces),
            newly_confirmed_student_ids=newly_confirmed,
            tracks=visible_tracks,
            events=frame_events,
            metrics=metrics,
        )

    def clear_session(self) -> None:
        self.session_config = None
        self.matcher = None
        self.tracker = None
        self.resolver = None
        self.student_name_map = {}
        self._roster_ids = set()
        self.last_recognition_frame = {}
        self.attendance = {}
        self.events = []
        self.frames_processed = 0
        self.detection_calls = 0
        self.recognition_calls = 0
        self.recognition_faces = 0
        self.alignment_failures = 0
        self.detector_times = []
        self.recognizer_times = []

    def get_runtime_info(self) -> dict:
        runtime_config = self.config.get("runtime", {})

        detector_providers = (
            list(self.detector.providers) if self.detector else []
        )

        recognizer_providers = (
            list(self.recognizer.providers) if self.recognizer else []
        )

        return {
            "models_loaded": self.models_loaded,
            "models_warmed_up": self._models_warmed_up,
            "session_active": self.session_active,
            "ready_for_frames": self.ready_for_frames,

            "requested_provider": str(
                runtime_config.get("provider", "cuda")
            ).lower(),

            "allow_cpu_fallback": bool(
                runtime_config.get("allow_cpu_fallback", False)
            ),

            "device_id": int(
                runtime_config.get("device_id", 0)
            ),

            "effective_provider": (
                detector_providers[0]
                if detector_providers
                else None
            ),

            "session_id": (
                self.session_config.session_id
                if self.session_config
                else None
            ),

            "class_id": (
                self.session_config.class_id
                if self.session_config
                else None
            ),

            "roster_count": (
                len(self.session_config.roster_student_ids)
                if self.session_config
                else 0
            ),

            "gallery_student_count": len(self.student_name_map),

            "frames_processed": self.frames_processed,
            "detection_calls": self.detection_calls,
            "recognition_calls": self.recognition_calls,
            "recognition_faces": self.recognition_faces,
            "alignment_failures": self.alignment_failures,
            "present_count": len(self.attendance),

            "detector_providers": detector_providers,
            "recognizer_providers": recognizer_providers,
        }