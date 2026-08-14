"""Data schemas used by the dynamic attendance system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


CameraSource = Union[str, int]


class SessionStatus(str, Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class AttendanceStatus(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"


class RecognitionStatus(str, Enum):
    PENDING = "PENDING"
    MATCH = "MATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"
    CONFIRMED = "CONFIRMED"


@dataclass
class OutputOptions:
    save_csv: bool = True
    save_events: bool = True
    save_summary: bool = True
    save_video: bool = False
    output_directory: Path = Path("outputs")
    output_prefix: Optional[str] = None


@dataclass
class SessionConfig:
    """Dynamic information for one attendance session."""

    session_id: str
    class_id: str
    source: CameraSource
    gallery_path: Path
    roster_student_ids: List[str]
    course_id: Optional[str] = None
    camera_id: Optional[str] = None
    started_at: Optional[datetime] = None
    output: OutputOptions = field(default_factory=OutputOptions)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = self.session_id.strip()
        self.class_id = self.class_id.strip()
        self.gallery_path = Path(self.gallery_path)

        if not self.session_id:
            raise ValueError("session_id cannot be empty.")
        if not self.class_id:
            raise ValueError("class_id cannot be empty.")

        if isinstance(self.source, str):
            self.source = self.source.strip()
            if not self.source:
                raise ValueError("source cannot be empty.")

        if isinstance(self.source, int) and self.source < 0:
            raise ValueError("Camera source index cannot be negative.")

        if not self.roster_student_ids:
            raise ValueError("roster_student_ids cannot be empty.")

        normalized_roster = []
        for student_id in self.roster_student_ids:
            normalized_id = str(student_id).strip()
            if not normalized_id:
                raise ValueError("Roster contains an empty student ID.")
            normalized_roster.append(normalized_id)

        if len(normalized_roster) != len(set(normalized_roster)):
            raise ValueError("roster_student_ids contains duplicates.")

        self.roster_student_ids = normalized_roster

        if self.output.output_prefix is None:
            self.output.output_prefix = self.session_id


@dataclass
class RecognitionEvent:
    event_id: str
    event_type: str
    session_id: str
    student_id: str
    track_id: int
    frame_index: int
    timestamp_seconds: float
    full_name: str = ""
    score: float = 0.0
    margin: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AttendanceRecord:
    session_id: str
    student_id: str
    status: AttendanceStatus
    full_name: str = ""
    first_seen_seconds: Optional[float] = None
    confirmed_at_seconds: Optional[float] = None
    last_detected_seconds: Optional[float] = None
    last_recognized_seconds: Optional[float] = None
    best_score: Optional[float] = None
    best_margin: Optional[float] = None
    observation_count: int = 0
    track_ids: Set[int] = field(default_factory=set)


@dataclass
class TrackResult:
    track_id: int
    bbox: List[float]
    detection_score: float
    hits: int
    last_seen_frame: int
    status: RecognitionStatus
    student_id: str = ""
    full_name: str = ""
    score: float = 0.0
    margin: float = 0.0
    observations: int = 0


@dataclass
class FrameMetrics:
    detection_ms: float = 0.0
    alignment_ms: float = 0.0
    recognition_ms: float = 0.0
    matching_ms: float = 0.0


@dataclass
class FrameResult:
    session_id: str
    frame_index: int
    timestamp_seconds: float
    detection_performed: bool = False
    detected_faces: int = 0
    active_tracks: int = 0
    recognized_faces: int = 0
    newly_confirmed_student_ids: List[str] = field(default_factory=list)
    tracks: List[TrackResult] = field(default_factory=list)
    events: List[RecognitionEvent] = field(default_factory=list)
    metrics: FrameMetrics = field(default_factory=FrameMetrics)


@dataclass
class ProcessingMetrics:
    frames_processed: int = 0
    detection_calls: int = 0
    recognition_calls: int = 0
    recognition_faces: int = 0
    tracks_created: int = 0
    source_fps: float = 0.0
    processing_seconds: float = 0.0
    processing_fps: float = 0.0
    average_detection_ms: float = 0.0
    average_recognition_ms: float = 0.0


@dataclass
class SessionResult:
    session_id: str
    class_id: str
    status: SessionStatus
    attendance: List[AttendanceRecord] = field(default_factory=list)
    events: List[RecognitionEvent] = field(default_factory=list)
    metrics: ProcessingMetrics = field(default_factory=ProcessingMetrics)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def present_student_ids(self) -> List[str]:
        return sorted(
            record.student_id
            for record in self.attendance
            if record.status == AttendanceStatus.PRESENT
        )

    @property
    def absent_student_ids(self) -> List[str]:
        return sorted(
            record.student_id
            for record in self.attendance
            if record.status == AttendanceStatus.ABSENT
        )