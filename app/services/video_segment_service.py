from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class VideoSegmentInfo:
    source_path: Path
    output_path: Path

    start_seconds: int
    duration_seconds: int
    end_seconds: int

    source_duration_seconds: float
    fps: float

    start_frame: int
    end_frame: int
    frames_written: int


def get_video_duration_seconds(
        video_path: Path,
) -> float:
    video_path = Path(video_path).resolve()

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open video: {video_path}"
            )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        frame_count = float(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if fps <= 0:
            raise RuntimeError(
                "Video FPS is invalid."
            )

        if frame_count <= 0:
            raise RuntimeError(
                "Video frame count is invalid."
            )

        return frame_count / fps

    finally:
        capture.release()


def extract_video_segment(
        source_path: Path,
        output_path: Path,
        start_seconds: int,
        duration_seconds: int,
) -> VideoSegmentInfo:
    source_path = Path(
        source_path
    ).resolve()

    output_path = Path(
        output_path
    ).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Source video does not exist: {source_path}"
        )

    if start_seconds < 0:
        raise ValueError(
            "start_seconds cannot be negative."
        )

    if duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be greater than zero."
        )

    end_seconds = (
        start_seconds
        + duration_seconds
    )

    capture = cv2.VideoCapture(
        str(source_path)
    )

    writer = None

    try:
        if not capture.isOpened():
            raise RuntimeError(
                f"Unable to open source video: {source_path}"
            )

        fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        if fps <= 0:
            raise RuntimeError(
                "Source video FPS is invalid."
            )

        if total_frames <= 0:
            raise RuntimeError(
                "Source video has no readable frames."
            )

        if width <= 0 or height <= 0:
            raise RuntimeError(
                "Source video dimensions are invalid."
            )

        source_duration_seconds = (
            total_frames / fps
        )

        if end_seconds > source_duration_seconds:
            raise ValueError(
                "Requested scan segment exceeds "
                "the available video duration. "
                f"requested_end={end_seconds:.2f}s, "
                f"video_duration={source_duration_seconds:.2f}s"
            )

        start_frame = int(
            round(
                start_seconds * fps
            )
        )

        end_frame = int(
            round(
                end_seconds * fps
            )
        )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            start_frame,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fourcc = cv2.VideoWriter_fourcc(
            *"mp4v"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (
                width,
                height,
            ),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Unable to create segment video: {output_path}"
            )

        frames_written = 0

        current_frame = start_frame

        while current_frame < end_frame:
            ok, frame = capture.read()

            if not ok:
                break

            writer.write(
                frame
            )

            frames_written += 1
            current_frame += 1

        if frames_written == 0:
            raise RuntimeError(
                "No frames were written to the scan segment."
            )

        return VideoSegmentInfo(
            source_path=source_path,
            output_path=output_path,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            end_seconds=end_seconds,
            source_duration_seconds=source_duration_seconds,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            frames_written=frames_written,
        )

    finally:
        capture.release()

        if writer is not None:
            writer.release()