import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.scrfd_detector import (
    SCRFDDetector,
)
from smart_attendance_ai.simple_tracker import (
    SimpleFaceTracker,
)


def main():
    with (ROOT / "configs" / "ai.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    tracking_config = config["tracking"]

    video_path = (
        ROOT
        / config["video"]["development_path"]
    )

    output_path = (
        ROOT
        / "outputs"
        / "videos"
        / "development_tracking.mp4"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = SCRFDDetector(
        model_path=(
            ROOT
            / config["models"]["detector"]["path"]
        ),
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
    )

    tracker = SimpleFaceTracker(
        minimum_iou=tracking_config[
            "minimum_iou"
        ],
        maximum_center_distance=tracking_config[
            "maximum_center_distance"
        ],
        maximum_age_frames=tracking_config[
            "maximum_age_frames"
        ],
        bbox_smoothing=tracking_config[
            "bbox_smoothing"
        ],
    )

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open {video_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Cannot create output video."
        )

    detection_interval = int(
        tracking_config[
            "detection_interval_frames"
        ]
    )

    minimum_hits = int(
        tracking_config["minimum_hits"]
    )

    frame_index = 0
    detection_calls = 0
    detection_times = []

    processing_start = time.perf_counter()

    print("=" * 70)
    print("FACE TRACKING TEST")
    print("=" * 70)
    print(f"Input             : {video_path}")
    print(f"Output            : {output_path}")
    print(f"Frames            : {total_frames}")
    print(f"Detection interval: {detection_interval}")
    print(f"Providers         : {detector.providers}")
    print()

    success, first_frame = capture.read()

    if not success:
        capture.release()
        writer.release()
        raise RuntimeError(
            "Cannot read first frame."
        )

    detector.detect(first_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        success, frame = capture.read()

        if not success:
            break

        if frame_index % detection_interval == 0:
            start = time.perf_counter()
            detections = detector.detect(frame)
            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000.0

            detection_calls += 1
            detection_times.append(
                elapsed_ms
            )

            tracks = tracker.update(
                detections=detections,
                frame_index=frame_index,
            )
        else:
            tracks = tracker.get_active_tracks(
                frame_index
            )

        for track in tracks:
            stale_frames = (
                frame_index
                - track.last_seen_frame
            )

            if stale_frames > (
                detection_interval * 2
            ):
                continue

            x1, y1, x2, y2 = (
                track.bbox.astype(int)
            )

            confirmed = (
                track.hits >= minimum_hits
            )

            color = (
                (0, 220, 0)
                if confirmed
                else (0, 200, 255)
            )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                (
                    f"T{track.track_id} "
                    f"h={track.hits}"
                ),
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 42),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            (
                f"frame={frame_index} "
                f"active_tracks={len(tracks)} "
                f"detection_every={detection_interval}"
            ),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        frame_index += 1

        if (
            frame_index % 300 == 0
            or frame_index == total_frames
        ):
            print(
                f"Processed "
                f"{frame_index}/{total_frames}"
            )

    capture.release()
    writer.release()

    processing_seconds = (
        time.perf_counter()
        - processing_start
    )

    processing_fps = (
        frame_index / processing_seconds
        if processing_seconds > 0
        else 0.0
    )

    report = {
        "input_video": str(
            video_path.relative_to(ROOT)
        ),
        "output_video": str(
            output_path.relative_to(ROOT)
        ),
        "frames_processed": frame_index,
        "video_fps": fps,
        "detection_interval_frames": (
            detection_interval
        ),
        "detection_calls": detection_calls,
        "total_created_tracks": (
            tracker.total_created_tracks
        ),
        "processing_seconds": round(
            processing_seconds,
            3,
        ),
        "processing_fps": round(
            processing_fps,
            3,
        ),
        "detection_ms": {
            "mean": round(
                float(np.mean(detection_times)),
                3,
            ),
            "p50": round(
                float(
                    np.percentile(
                        detection_times,
                        50,
                    )
                ),
                3,
            ),
            "p95": round(
                float(
                    np.percentile(
                        detection_times,
                        95,
                    )
                ),
                3,
            ),
        },
    }

    report_path = (
        ROOT
        / "outputs"
        / "reports"
        / "tracking_report.json"
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("TRACKING SUMMARY")
    print("=" * 70)
    print(f"Frames processed    : {frame_index}")
    print(f"Detection calls     : {detection_calls}")
    print(
        f"Tracks created      : "
        f"{tracker.total_created_tracks}"
    )
    print(
        f"Processing FPS      : "
        f"{processing_fps:.3f}"
    )
    print(f"Output video        : {output_path}")
    print(f"Report              : {report_path}")


if __name__ == "__main__":
    main()