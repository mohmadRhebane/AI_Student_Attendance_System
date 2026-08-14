import json
import math
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


def save_image(image_path: Path, image: np.ndarray):
    image_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if not success:
        raise RuntimeError(
            f"Cannot encode image: {image_path}"
        )

    encoded.tofile(str(image_path))


def percentile(values, percentage):
    if not values:
        return None

    return round(
        float(np.percentile(values, percentage)),
        3,
    )


def summarize(values):
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "maximum": None,
            "mean": None,
        }

    array = np.asarray(values, dtype=np.float32)

    return {
        "count": int(array.size),
        "minimum": round(float(array.min()), 3),
        "p10": percentile(values, 10),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "maximum": round(float(array.max()), 3),
        "mean": round(float(array.mean()), 3),
    }


def draw_detections(
    frame,
    detections,
    frame_index,
    timestamp_seconds,
):
    preview = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.astype(int)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 220, 0),
            2,
        )

        for point in detection.landmarks:
            px, py = point.astype(int)

            cv2.circle(
                preview,
                (px, py),
                2,
                (0, 255, 255),
                -1,
            )

        face_width = int(detection.width)
        face_height = int(detection.height)

        cv2.putText(
            preview,
            (
                f"{detection.score:.2f} "
                f"{face_width}x{face_height}"
            ),
            (x1, max(22, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 42),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        preview,
        (
            f"frame={frame_index} "
            f"time={timestamp_seconds:.1f}s "
            f"faces={len(detections)}"
        ),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return preview


def resize_with_letterbox(
    image,
    output_width,
    output_height,
):
    input_height, input_width = image.shape[:2]

    scale = min(
        output_width / input_width,
        output_height / input_height,
    )

    resized_width = max(
        1,
        int(round(input_width * scale)),
    )

    resized_height = max(
        1,
        int(round(input_height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (output_height, output_width, 3),
        dtype=np.uint8,
    )

    x_offset = (
        output_width - resized_width
    ) // 2

    y_offset = (
        output_height - resized_height
    ) // 2

    canvas[
        y_offset:y_offset + resized_height,
        x_offset:x_offset + resized_width,
    ] = resized

    return canvas


def create_contact_sheet(
    preview_frames,
    output_path,
):
    if not preview_frames:
        return

    columns = 4
    tile_width = 320
    tile_height = 200

    rows = math.ceil(
        len(preview_frames) / columns
    )

    sheet = np.full(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        30,
        dtype=np.uint8,
    )

    for index, preview in enumerate(
        preview_frames
    ):
        row = index // columns
        column = index % columns

        tile = resize_with_letterbox(
            preview,
            tile_width,
            tile_height,
        )

        y1 = row * tile_height
        x1 = column * tile_width

        sheet[
            y1:y1 + tile_height,
            x1:x1 + tile_width,
        ] = tile

    save_image(output_path, sheet)


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    video_config = config["video"]

    detector_path = (
        ROOT
        / config["models"]["detector"]["path"]
    )

    video_path = (
        ROOT
        / video_config["development_path"]
    )

    if not video_path.exists():
        raise FileNotFoundError(
            f"Development video not found: "
            f"{video_path}"
        )

    detector = SCRFDDetector(
        model_path=detector_path,
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

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    frame_width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    frame_height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    if fps <= 0 or total_frames <= 0:
        capture.release()

        raise RuntimeError(
            "Video metadata is invalid."
        )

    duration_seconds = total_frames / fps

    interval_seconds = float(
        video_config["probe_interval_seconds"]
    )

    max_samples = int(
        video_config["probe_max_samples"]
    )

    preview_count = int(
        video_config["probe_preview_count"]
    )

    frame_step = max(
        1,
        int(round(fps * interval_seconds)),
    )

    sample_indices = list(
        range(0, total_frames, frame_step)
    )

    if len(sample_indices) > max_samples:
        sample_indices = np.linspace(
            0,
            total_frames - 1,
            max_samples,
            dtype=np.int64,
        ).tolist()

    sample_indices = sorted(
        set(int(index) for index in sample_indices)
    )

    preview_positions = set()

    if sample_indices:
        preview_positions = set(
            np.linspace(
                0,
                len(sample_indices) - 1,
                min(
                    preview_count,
                    len(sample_indices),
                ),
                dtype=np.int64,
            ).tolist()
        )

    print("=" * 70)
    print("DEVELOPMENT VIDEO PROBE")
    print("=" * 70)
    print(f"Video       : {video_path}")
    print(f"Resolution  : {frame_width}x{frame_height}")
    print(f"FPS         : {fps:.3f}")
    print(f"Frames      : {total_frames}")
    print(f"Duration    : {duration_seconds:.3f} seconds")
    print(f"Samples     : {len(sample_indices)}")
    print(f"Providers   : {detector.providers}")
    print()

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    warmup_success, warmup_frame = capture.read()

    if not warmup_success:
        capture.release()
        raise RuntimeError(
            "Could not read warm-up frame."
        )

    print("Warming up SCRFD...")
    detector.detect(warmup_frame)

    detection_times = []
    face_counts = []
    face_scores = []
    original_face_sizes = []
    detector_input_face_sizes = []
    records = []
    preview_frames = []

    detector_scale = min(
        detector.input_width / frame_width,
        detector.input_height / frame_height,
    )

    print("Probing sampled frames...")

    for sample_position, frame_index in enumerate(
        sample_indices
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        read_success, frame = capture.read()

        if not read_success or frame is None:
            records.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(
                        frame_index / fps,
                        3,
                    ),
                    "read_success": False,
                    "face_count": 0,
                }
            )
            continue

        start_time = time.perf_counter()

        detections = detector.detect(frame)

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        detection_times.append(elapsed_ms)
        face_counts.append(len(detections))

        frame_faces = []

        for detection in detections:
            minimum_side = min(
                detection.width,
                detection.height,
            )

            scaled_minimum_side = (
                minimum_side * detector_scale
            )

            face_scores.append(detection.score)
            original_face_sizes.append(minimum_side)
            detector_input_face_sizes.append(
                scaled_minimum_side
            )

            frame_faces.append(
                {
                    "score": round(
                        detection.score,
                        6,
                    ),
                    "bbox": (
                        detection.bbox.round(2).tolist()
                    ),
                    "landmarks": (
                        detection.landmarks
                        .round(2)
                        .tolist()
                    ),
                    "width": round(
                        detection.width,
                        2,
                    ),
                    "height": round(
                        detection.height,
                        2,
                    ),
                    "minimum_side_original": round(
                        minimum_side,
                        2,
                    ),
                    "minimum_side_detector_input": (
                        round(
                            scaled_minimum_side,
                            2,
                        )
                    ),
                }
            )

        timestamp_seconds = frame_index / fps

        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    timestamp_seconds,
                    3,
                ),
                "read_success": True,
                "detection_time_ms": round(
                    elapsed_ms,
                    3,
                ),
                "face_count": len(detections),
                "faces": frame_faces,
            }
        )

        if sample_position in preview_positions:
            preview_frames.append(
                draw_detections(
                    frame=frame,
                    detections=detections,
                    frame_index=frame_index,
                    timestamp_seconds=(
                        timestamp_seconds
                    ),
                )
            )

        print(
            f"[{sample_position + 1:03d}/"
            f"{len(sample_indices):03d}] "
            f"frame={frame_index} | "
            f"time={timestamp_seconds:.1f}s | "
            f"faces={len(detections)} | "
            f"{elapsed_ms:.2f} ms"
        )

    capture.release()

    faces_under_16 = sum(
        size < 16
        for size in detector_input_face_sizes
    )

    faces_under_24 = sum(
        size < 24
        for size in detector_input_face_sizes
    )

    faces_under_32 = sum(
        size < 32
        for size in detector_input_face_sizes
    )

    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    contact_sheet_path = (
        reports_dir
        / "development_video_probe.jpg"
    )

    create_contact_sheet(
        preview_frames=preview_frames,
        output_path=contact_sheet_path,
    )

    report = {
        "video": {
            "path": str(
                video_path.relative_to(ROOT)
            ),
            "width": frame_width,
            "height": frame_height,
            "fps": round(fps, 6),
            "total_frames": total_frames,
            "duration_seconds": round(
                duration_seconds,
                3,
            ),
        },
        "detector": {
            "input_width": detector.input_width,
            "input_height": detector.input_height,
            "score_threshold": (
                detector.score_threshold
            ),
            "nms_threshold": (
                detector.nms_threshold
            ),
            "providers": detector.providers,
            "frame_to_detector_scale": round(
                detector_scale,
                6,
            ),
        },
        "probe": {
            "sample_interval_seconds": (
                interval_seconds
            ),
            "requested_samples": len(
                sample_indices
            ),
            "successful_samples": len(
                detection_times
            ),
            "frames_with_no_faces": sum(
                count == 0
                for count in face_counts
            ),
            "total_detected_faces": int(
                sum(face_counts)
            ),
        },
        "statistics": {
            "detection_time_ms": summarize(
                detection_times
            ),
            "faces_per_frame": summarize(
                face_counts
            ),
            "detection_scores": summarize(
                face_scores
            ),
            "face_minimum_side_original": summarize(
                original_face_sizes
            ),
            "face_minimum_side_detector_input": (
                summarize(
                    detector_input_face_sizes
                )
            ),
            "small_faces_at_detector_input": {
                "under_16_pixels": faces_under_16,
                "under_24_pixels": faces_under_24,
                "under_32_pixels": faces_under_32,
            },
        },
        "records": records,
        "contact_sheet": str(
            contact_sheet_path.relative_to(ROOT)
        ),
    }

    report_path = (
        reports_dir
        / "development_video_probe.json"
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
    print("VIDEO PROBE SUMMARY")
    print("=" * 70)
    print(f"Successful samples : {len(detection_times)}")
    print(
        f"Frames without faces: "
        f"{sum(count == 0 for count in face_counts)}"
    )
    print(f"Detected faces     : {sum(face_counts)}")
    print(
        f"Detection ms p50/p95: "
        f"{percentile(detection_times, 50)} / "
        f"{percentile(detection_times, 95)}"
    )
    print(
        f"Faces/frame mean/max: "
        f"{np.mean(face_counts) if face_counts else 0:.3f} / "
        f"{max(face_counts) if face_counts else 0}"
    )
    print(
        f"Original face size p10/p50: "
        f"{percentile(original_face_sizes, 10)} / "
        f"{percentile(original_face_sizes, 50)}"
    )
    print(
        f"Detector face size p10/p50: "
        f"{percentile(detector_input_face_sizes, 10)} / "
        f"{percentile(detector_input_face_sizes, 50)}"
    )
    print(f"Faces under 16 px : {faces_under_16}")
    print(f"Faces under 24 px : {faces_under_24}")
    print(f"Faces under 32 px : {faces_under_32}")
    print(f"Contact sheet     : {contact_sheet_path}")
    print(f"JSON report       : {report_path}")


if __name__ == "__main__":
    main()