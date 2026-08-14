import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.face_alignment import (
    align_face,
    calculate_face_quality,
)
from smart_attendance_ai.scrfd_detector import (
    SCRFDDetector,
)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError(f"Cannot encode {path}")

    encoded.tofile(str(path))


def resize_letterbox(
    image,
    output_width,
    output_height,
):
    source_height, source_width = image.shape[:2]

    scale = min(
        output_width / source_width,
        output_height / source_height,
    )

    resized_width = max(
        1,
        int(round(source_width * scale)),
    )

    resized_height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (output_height, output_width, 3),
        245,
        dtype=np.uint8,
    )

    x = (output_width - resized_width) // 2
    y = (output_height - resized_height) // 2

    canvas[
        y:y + resized_height,
        x:x + resized_width,
    ] = resized

    return canvas


def extract_context(
    frame,
    bbox,
    scale_factor,
):
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    side = max(width, height) * scale_factor

    crop_x1 = max(
        0,
        int(round(center_x - side / 2.0)),
    )

    crop_y1 = max(
        0,
        int(round(center_y - side / 2.0)),
    )

    crop_x2 = min(
        frame_width,
        int(round(center_x + side / 2.0)),
    )

    crop_y2 = min(
        frame_height,
        int(round(center_y + side / 2.0)),
    )

    return frame[
        crop_y1:crop_y2,
        crop_x1:crop_x2,
    ].copy()


def create_contact_sheet(
    candidates,
    output_path,
):
    if not candidates:
        return

    columns = 4
    cell_width = 310
    cell_height = 180
    rows = math.ceil(len(candidates) / columns)

    sheet = np.full(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for index, candidate in enumerate(candidates):
        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        context = resize_letterbox(
            candidate["context"],
            170,
            130,
        )

        aligned = resize_letterbox(
            candidate["aligned"],
            112,
            112,
        )

        sheet[
            y + 5:y + 135,
            x + 5:x + 175,
        ] = context

        sheet[
            y + 14:y + 126,
            x + 190:x + 302,
        ] = aligned

        cv2.putText(
            sheet,
            candidate["candidate_id"],
            (x + 7, y + 153),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            f"t={candidate['timestamp']:.1f}s",
            (x + 195, y + 153),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    save_image(output_path, sheet)


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    calibration_config = config["calibration"]

    video_path = (
        ROOT
        / config["video"]["development_path"]
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

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    duration_seconds = total_frames / fps

    start_seconds = float(
        calibration_config["start_seconds"]
    )

    end_seconds = min(
        float(calibration_config["end_seconds"]),
        duration_seconds,
    )

    interval_seconds = float(
        calibration_config[
            "sample_interval_seconds"
        ]
    )

    timestamps = np.arange(
        start_seconds,
        end_seconds + 0.001,
        interval_seconds,
    ).tolist()

    aligned_dir = (
        ROOT
        / "data"
        / "previews"
        / "calibration_candidates"
        / "aligned"
    )

    context_dir = (
        ROOT
        / "data"
        / "previews"
        / "calibration_candidates"
        / "context"
    )

    frame_preview_dir = (
        ROOT
        / "outputs"
        / "reports"
        / "calibration_frames"
    )

    aligned_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    context_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_preview_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, warmup_frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "Cannot read video warm-up frame."
        )

    print("=" * 70)
    print("CALIBRATION FACE EXTRACTION")
    print("=" * 70)
    print(f"Video      : {video_path}")
    print(f"Timestamps : {timestamps}")
    print(f"Providers  : {detector.providers}")
    print()
    print("Warming up SCRFD...")

    detector.detect(warmup_frame)

    manifest_rows = []
    contact_candidates = []
    rejected_counts = {}

    for timestamp in timestamps:
        frame_index = int(round(timestamp * fps))

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = capture.read()

        if not success or frame is None:
            print(
                f"[WARNING] Cannot read "
                f"t={timestamp:.1f}s"
            )
            continue

        detections = detector.detect(frame)

        detections = sorted(
            detections,
            key=lambda detection: (
                float(detection.bbox[1]),
                float(detection.bbox[0]),
            ),
        )

        annotated_frame = frame.copy()
        accepted_in_frame = 0

        for detection_index, detection in enumerate(
            detections,
            start=1,
        ):
            minimum_side = min(
                detection.width,
                detection.height,
            )

            if (
                detection.score
                < calibration_config[
                    "min_detection_score"
                ]
            ):
                rejected_counts[
                    "low_detection_score"
                ] = (
                    rejected_counts.get(
                        "low_detection_score",
                        0,
                    )
                    + 1
                )
                continue

            if (
                minimum_side
                < calibration_config[
                    "min_face_size_original"
                ]
            ):
                rejected_counts[
                    "face_too_small"
                ] = (
                    rejected_counts.get(
                        "face_too_small",
                        0,
                    )
                    + 1
                )
                continue

            try:
                aligned_face, _ = align_face(
                    image=frame,
                    landmarks=detection.landmarks,
                    output_size=112,
                )

                quality = calculate_face_quality(
                    aligned_face
                )

            except Exception:
                rejected_counts[
                    "alignment_failed"
                ] = (
                    rejected_counts.get(
                        "alignment_failed",
                        0,
                    )
                    + 1
                )
                continue

            if (
                quality["blur_score"]
                < calibration_config[
                    "min_blur_score"
                ]
            ):
                rejected_counts[
                    "face_blurry"
                ] = (
                    rejected_counts.get(
                        "face_blurry",
                        0,
                    )
                    + 1
                )
                continue

            candidate_id = (
                f"F{frame_index:06d}"
                f"_D{detection_index:02d}"
            )

            aligned_filename = (
                f"{candidate_id}_aligned.jpg"
            )

            context_filename = (
                f"{candidate_id}_context.jpg"
            )

            context = extract_context(
                frame=frame,
                bbox=detection.bbox,
                scale_factor=calibration_config[
                    "context_scale"
                ],
            )

            aligned_path = (
                aligned_dir / aligned_filename
            )

            context_path = (
                context_dir / context_filename
            )

            save_image(
                aligned_path,
                aligned_face,
            )

            save_image(
                context_path,
                context,
            )

            x1, y1, x2, y2 = (
                detection.bbox.astype(int)
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 220, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                candidate_id,
                (x1, max(22, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

            manifest_rows.append(
                {
                    "candidate_id": candidate_id,
                    "frame_index": frame_index,
                    "timestamp_seconds": (
                        f"{timestamp:.3f}"
                    ),
                    "detection_index": (
                        detection_index
                    ),
                    "aligned_filename": (
                        aligned_filename
                    ),
                    "context_filename": (
                        context_filename
                    ),
                    "detection_score": (
                        f"{detection.score:.6f}"
                    ),
                    "minimum_face_side": (
                        f"{minimum_side:.3f}"
                    ),
                    "blur_score": (
                        f"{quality['blur_score']:.3f}"
                    ),
                    "brightness": (
                        f"{quality['brightness']:.3f}"
                    ),
                    "student_id": "",
                    "include": "Y",
                    "notes": "",
                }
            )

            contact_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "timestamp": timestamp,
                    "aligned": aligned_face,
                    "context": context,
                }
            )

            accepted_in_frame += 1

        cv2.rectangle(
            annotated_frame,
            (0, 0),
            (annotated_frame.shape[1], 45),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated_frame,
            (
                f"time={timestamp:.1f}s "
                f"frame={frame_index} "
                f"candidates={accepted_in_frame}"
            ),
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame_preview_path = (
            frame_preview_dir
            / f"frame_{frame_index:06d}.jpg"
        )

        save_image(
            frame_preview_path,
            annotated_frame,
        )

        print(
            f"t={timestamp:5.1f}s | "
            f"frame={frame_index:4d} | "
            f"detected={len(detections):2d} | "
            f"candidates={accepted_in_frame:2d}"
        )

    capture.release()

    manifest_path = (
        ROOT
        / "data"
        / "generated"
        / "calibration_candidates.csv"
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "candidate_id",
        "frame_index",
        "timestamp_seconds",
        "detection_index",
        "aligned_filename",
        "context_filename",
        "detection_score",
        "minimum_face_side",
        "blur_score",
        "brightness",
        "student_id",
        "include",
        "notes",
    ]

    with manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(manifest_rows)

    contact_sheet_path = (
        ROOT
        / "outputs"
        / "reports"
        / "calibration_candidates.jpg"
    )

    create_contact_sheet(
        contact_candidates,
        contact_sheet_path,
    )

    print()
    print("=" * 70)
    print("CALIBRATION EXTRACTION SUMMARY")
    print("=" * 70)
    print(
        f"Accepted candidates: "
        f"{len(manifest_rows)}"
    )
    print(f"Rejected           : {rejected_counts}")
    print(f"Manifest CSV       : {manifest_path}")
    print(f"Contact sheet      : {contact_sheet_path}")
    print(f"Frame previews     : {frame_preview_dir}")
    print()
    print(
        "Fill student_id manually. "
        "Do not assign STU_001."
    )


if __name__ == "__main__":
    main()