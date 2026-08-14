import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.arcface_recognizer import (
    ArcFaceRecognizer,
)
from smart_attendance_ai.face_alignment import (
    align_face,
    calculate_face_quality,
)
from smart_attendance_ai.face_matcher import (
    FaceGalleryMatcher,
)
from smart_attendance_ai.scrfd_detector import (
    SCRFDDetector,
)


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )

    if not success:
        raise RuntimeError(f"Cannot encode {path}")

    encoded.tofile(str(path))


def resize_letterbox(image, width, height):
    source_height, source_width = image.shape[:2]

    scale = min(
        width / source_width,
        height / source_height,
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

    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    x = (width - resized_width) // 2
    y = (height - resized_height) // 2

    canvas[
        y:y + resized_height,
        x:x + resized_width,
    ] = resized

    return canvas


def create_contact_sheet(frames, output_path):
    if not frames:
        return

    columns = 2
    tile_width = 640
    tile_height = 360
    rows = math.ceil(len(frames) / columns)

    sheet = np.zeros(
        (
            rows * tile_height,
            columns * tile_width,
            3,
        ),
        dtype=np.uint8,
    )

    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns

        tile = resize_letterbox(
            frame,
            tile_width,
            tile_height,
        )

        y = row * tile_height
        x = column * tile_width

        sheet[
            y:y + tile_height,
            x:x + tile_width,
        ] = tile

    save_image(output_path, sheet)


def draw_face_result(
    frame,
    bbox,
    status,
    label,
):
    colors = {
        "MATCH": (0, 210, 0),
        "AMBIGUOUS": (0, 200, 255),
        "UNKNOWN": (0, 0, 230),
        "SKIPPED": (150, 150, 150),
    }

    color = colors.get(status, (255, 255, 255))

    x1, y1, x2, y2 = np.asarray(
        bbox,
        dtype=int,
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2,
    )

    text_size, _ = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        2,
    )

    text_width, text_height = text_size

    label_y1 = max(0, y1 - text_height - 10)
    label_y2 = y1

    cv2.rectangle(
        frame,
        (x1, label_y1),
        (min(frame.shape[1] - 1, x1 + text_width + 8), label_y2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 4, max(text_height + 2, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }

    array = np.asarray(values, dtype=np.float32)

    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 3),
        "p50": round(
            float(np.percentile(array, 50)),
            3,
        ),
        "p95": round(
            float(np.percentile(array, 95)),
            3,
        ),
        "maximum": round(float(array.max()), 3),
    }


def main():
    with (ROOT / "configs" / "ai.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    recognizer_config = config["recognizer"]
    matcher_config = config["matcher"]
    video_config = config["video"]
    test_config = config["video_recognition"]

    video_path = (
        ROOT / video_config["development_path"]
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

    recognizer = ArcFaceRecognizer(
        model_path=(
            ROOT
            / config["models"]["recognizer"]["path"]
        ),
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
        normalize=recognizer_config[
            "l2_normalize"
        ],
    )

    matcher = FaceGalleryMatcher(
        gallery_path=(
            ROOT / config["data"]["gallery_npz"]
        ),
        top_k=matcher_config["top_k"],
        min_score=matcher_config["min_score"],
        min_margin=matcher_config["min_margin"],
        max_candidates=matcher_config[
            "max_candidates"
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

    interval = float(
        test_config["sample_interval_seconds"]
    )

    frame_step = max(
        1,
        int(round(fps * interval)),
    )

    sample_indices = list(
        range(0, total_frames, frame_step)
    )

    max_samples = int(test_config["max_samples"])

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

    preview_count = min(
        int(test_config["preview_count"]),
        len(sample_indices),
    )

    preview_positions = set(
        np.linspace(
            0,
            len(sample_indices) - 1,
            preview_count,
            dtype=np.int64,
        ).tolist()
    )

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, warmup_frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError("Cannot read video.")

    print("=" * 70)
    print("SAMPLED VIDEO FACE RECOGNITION")
    print("=" * 70)
    print(f"Video               : {video_path}")
    print(f"Samples             : {len(sample_indices)}")
    print(f"Detector providers  : {detector.providers}")
    print(f"Recognizer providers: {recognizer.providers}")
    print(f"Matcher             : Top-{matcher.top_k} mean")
    print(
        f"Policy              : score>="
        f"{matcher.min_score}, margin>="
        f"{matcher.min_margin}"
    )
    print()

    print("Warming up GPU models...")
    detector.detect(warmup_frame)
    recognizer.warm_up()

    status_counts = Counter()
    quality_reasons = Counter()
    matched_students = Counter()
    ambiguous_pairs = Counter()

    detection_times = []
    recognition_times = []
    processed_faces = 0
    duplicate_match_frames = 0

    frame_records = []
    preview_frames = []

    for sample_position, frame_index in enumerate(
        sample_indices
    ):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index,
        )

        success, frame = capture.read()

        if not success or frame is None:
            continue

        annotated = frame.copy()

        detection_start = time.perf_counter()
        detections = detector.detect(frame)
        detection_ms = (
            time.perf_counter() - detection_start
        ) * 1000.0

        detection_times.append(detection_ms)

        pending_faces = []
        pending_records = []
        face_records = []

        for detection in detections:
            minimum_side = min(
                detection.width,
                detection.height,
            )

            base_record = {
                "bbox": (
                    detection.bbox.round(2).tolist()
                ),
                "detection_score": round(
                    detection.score,
                    6,
                ),
                "minimum_face_side": round(
                    minimum_side,
                    2,
                ),
            }

            if (
                detection.score
                < test_config[
                    "min_detection_score"
                ]
            ):
                base_record.update(
                    {
                        "status": "SKIPPED",
                        "reason": (
                            "low_detection_score"
                        ),
                    }
                )

                face_records.append(base_record)
                status_counts["SKIPPED"] += 1
                quality_reasons[
                    "low_detection_score"
                ] += 1

                draw_face_result(
                    annotated,
                    detection.bbox,
                    "SKIPPED",
                    "SKIP low-det",
                )
                continue

            if (
                minimum_side
                < test_config[
                    "min_face_size_original"
                ]
            ):
                base_record.update(
                    {
                        "status": "SKIPPED",
                        "reason": "face_too_small",
                    }
                )

                face_records.append(base_record)
                status_counts["SKIPPED"] += 1
                quality_reasons[
                    "face_too_small"
                ] += 1

                draw_face_result(
                    annotated,
                    detection.bbox,
                    "SKIPPED",
                    "SKIP small",
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

            except Exception as error:
                base_record.update(
                    {
                        "status": "SKIPPED",
                        "reason": (
                            f"alignment_failed: {error}"
                        ),
                    }
                )

                face_records.append(base_record)
                status_counts["SKIPPED"] += 1
                quality_reasons[
                    "alignment_failed"
                ] += 1

                draw_face_result(
                    annotated,
                    detection.bbox,
                    "SKIPPED",
                    "SKIP align",
                )
                continue

            rejection_reason = None

            if (
                quality["blur_score"]
                < test_config["min_blur_score"]
            ):
                rejection_reason = "face_blurry"

            elif (
                quality["brightness"]
                < test_config["min_brightness"]
            ):
                rejection_reason = "face_too_dark"

            elif (
                quality["brightness"]
                > test_config["max_brightness"]
            ):
                rejection_reason = "face_too_bright"

            if rejection_reason is not None:
                base_record.update(
                    {
                        "status": "SKIPPED",
                        "reason": rejection_reason,
                        "quality": quality,
                    }
                )

                face_records.append(base_record)
                status_counts["SKIPPED"] += 1
                quality_reasons[
                    rejection_reason
                ] += 1

                draw_face_result(
                    annotated,
                    detection.bbox,
                    "SKIPPED",
                    f"SKIP {rejection_reason}",
                )
                continue

            pending_faces.append(aligned_face)

            pending_records.append(
                {
                    "detection": detection,
                    "base_record": base_record,
                    "quality": quality,
                }
            )

        frame_matched_ids = []

        if pending_faces:
            recognition_start = time.perf_counter()

            embeddings = recognizer.embed_faces(
                pending_faces
            )

            recognition_ms = (
                time.perf_counter()
                - recognition_start
            ) * 1000.0

            recognition_times.append(
                recognition_ms
            )

            processed_faces += len(pending_faces)

            for pending, embedding in zip(
                pending_records,
                embeddings,
            ):
                result = matcher.match(embedding)

                status_counts[result.status] += 1

                candidates = [
                    {
                        "student_id": candidate.student_id,
                        "full_name": candidate.full_name,
                        "score": round(
                            candidate.score,
                            6,
                        ),
                        "reference_images": (
                            candidate.top_reference_images
                        ),
                    }
                    for candidate in result.candidates
                ]

                record = pending["base_record"]

                record.update(
                    {
                        "status": result.status,
                        "student_id": (
                            result.student_id
                        ),
                        "full_name": result.full_name,
                        "score": round(
                            result.score,
                            6,
                        ),
                        "second_score": round(
                            result.second_score,
                            6,
                        ),
                        "margin": round(
                            result.margin,
                            6,
                        ),
                        "quality": pending[
                            "quality"
                        ],
                        "candidates": candidates,
                    }
                )

                face_records.append(record)

                if result.status == "MATCH":
                    matched_students[
                        result.student_id
                    ] += 1

                    frame_matched_ids.append(
                        result.student_id
                    )

                    label = (
                        f"{result.student_id} "
                        f"{result.score:.2f} "
                        f"m={result.margin:.2f}"
                    )

                elif result.status == "AMBIGUOUS":
                    second_student_id = (
                        result.candidates[1].student_id
                        if len(result.candidates) > 1
                        else "?"
                    )

                    pair_key = "|".join(
                        sorted(
                            [
                                result.student_id,
                                second_student_id,
                            ]
                        )
                    )

                    ambiguous_pairs[pair_key] += 1

                    label = (
                        f"AMB {result.student_id}/"
                        f"{second_student_id} "
                        f"{result.score:.2f}"
                    )

                else:
                    label = (
                        f"UNKNOWN {result.score:.2f}"
                    )

                draw_face_result(
                    annotated,
                    pending["detection"].bbox,
                    result.status,
                    label,
                )

        duplicate_ids = [
            student_id
            for student_id, count
            in Counter(frame_matched_ids).items()
            if count > 1
        ]

        if duplicate_ids:
            duplicate_match_frames += 1

        timestamp_seconds = frame_index / fps

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 42),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            (
                f"time={timestamp_seconds:.1f}s "
                f"faces={len(detections)} "
                f"match={sum(1 for item in face_records if item['status'] == 'MATCH')} "
                f"amb={sum(1 for item in face_records if item['status'] == 'AMBIGUOUS')}"
            ),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame_records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    timestamp_seconds,
                    3,
                ),
                "detected_faces": len(
                    detections
                ),
                "detection_time_ms": round(
                    detection_ms,
                    3,
                ),
                "duplicate_match_ids": (
                    duplicate_ids
                ),
                "faces": face_records,
            }
        )

        if sample_position in preview_positions:
            preview_frames.append(annotated)

        print(
            f"[{sample_position + 1:03d}/"
            f"{len(sample_indices):03d}] "
            f"time={timestamp_seconds:.1f}s | "
            f"det={len(detections)} | "
            f"match={sum(1 for item in face_records if item['status'] == 'MATCH')} | "
            f"amb={sum(1 for item in face_records if item['status'] == 'AMBIGUOUS')} | "
            f"unknown={sum(1 for item in face_records if item['status'] == 'UNKNOWN')} | "
            f"skip={sum(1 for item in face_records if item['status'] == 'SKIPPED')}"
        )

    capture.release()

    reports_dir = ROOT / "outputs" / "reports"

    contact_sheet_path = (
        reports_dir
        / "development_recognition_samples.jpg"
    )

    create_contact_sheet(
        preview_frames,
        contact_sheet_path,
    )

    report = {
        "video": str(video_path.relative_to(ROOT)),
        "samples": len(frame_records),
        "policy": {
            "matcher_min_score": matcher.min_score,
            "matcher_min_margin": (
                matcher.min_margin
            ),
            "top_k": matcher.top_k,
            "quality": test_config,
        },
        "status_counts": dict(status_counts),
        "quality_rejection_reasons": dict(
            quality_reasons
        ),
        "matched_student_observations": dict(
            matched_students.most_common()
        ),
        "ambiguous_pairs": dict(
            ambiguous_pairs.most_common()
        ),
        "duplicate_match_frames": (
            duplicate_match_frames
        ),
        "processed_faces": processed_faces,
        "performance": {
            "detection_time_ms": summarize(
                detection_times
            ),
            "recognition_batch_time_ms": summarize(
                recognition_times
            ),
            "average_recognition_ms_per_face": (
                round(
                    sum(recognition_times)
                    / processed_faces,
                    3,
                )
                if processed_faces
                else None
            ),
        },
        "frames": frame_records,
        "contact_sheet": str(
            contact_sheet_path.relative_to(ROOT)
        ),
    }

    report_path = (
        reports_dir
        / "development_recognition_samples.json"
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
    print("SAMPLED RECOGNITION SUMMARY")
    print("=" * 70)
    print(f"Samples              : {len(frame_records)}")
    print(f"Processed faces      : {processed_faces}")
    print(f"Matches              : {status_counts['MATCH']}")
    print(f"Ambiguous            : {status_counts['AMBIGUOUS']}")
    print(f"Unknown              : {status_counts['UNKNOWN']}")
    print(f"Skipped              : {status_counts['SKIPPED']}")
    print(
        f"Duplicate match frames: "
        f"{duplicate_match_frames}"
    )
    print(
        f"Detection p50/p95 ms : "
        f"{summarize(detection_times)['p50']} / "
        f"{summarize(detection_times)['p95']}"
    )
    print(
        f"Recognition ms/face  : "
        f"{report['performance']['average_recognition_ms_per_face']}"
    )
    print()
    print("Matched observations per student:")

    for student_id, count in (
        matched_students.most_common()
    ):
        print(f"  {student_id}: {count}")

    print()
    print("Ambiguous pairs:")

    if not ambiguous_pairs:
        print("  None")
    else:
        for pair, count in (
            ambiguous_pairs.most_common()
        ):
            print(f"  {pair}: {count}")

    print()
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"JSON report  : {report_path}")


if __name__ == "__main__":
    main()
