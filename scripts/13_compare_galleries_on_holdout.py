import json
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


def save_image(path, image):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 93],
    )

    if not success:
        raise RuntimeError(f"Cannot encode {path}")

    encoded.tofile(str(path))


def draw_result(frame, bbox, result):
    colors = {
        "MATCH": (0, 210, 0),
        "AMBIGUOUS": (0, 200, 255),
        "UNKNOWN": (0, 0, 230),
    }

    color = colors[result.status]
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

    if result.status == "MATCH":
        label = (
            f"{result.student_id} "
            f"{result.score:.2f} "
            f"m={result.margin:.2f}"
        )

    elif result.status == "AMBIGUOUS":
        second_id = (
            result.candidates[1].student_id
            if len(result.candidates) > 1
            else "?"
        )

        label = (
            f"AMB {result.student_id}/"
            f"{second_id}"
        )

    else:
        label = f"UNKNOWN {result.score:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def result_to_dict(result):
    return {
        "status": result.status,
        "student_id": result.student_id,
        "full_name": result.full_name,
        "score": round(result.score, 6),
        "second_score": round(
            result.second_score,
            6,
        ),
        "margin": round(result.margin, 6),
        "candidates": [
            {
                "student_id": candidate.student_id,
                "score": round(
                    candidate.score,
                    6,
                ),
            }
            for candidate in result.candidates
        ],
    }


def summarize(values):
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
        }

    array = np.asarray(
        values,
        dtype=np.float32,
    )

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
    }


def attendance_summary(
    observations,
    expected_present,
    expected_absent,
    minimum_observations,
):
    confirmed = sorted(
        student_id
        for student_id, count
        in observations.items()
        if count >= minimum_observations
    )

    missed_present = sorted(
        set(expected_present) - set(confirmed)
    )

    false_present = sorted(
        set(expected_absent) & set(confirmed)
    )

    return {
        "minimum_observations": (
            minimum_observations
        ),
        "confirmed": confirmed,
        "missed_present": missed_present,
        "false_present": false_present,
        "observations": dict(
            observations.most_common()
        ),
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
    quality_config = config["video_recognition"]
    validation_config = config["validation"]

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
        normalize=True,
    )

    original_matcher = FaceGalleryMatcher(
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

    calibrated_matcher = FaceGalleryMatcher(
        gallery_path=(
            ROOT
            / config["data"][
                "calibrated_gallery_npz"
            ]
        ),
        top_k=matcher_config["top_k"],
        min_score=matcher_config["min_score"],
        min_margin=matcher_config["min_margin"],
        max_candidates=matcher_config[
            "max_candidates"
        ],
    )

    video_path = (
        ROOT
        / config["video"]["development_path"]
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

    start_seconds = float(
        validation_config["start_seconds"]
    )

    end_seconds = float(
        validation_config["end_seconds"]
    )

    interval_seconds = float(
        validation_config[
            "sample_interval_seconds"
        ]
    )

    preview_interval = float(
        validation_config[
            "preview_interval_seconds"
        ]
    )

    start_frame = max(
        0,
        int(round(start_seconds * fps)),
    )

    end_frame = min(
        total_frames - 1,
        int(round(end_seconds * fps)),
    )

    frame_step = max(
        1,
        int(round(interval_seconds * fps)),
    )

    sample_indices = list(
        range(
            start_frame,
            end_frame + 1,
            frame_step,
        )
    )

    preview_step = max(
        1,
        int(round(
            preview_interval / interval_seconds
        )),
    )

    expected_present = [
        str(value)
        for value
        in validation_config[
            "expected_present"
        ]
    ]

    expected_absent = [
        str(value)
        for value
        in validation_config[
            "expected_absent"
        ]
    ]

    minimum_observations = int(
        validation_config[
            "minimum_attendance_observations"
        ]
    )

    results = {
        "original": {
            "status": Counter(),
            "observations": Counter(),
            "duplicate_frames": 0,
        },
        "calibrated": {
            "status": Counter(),
            "observations": Counter(),
            "duplicate_frames": 0,
        },
    }

    detection_times = []
    recognition_times = []
    skipped = Counter()
    frame_records = []

    preview_dir = (
        ROOT
        / "outputs"
        / "reports"
        / "calibrated_validation_frames"
    )

    print("=" * 70)
    print("ORIGINAL VS CALIBRATED GALLERY")
    print("=" * 70)
    print(
        f"Validation range : "
        f"{start_seconds}s - {end_seconds}s"
    )
    print(f"Sample frames    : {len(sample_indices)}")
    print(f"Absent student   : {expected_absent}")
    print(f"Detector         : {detector.providers}")
    print(f"Recognizer       : {recognizer.providers}")
    print()

    capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        start_frame,
    )

    success, warmup_frame = capture.read()

    if not success:
        capture.release()
        raise RuntimeError(
            "Cannot read validation video."
        )

    print("Warming up GPU models...")
    detector.detect(warmup_frame)
    recognizer.warm_up()

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

        timestamp = frame_index / fps
        annotated = frame.copy()

        start = time.perf_counter()
        detections = detector.detect(frame)
        detection_ms = (
            time.perf_counter() - start
        ) * 1000.0

        detection_times.append(detection_ms)

        aligned_faces = []
        accepted_detections = []
        quality_records = []

        for detection in detections:
            minimum_side = min(
                detection.width,
                detection.height,
            )

            if (
                detection.score
                < quality_config[
                    "min_detection_score"
                ]
            ):
                skipped[
                    "low_detection_score"
                ] += 1
                continue

            if (
                minimum_side
                < quality_config[
                    "min_face_size_original"
                ]
            ):
                skipped["face_too_small"] += 1
                continue

            try:
                aligned, _ = align_face(
                    frame,
                    detection.landmarks,
                    output_size=112,
                )

                quality = calculate_face_quality(
                    aligned
                )

            except Exception:
                skipped[
                    "alignment_failed"
                ] += 1
                continue

            if (
                quality["blur_score"]
                < quality_config[
                    "min_blur_score"
                ]
            ):
                skipped["face_blurry"] += 1
                continue

            if (
                quality["brightness"]
                < quality_config[
                    "min_brightness"
                ]
            ):
                skipped["face_too_dark"] += 1
                continue

            if (
                quality["brightness"]
                > quality_config[
                    "max_brightness"
                ]
            ):
                skipped["face_too_bright"] += 1
                continue

            aligned_faces.append(aligned)
            accepted_detections.append(detection)
            quality_records.append(quality)

        original_frame_ids = []
        calibrated_frame_ids = []
        faces_json = []

        if aligned_faces:
            start = time.perf_counter()

            embeddings = recognizer.embed_faces(
                aligned_faces
            )

            recognition_ms = (
                time.perf_counter() - start
            ) * 1000.0

            recognition_times.append(
                recognition_ms
            )

            for detection, quality, embedding in zip(
                accepted_detections,
                quality_records,
                embeddings,
            ):
                original_result = (
                    original_matcher.match(
                        embedding
                    )
                )

                calibrated_result = (
                    calibrated_matcher.match(
                        embedding
                    )
                )

                results["original"]["status"][
                    original_result.status
                ] += 1

                results["calibrated"]["status"][
                    calibrated_result.status
                ] += 1

                if original_result.status == "MATCH":
                    results["original"][
                        "observations"
                    ][
                        original_result.student_id
                    ] += 1

                    original_frame_ids.append(
                        original_result.student_id
                    )

                if (
                    calibrated_result.status
                    == "MATCH"
                ):
                    results["calibrated"][
                        "observations"
                    ][
                        calibrated_result.student_id
                    ] += 1

                    calibrated_frame_ids.append(
                        calibrated_result.student_id
                    )

                draw_result(
                    annotated,
                    detection.bbox,
                    calibrated_result,
                )

                faces_json.append(
                    {
                        "bbox": (
                            detection.bbox
                            .round(2)
                            .tolist()
                        ),
                        "quality": quality,
                        "original": result_to_dict(
                            original_result
                        ),
                        "calibrated": result_to_dict(
                            calibrated_result
                        ),
                    }
                )

        for method, frame_ids in [
            ("original", original_frame_ids),
            ("calibrated", calibrated_frame_ids),
        ]:
            if any(
                count > 1
                for count
                in Counter(frame_ids).values()
            ):
                results[method][
                    "duplicate_frames"
                ] += 1

        cv2.rectangle(
            annotated,
            (0, 0),
            (annotated.shape[1], 44),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            (
                f"CALIBRATED | "
                f"time={timestamp:.1f}s "
                f"det={len(detections)} "
                f"processed={len(aligned_faces)}"
            ),
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if sample_position % preview_step == 0:
            save_image(
                preview_dir
                / f"frame_{frame_index:06d}.jpg",
                annotated,
            )

        frame_records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": round(
                    timestamp,
                    3,
                ),
                "detected": len(detections),
                "processed": len(aligned_faces),
                "faces": faces_json,
            }
        )

        print(
            f"[{sample_position + 1:02d}/"
            f"{len(sample_indices):02d}] "
            f"t={timestamp:.1f}s | "
            f"det={len(detections)} | "
            f"original_match="
            f"{sum(1 for item in faces_json if item['original']['status'] == 'MATCH')} | "
            f"calibrated_match="
            f"{sum(1 for item in faces_json if item['calibrated']['status'] == 'MATCH')}"
        )

    capture.release()

    original_attendance = attendance_summary(
        results["original"]["observations"],
        expected_present,
        expected_absent,
        minimum_observations,
    )

    calibrated_attendance = attendance_summary(
        results["calibrated"]["observations"],
        expected_present,
        expected_absent,
        minimum_observations,
    )

    report = {
        "validation_range_seconds": [
            start_seconds,
            end_seconds,
        ],
        "sample_count": len(frame_records),
        "expected_present": expected_present,
        "expected_absent": expected_absent,
        "skipped": dict(skipped),
        "performance": {
            "detection_ms": summarize(
                detection_times
            ),
            "recognition_batch_ms": summarize(
                recognition_times
            ),
        },
        "original": {
            "status": dict(
                results["original"]["status"]
            ),
            "duplicate_frames": (
                results["original"][
                    "duplicate_frames"
                ]
            ),
            "attendance": original_attendance,
        },
        "calibrated": {
            "status": dict(
                results["calibrated"]["status"]
            ),
            "duplicate_frames": (
                results["calibrated"][
                    "duplicate_frames"
                ]
            ),
            "attendance": (
                calibrated_attendance
            ),
        },
        "frames": frame_records,
        "preview_directory": str(
            preview_dir.relative_to(ROOT)
        ),
    }

    report_path = (
        ROOT
        / "outputs"
        / "reports"
        / "gallery_holdout_comparison.json"
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
    print("HOLDOUT COMPARISON SUMMARY")
    print("=" * 70)

    for method, attendance in [
        ("ORIGINAL", original_attendance),
        ("CALIBRATED", calibrated_attendance),
    ]:
        key = method.lower()

        print()
        print(method)
        print(
            f"  Match       : "
            f"{results[key]['status']['MATCH']}"
        )
        print(
            f"  Ambiguous   : "
            f"{results[key]['status']['AMBIGUOUS']}"
        )
        print(
            f"  Unknown     : "
            f"{results[key]['status']['UNKNOWN']}"
        )
        print(
            f"  Confirmed   : "
            f"{attendance['confirmed']}"
        )
        print(
            f"  Missed      : "
            f"{attendance['missed_present']}"
        )
        print(
            f"  False absent: "
            f"{attendance['false_present']}"
        )
        print(
            f"  Duplicates  : "
            f"{results[key]['duplicate_frames']}"
        )

    print()
    print(f"Preview frames: {preview_dir}")
    print(f"JSON report   : {report_path}")


if __name__ == "__main__":
    main()