import csv
import hashlib
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

from smart_attendance_ai.scrfd_detector import SCRFDDetector


def load_image(image_path: Path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image(image_path: Path, image: np.ndarray):
    image_path.parent.mkdir(parents=True, exist_ok=True)

    extension = image_path.suffix.lower()

    if extension not in {".jpg", ".jpeg", ".png"}:
        extension = ".jpg"
        image_path = image_path.with_suffix(extension)

    success, encoded = cv2.imencode(extension, image)

    if not success:
        raise RuntimeError(f"Could not encode preview: {image_path}")

    encoded.tofile(str(image_path))


def calculate_sha256(file_path: Path):
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def draw_detections(image, detections, accepted):
    preview = image.copy()
    color = (0, 200, 0) if accepted else (0, 0, 255)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.astype(int)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        for point in detection.landmarks:
            px, py = point.astype(int)
            cv2.circle(preview, (px, py), 2, (0, 255, 255), -1)

        label = f"{detection.score:.3f}"

        cv2.putText(
            preview,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return preview


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    detector_model = ROOT / config["models"]["detector"]["path"]
    expected_hash = config["models"]["detector"]["sha256"].upper()
    actual_hash = calculate_sha256(detector_model)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "SCRFD model SHA256 does not match configs/ai.yaml.\n"
            f"Expected: {expected_hash}\n"
            f"Actual  : {actual_hash}"
        )

    detector_config = config["detector"]
    data_config = config["data"]

    detector = SCRFDDetector(
        model_path=detector_model,
        input_size=(
            detector_config["input_width"],
            detector_config["input_height"],
        ),
        score_threshold=detector_config["score_threshold"],
        nms_threshold=detector_config["nms_threshold"],
    )

    students_csv = ROOT / data_config["students_csv"]
    images_dir = ROOT / data_config["student_images"]
    accepted_dir = ROOT / data_config["accepted_previews"]
    rejected_dir = ROOT / data_config["rejected_previews"]

    accepted_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    with students_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    report_records = []
    accepted_count = 0
    rejected_count = 0
    processing_times = []

    print("=" * 70)
    print("SCRFD ENROLLMENT IMAGE TEST")
    print("=" * 70)
    print(f"Providers : {detector.providers}")
    print(f"Images    : {len(rows)}")
    print(f"Threshold : {detector.score_threshold}")
    print()

    for position, row in enumerate(rows, start=1):
        student_id = row["student_id"].strip()
        image_filename = row["image_filename"].strip()
        image_path = images_dir / image_filename

        image = load_image(image_path)

        if image is None:
            reason = "image_read_failed"
            detections = []
            elapsed_ms = 0.0
            accepted = False
        else:
            start_time = time.perf_counter()
            detections = detector.detect(image)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            processing_times.append(elapsed_ms)

            if len(detections) == 0:
                reason = "no_face"
                accepted = False

            elif len(detections) > 1:
                reason = "multiple_faces"
                accepted = False

            else:
                detection = detections[0]
                min_face_side = min(detection.width, detection.height)

                if detection.score < detector_config["enrollment_min_score"]:
                    reason = "low_detection_score"
                    accepted = False

                elif min_face_side < detector_config["min_face_size"]:
                    reason = "face_too_small"
                    accepted = False

                else:
                    reason = "accepted"
                    accepted = True

        if accepted:
            accepted_count += 1
            output_dir = accepted_dir
            status_text = "ACCEPTED"
        else:
            rejected_count += 1
            output_dir = rejected_dir
            status_text = f"REJECTED: {reason}"

        if image is not None:
            preview = draw_detections(image, detections, accepted)
            preview_path = output_dir / image_filename
            save_image(preview_path, preview)
        else:
            preview_path = None

        faces_json = []

        for detection in detections:
            faces_json.append(
                {
                    "score": round(detection.score, 6),
                    "bbox": detection.bbox.round(2).tolist(),
                    "landmarks": detection.landmarks.round(2).tolist(),
                    "width": round(detection.width, 2),
                    "height": round(detection.height, 2),
                }
            )

        report_records.append(
            {
                "student_id": student_id,
                "image_filename": image_filename,
                "accepted": accepted,
                "reason": reason,
                "face_count": len(detections),
                "processing_ms": round(elapsed_ms, 3),
                "faces": faces_json,
                "preview_path": (
                    str(preview_path.relative_to(ROOT))
                    if preview_path is not None
                    else None
                ),
            }
        )

        print(
            f"[{position:02d}/{len(rows):02d}] "
            f"{student_id} | {image_filename} | "
            f"faces={len(detections)} | "
            f"{elapsed_ms:.2f} ms | {status_text}"
        )

    average_ms = (
        float(np.mean(processing_times))
        if processing_times
        else 0.0
    )

    report = {
        "model": str(detector_model.relative_to(ROOT)),
        "model_sha256": actual_hash,
        "providers": detector.providers,
        "total_images": len(rows),
        "accepted": accepted_count,
        "rejected": rejected_count,
        "average_processing_ms": round(average_ms, 3),
        "settings": detector_config,
        "records": report_records,
    }

    report_path = ROOT / "outputs" / "reports" / "detection_test.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 70)
    print("DETECTION TEST SUMMARY")
    print("=" * 70)
    print(f"Total images        : {len(rows)}")
    print(f"Accepted            : {accepted_count}")
    print(f"Rejected            : {rejected_count}")
    print(f"Average total time  : {average_ms:.3f} ms/image")
    print(f"Report              : {report_path}")
    print(f"Accepted previews   : {accepted_dir}")
    print(f"Rejected previews   : {rejected_dir}")

    if rejected_count > 0:
        print()
        print("Review the rejected previews before creating embeddings.")


if __name__ == "__main__":
    main()