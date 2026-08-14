import csv
import json
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
        image_path = image_path.with_suffix(".jpg")
        extension = ".jpg"

    if extension in {".jpg", ".jpeg"}:
        success, encoded = cv2.imencode(
            extension,
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    else:
        success, encoded = cv2.imencode(extension, image)

    if not success:
        raise RuntimeError(f"Failed to encode image: {image_path}")

    encoded.tofile(str(image_path))


def create_contact_sheet(items, output_path: Path):
    if not items:
        return

    columns = 5
    cell_width = 150
    cell_height = 155
    face_size = 112
    horizontal_offset = (cell_width - face_size) // 2

    rows = math.ceil(len(items) / columns)

    sheet = np.full(
        (rows * cell_height, columns * cell_width, 3),
        245,
        dtype=np.uint8,
    )

    for index, item in enumerate(items):
        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        face = item["aligned_face"]

        sheet[
            y + 5:y + 5 + face_size,
            x + horizontal_offset:x + horizontal_offset + face_size,
        ] = face

        cv2.putText(
            sheet,
            item["student_id"],
            (x + 12, y + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            item["image_number"],
            (x + 90, y + 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )

    save_image(output_path, sheet)


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    data_config = config["data"]

    detector_path = ROOT / config["models"]["detector"]["path"]
    students_csv = ROOT / data_config["students_csv"]
    images_dir = ROOT / data_config["student_images"]
    aligned_dir = ROOT / data_config["aligned_previews"]

    aligned_dir.mkdir(parents=True, exist_ok=True)

    detector = SCRFDDetector(
        model_path=detector_path,
        input_size=(
            detector_config["input_width"],
            detector_config["input_height"],
        ),
        score_threshold=detector_config["score_threshold"],
        nms_threshold=detector_config["nms_threshold"],
    )

    with students_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    records = []
    contact_sheet_items = []
    successful = 0
    failed = 0
    quality_warnings = 0

    print("=" * 70)
    print("ARCFACE FACE ALIGNMENT TEST")
    print("=" * 70)
    print(f"Providers  : {detector.providers}")
    print(f"Input      : {images_dir}")
    print(f"Output     : {aligned_dir}")
    print()

    for position, row in enumerate(rows, start=1):
        student_id = row["student_id"].strip()
        image_filename = row["image_filename"].strip()
        image_path = images_dir / image_filename

        image = load_image(image_path)

        status = "FAILED"
        reason = None
        quality = None
        transform_list = None

        try:
            if image is None:
                raise ValueError("image_read_failed")

            detections = detector.detect(image)

            if len(detections) == 0:
                raise ValueError("no_face")

            if len(detections) > 1:
                raise ValueError("multiple_faces")

            detection = detections[0]

            aligned_face, transform = align_face(
                image=image,
                landmarks=detection.landmarks,
                output_size=112,
            )

            quality = calculate_face_quality(aligned_face)
            transform_list = transform.round(8).tolist()

            warnings = []

            if quality["blur_score"] < 40.0:
                warnings.append("possibly_blurry")

            if quality["brightness"] < 40.0:
                warnings.append("too_dark")

            if quality["brightness"] > 215.0:
                warnings.append("too_bright")

            if quality["contrast"] < 20.0:
                warnings.append("low_contrast")

            output_path = aligned_dir / image_filename
            save_image(output_path, aligned_face)

            image_stem = Path(image_filename).stem
            image_number = image_stem.rsplit("_", 1)[-1]

            contact_sheet_items.append(
                {
                    "student_id": student_id,
                    "image_number": f"img {image_number}",
                    "aligned_face": aligned_face,
                }
            )

            successful += 1
            status = "ALIGNED"

            if warnings:
                quality_warnings += 1
                reason = ",".join(warnings)
                status_text = f"ALIGNED_WITH_WARNING: {reason}"
            else:
                reason = "ok"
                status_text = "ALIGNED"

        except Exception as error:
            failed += 1
            reason = str(error)
            output_path = None
            status_text = f"FAILED: {reason}"

        records.append(
            {
                "student_id": student_id,
                "full_name": row["full_name"].strip(),
                "image_filename": image_filename,
                "status": status,
                "reason": reason,
                "quality": quality,
                "transform": transform_list,
                "aligned_path": (
                    str(output_path.relative_to(ROOT))
                    if output_path is not None
                    else None
                ),
            }
        )

        print(
            f"[{position:02d}/{len(rows):02d}] "
            f"{student_id} | {image_filename} | {status_text}"
        )

    contact_sheet_path = (
        ROOT
        / "outputs"
        / "reports"
        / "alignment_contact_sheet.jpg"
    )

    create_contact_sheet(
        contact_sheet_items,
        contact_sheet_path,
    )

    report = {
        "total_images": len(rows),
        "successful": successful,
        "failed": failed,
        "quality_warnings": quality_warnings,
        "aligned_size": [112, 112],
        "records": records,
    }

    report_path = (
        ROOT
        / "outputs"
        / "reports"
        / "alignment_report.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)
    print(f"Total images       : {len(rows)}")
    print(f"Successfully aligned: {successful}")
    print(f"Failed             : {failed}")
    print(f"Quality warnings   : {quality_warnings}")
    print(f"Aligned directory  : {aligned_dir}")
    print(f"Contact sheet      : {contact_sheet_path}")
    print(f"JSON report        : {report_path}")


if __name__ == "__main__":
    main()