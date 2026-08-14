import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "students.csv"
IMAGES_DIR = PROJECT_ROOT / "data" / "raw" / "student_images"

REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "reports"
    / "data_validation.json"
)

REQUIRED_COLUMNS = [
    "student_id",
    "full_name",
    "image_filename",
]

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}

STUDENT_ID_PATTERN = re.compile(r"^STU_\d{3,}$")

MIN_RECOMMENDED_IMAGES = 3
MIN_RECOMMENDED_IMAGE_SIDE = 112


def load_image_unicode(image_path):
    """
    Read an image reliably on Windows, including paths containing
    Unicode characters.
    """

    try:
        image_bytes = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        if image_bytes.size == 0:
            return None

        return cv2.imdecode(
            image_bytes,
            cv2.IMREAD_COLOR,
        )

    except Exception:
        return None


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def main():
    errors = []
    warnings = []

    def add_error(message):
        errors.append(message)
        print(f"[ERROR] {message}")

    def add_warning(message):
        warnings.append(message)
        print(f"[WARNING] {message}")

    print("=" * 70)
    print("SMART ATTENDANCE DATA VALIDATION")
    print("=" * 70)

    print(f"CSV file   : {CSV_PATH}")
    print(f"Images dir : {IMAGES_DIR}")

    if not CSV_PATH.exists():
        add_error(f"CSV file does not exist: {CSV_PATH}")

    if not IMAGES_DIR.exists():
        add_error(
            f"Student images directory does not exist: {IMAGES_DIR}"
        )

    if errors:
        write_report(
            status="FAILED",
            errors=errors,
            warnings=warnings,
            csv_rows=0,
            student_counts={},
            valid_images=0,
            unreferenced_images=[],
        )

        raise SystemExit(1)

    try:
        with CSV_PATH.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(csv_file)

            fieldnames = reader.fieldnames or []

            missing_columns = [
                column
                for column in REQUIRED_COLUMNS
                if column not in fieldnames
            ]

            if missing_columns:
                add_error(
                    "CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )

            extra_columns = [
                column
                for column in fieldnames
                if column not in REQUIRED_COLUMNS
            ]

            if extra_columns:
                add_warning(
                    "CSV contains extra columns: "
                    + ", ".join(extra_columns)
                )

            rows = list(reader)

    except UnicodeDecodeError as error:
        add_error(
            "CSV is not valid UTF-8. Save it as CSV UTF-8 in Excel. "
            f"Details: {error}"
        )

        rows = []

    except Exception as error:
        add_error(f"Could not read CSV: {error}")
        rows = []

    if not rows:
        add_error("CSV does not contain any student-image rows.")

    student_names = defaultdict(set)
    student_counts = Counter()

    seen_student_image_pairs = set()
    referenced_image_names = set()

    file_owners = defaultdict(set)
    content_hash_records = defaultdict(list)

    valid_images = 0

    for line_number, row in enumerate(rows, start=2):
        student_id = (row.get("student_id") or "").strip()
        full_name = (row.get("full_name") or "").strip()
        image_filename = (
            row.get("image_filename") or ""
        ).strip()

        row_reference = f"CSV line {line_number}"

        if not student_id:
            add_error(f"{row_reference}: student_id is empty.")

        if not full_name:
            add_error(f"{row_reference}: full_name is empty.")

        if not image_filename:
            add_error(
                f"{row_reference}: image_filename is empty."
            )

        if not student_id or not full_name or not image_filename:
            continue

        if not STUDENT_ID_PATTERN.fullmatch(student_id):
            add_error(
                f"{row_reference}: invalid student_id "
                f"{student_id!r}; expected format STU_001."
            )

        filename_path = Path(image_filename)

        if filename_path.name != image_filename:
            add_error(
                f"{row_reference}: image_filename must contain "
                f"a filename only, not a path: {image_filename!r}"
            )

            continue

        extension = filename_path.suffix.lower()

        if extension not in ALLOWED_EXTENSIONS:
            add_error(
                f"{row_reference}: unsupported image extension "
                f"{extension!r} for {image_filename!r}."
            )

        pair = (
            student_id.casefold(),
            image_filename.casefold(),
        )

        if pair in seen_student_image_pairs:
            add_error(
                f"{row_reference}: duplicate student/image pair: "
                f"{student_id}, {image_filename}"
            )
        else:
            seen_student_image_pairs.add(pair)

        student_names[student_id].add(full_name)
        student_counts[student_id] += 1

        referenced_image_names.add(
            image_filename.casefold()
        )

        file_owners[
            image_filename.casefold()
        ].add(student_id)

        image_path = IMAGES_DIR / image_filename

        if not image_path.exists():
            add_error(
                f"{row_reference}: image file is missing: "
                f"{image_filename}"
            )

            continue

        if not image_path.is_file():
            add_error(
                f"{row_reference}: image path is not a file: "
                f"{image_filename}"
            )

            continue

        try:
            raw_image_bytes = image_path.read_bytes()

        except Exception as error:
            add_error(
                f"{row_reference}: could not read image bytes "
                f"for {image_filename}: {error}"
            )

            continue

        content_hash = sha256_bytes(raw_image_bytes)

        content_hash_records[content_hash].append(
            {
                "student_id": student_id,
                "image_filename": image_filename,
            }
        )

        image = load_image_unicode(image_path)

        if image is None:
            add_error(
                f"{row_reference}: OpenCV could not decode "
                f"{image_filename}; the image may be corrupted."
            )

            continue

        if image.ndim != 3 or image.shape[2] != 3:
            add_error(
                f"{row_reference}: unexpected image shape "
                f"{image.shape} for {image_filename}."
            )

            continue

        height, width = image.shape[:2]

        if height <= 0 or width <= 0:
            add_error(
                f"{row_reference}: invalid dimensions "
                f"{width}x{height} for {image_filename}."
            )

            continue

        if min(width, height) < MIN_RECOMMENDED_IMAGE_SIDE:
            add_warning(
                f"{row_reference}: image {image_filename} is only "
                f"{width}x{height}; higher resolution is recommended."
            )

        valid_images += 1

    for student_id, names in sorted(student_names.items()):
        if len(names) > 1:
            add_error(
                f"Student {student_id} has inconsistent full names: "
                + " | ".join(sorted(names))
            )

    for image_filename, owners in sorted(file_owners.items()):
        if len(owners) > 1:
            add_error(
                f"Image {image_filename!r} is assigned to multiple "
                f"students: {', '.join(sorted(owners))}"
            )

    for content_hash, records in content_hash_records.items():
        if len(records) <= 1:
            continue

        owners = {
            record["student_id"]
            for record in records
        }

        filenames = [
            record["image_filename"]
            for record in records
        ]

        if len(owners) > 1:
            add_error(
                "Identical image content is assigned to different "
                f"students: {', '.join(filenames)}; "
                f"SHA256={content_hash}"
            )
        else:
            add_warning(
                "Duplicate image content found for the same student: "
                f"{', '.join(filenames)}"
            )

    for student_id, count in sorted(student_counts.items()):
        if count < MIN_RECOMMENDED_IMAGES:
            add_warning(
                f"Student {student_id} has only {count} image(s); "
                f"at least {MIN_RECOMMENDED_IMAGES} are recommended."
            )

    disk_images = {
        path.name.casefold(): path.name
        for path in IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in ALLOWED_EXTENSIONS
    }

    unreferenced_images = sorted(
        original_name
        for normalized_name, original_name in disk_images.items()
        if normalized_name not in referenced_image_names
    )

    for image_filename in unreferenced_images:
        add_warning(
            f"Image exists on disk but is not referenced by CSV: "
            f"{image_filename}"
        )

    if errors:
        status = "FAILED"
    elif warnings:
        status = "PASSED_WITH_WARNINGS"
    else:
        status = "PASSED"

    write_report(
        status=status,
        errors=errors,
        warnings=warnings,
        csv_rows=len(rows),
        student_counts=dict(sorted(student_counts.items())),
        valid_images=valid_images,
        unreferenced_images=unreferenced_images,
    )

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print(f"Status          : {status}")
    print(f"CSV rows        : {len(rows)}")
    print(f"Students        : {len(student_counts)}")
    print(f"Valid images    : {valid_images}")
    print(f"Errors          : {len(errors)}")
    print(f"Warnings        : {len(warnings)}")
    print(f"Report          : {REPORT_PATH}")

    print("\nImages per student:")

    for student_id, count in sorted(student_counts.items()):
        print(f"  {student_id}: {count}")

    if errors:
        raise SystemExit(1)


def write_report(
    status,
    errors,
    warnings,
    csv_rows,
    student_counts,
    valid_images,
    unreferenced_images,
):
    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "status": status,
        "csv_path": str(CSV_PATH),
        "images_directory": str(IMAGES_DIR),
        "csv_rows": csv_rows,
        "student_count": len(student_counts),
        "student_image_counts": student_counts,
        "valid_images": valid_images,
        "unreferenced_images": unreferenced_images,
        "errors": errors,
        "warnings": warnings,
    }

    REPORT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()