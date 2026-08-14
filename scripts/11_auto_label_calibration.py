import csv
import json
import math
import sys
from collections import defaultdict
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


def load_image(path: Path):
    try:
        encoded = np.fromfile(
            str(path),
            dtype=np.uint8,
        )
        return cv2.imdecode(
            encoded,
            cv2.IMREAD_COLOR,
        )
    except Exception:
        return None


def save_image(path: Path, image: np.ndarray):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError(f"Cannot encode {path}")

    encoded.tofile(str(path))


def normalize_rows(vectors):
    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    return (
        vectors / np.maximum(norms, 1e-12)
    ).astype(np.float32)


def solve_unique_assignment(score_matrix):
    number_of_faces, number_of_students = (
        score_matrix.shape
    )

    if number_of_faces > number_of_students:
        raise ValueError(
            "There are more faces than present students."
        )

    states = {
        0: (0.0, [])
    }

    for face_index in range(number_of_faces):
        new_states = {}

        for mask, (
            previous_score,
            previous_assignment,
        ) in states.items():
            for student_index in range(
                number_of_students
            ):
                student_bit = 1 << student_index

                if mask & student_bit:
                    continue

                new_mask = mask | student_bit

                total_score = (
                    previous_score
                    + float(
                        score_matrix[
                            face_index,
                            student_index,
                        ]
                    )
                )

                current_best = new_states.get(
                    new_mask
                )

                if (
                    current_best is None
                    or total_score
                    > current_best[0]
                ):
                    new_states[new_mask] = (
                        total_score,
                        previous_assignment
                        + [student_index],
                    )

        states = new_states

    _, best_assignment = max(
        states.values(),
        key=lambda item: item[0],
    )

    return best_assignment


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

    width = max(
        1,
        int(round(source_width * scale)),
    )

    height = max(
        1,
        int(round(source_height * scale)),
    )

    resized = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.full(
        (output_height, output_width, 3),
        245,
        dtype=np.uint8,
    )

    x = (output_width - width) // 2
    y = (output_height - height) // 2

    canvas[
        y:y + height,
        x:x + width,
    ] = resized

    return canvas


def create_review_sheet(
    selected_records,
    context_dir,
    aligned_dir,
    output_path,
):
    if not selected_records:
        return

    columns = 4
    cell_width = 320
    cell_height = 195
    rows = math.ceil(
        len(selected_records) / columns
    )

    sheet = np.full(
        (
            rows * cell_height,
            columns * cell_width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    for index, record in enumerate(
        selected_records
    ):
        row = index // columns
        column = index % columns

        x = column * cell_width
        y = row * cell_height

        context = load_image(
            context_dir
            / record["context_filename"]
        )

        aligned = load_image(
            aligned_dir
            / record["aligned_filename"]
        )

        if context is None or aligned is None:
            continue

        context = resize_letterbox(
            context,
            175,
            130,
        )

        aligned = resize_letterbox(
            aligned,
            112,
            112,
        )

        sheet[
            y + 5:y + 135,
            x + 5:x + 180,
        ] = context

        sheet[
            y + 14:y + 126,
            x + 196:x + 308,
        ] = aligned

        border_color = (
            (0, 180, 0)
            if record["assigned_rank"] == 1
            else (0, 170, 255)
        )

        cv2.rectangle(
            sheet,
            (x + 194, y + 12),
            (x + 310, y + 128),
            border_color,
            2,
        )

        cv2.putText(
            sheet,
            record["auto_student_id"],
            (x + 7, y + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (10, 10, 10),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            record["candidate_id"],
            (x + 100, y + 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            (
                f"score={record['assigned_score']:.3f} "
                f"rank={record['assigned_rank']}"
            ),
            (x + 7, y + 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            border_color,
            1,
            cv2.LINE_AA,
        )

    save_image(output_path, sheet)


def main():
    with (ROOT / "configs" / "ai.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    auto_config = config["auto_calibration"]
    recognizer_config = config["recognizer"]

    present_student_ids = [
        str(student_id)
        for student_id
        in auto_config["present_student_ids"]
    ]

    if "STU_001" in present_student_ids:
        raise RuntimeError(
            "STU_001 must not be present in "
            "auto_calibration.present_student_ids."
        )

    select_per_student = int(
        auto_config["select_per_student"]
    )

    minimum_assignment_score = float(
        auto_config[
            "minimum_assignment_score"
        ]
    )

    manifest_path = (
        ROOT
        / "data"
        / "generated"
        / "calibration_candidates.csv"
    )

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

    with manifest_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        manifest_rows = list(
            csv.DictReader(file)
        )

    if not manifest_rows:
        raise RuntimeError(
            "Calibration manifest is empty."
        )

    aligned_faces = []

    for row in manifest_rows:
        image_path = (
            aligned_dir
            / row["aligned_filename"]
        )

        image = load_image(image_path)

        if image is None:
            raise RuntimeError(
                f"Cannot read {image_path}"
            )

        if image.shape != (112, 112, 3):
            raise RuntimeError(
                f"Invalid aligned face shape "
                f"{image.shape}: {image_path}"
            )

        aligned_faces.append(image)

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

    print("=" * 70)
    print("AUTOMATIC CALIBRATION LABELLING")
    print("=" * 70)
    print(f"Candidates       : {len(aligned_faces)}")
    print(f"Present students : {len(present_student_ids)}")
    print(f"Providers        : {recognizer.providers}")
    print()

    print("Warming up ArcFace...")
    recognizer.warm_up()

    print("Generating calibration embeddings...")
    calibration_embeddings = (
        recognizer.embed_faces(aligned_faces)
    )

    gallery_path = (
        ROOT / config["data"]["gallery_npz"]
    )

    gallery = np.load(
        gallery_path,
        allow_pickle=False,
    )

    gallery_embeddings = normalize_rows(
        gallery["image_embeddings"]
    )

    gallery_student_ids = gallery[
        "image_student_ids"
    ].astype(str).tolist()

    gallery_student_names = dict(
        zip(
            gallery["student_ids"]
            .astype(str)
            .tolist(),
            gallery["student_names"]
            .astype(str)
            .tolist(),
        )
    )

    reference_indices = {}

    for student_id in present_student_ids:
        indices = [
            index
            for index, gallery_student_id
            in enumerate(gallery_student_ids)
            if gallery_student_id == student_id
        ]

        if not indices:
            raise RuntimeError(
                f"No original references for "
                f"{student_id}."
            )

        reference_indices[student_id] = indices

    score_matrix = np.empty(
        (
            len(calibration_embeddings),
            len(present_student_ids),
        ),
        dtype=np.float32,
    )

    for face_index, embedding in enumerate(
        calibration_embeddings
    ):
        for student_index, student_id in enumerate(
            present_student_ids
        ):
            references = gallery_embeddings[
                reference_indices[student_id]
            ]

            similarities = references @ embedding

            top_count = min(
                int(config["matcher"]["top_k"]),
                len(similarities),
            )

            top_scores = np.sort(
                similarities
            )[::-1][:top_count]

            score_matrix[
                face_index,
                student_index,
            ] = float(top_scores.mean())

    indices_by_frame = defaultdict(list)

    for index, row in enumerate(manifest_rows):
        indices_by_frame[
            int(row["frame_index"])
        ].append(index)

    assignments = []

    for frame_index in sorted(indices_by_frame):
        candidate_indices = indices_by_frame[
            frame_index
        ]

        frame_scores = score_matrix[
            candidate_indices
        ]

        student_assignment = (
            solve_unique_assignment(frame_scores)
        )

        for local_face_index, student_index in (
            enumerate(student_assignment)
        ):
            candidate_index = candidate_indices[
                local_face_index
            ]

            row_scores = score_matrix[
                candidate_index
            ]

            ranking = np.argsort(
                row_scores
            )[::-1]

            assigned_rank = (
                int(
                    np.where(
                        ranking == student_index
                    )[0][0]
                )
                + 1
            )

            raw_top1_index = int(ranking[0])
            raw_top2_index = int(ranking[1])

            assigned_student_id = (
                present_student_ids[
                    student_index
                ]
            )

            assignment = dict(
                manifest_rows[candidate_index]
            )

            assignment.update(
                {
                    "candidate_index": (
                        candidate_index
                    ),
                    "auto_student_id": (
                        assigned_student_id
                    ),
                    "auto_full_name": (
                        gallery_student_names.get(
                            assigned_student_id,
                            "",
                        )
                    ),
                    "assigned_score": float(
                        row_scores[student_index]
                    ),
                    "assigned_rank": (
                        assigned_rank
                    ),
                    "raw_top1_student_id": (
                        present_student_ids[
                            raw_top1_index
                        ]
                    ),
                    "raw_top1_score": float(
                        row_scores[
                            raw_top1_index
                        ]
                    ),
                    "raw_top2_student_id": (
                        present_student_ids[
                            raw_top2_index
                        ]
                    ),
                    "raw_top2_score": float(
                        row_scores[
                            raw_top2_index
                        ]
                    ),
                    "raw_margin": float(
                        row_scores[
                            raw_top1_index
                        ]
                        - row_scores[
                            raw_top2_index
                        ]
                    ),
                    "auto_include": "N",
                }
            )

            assignments.append(assignment)

    assignments_by_student = defaultdict(list)

    for assignment in assignments:
        assignments_by_student[
            assignment["auto_student_id"]
        ].append(assignment)

    warnings = []

    for student_id in present_student_ids:
        student_assignments = (
            assignments_by_student[student_id]
        )

        eligible = [
            assignment
            for assignment in student_assignments
            if assignment["assigned_score"]
            >= minimum_assignment_score
        ]

        eligible.sort(
            key=lambda assignment: (
                assignment["assigned_rank"] != 1,
                -assignment["assigned_score"],
                -float(
                    assignment[
                        "minimum_face_side"
                    ]
                ),
                -float(
                    assignment["blur_score"]
                ),
            )
        )

        selected = eligible[
            :select_per_student
        ]

        for assignment in selected:
            assignment["auto_include"] = "Y"

        if len(selected) < select_per_student:
            warnings.append(
                f"{student_id} has only "
                f"{len(selected)} selected candidate(s)."
            )

    assignments.sort(
        key=lambda assignment: (
            int(assignment["frame_index"]),
            int(assignment["detection_index"]),
        )
    )

    output_path = (
        ROOT
        / "data"
        / "generated"
        / "auto_calibration_labels.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(manifest_rows[0].keys()) + [
        "auto_student_id",
        "auto_full_name",
        "assigned_score",
        "assigned_rank",
        "raw_top1_student_id",
        "raw_top1_score",
        "raw_top2_student_id",
        "raw_top2_score",
        "raw_margin",
        "auto_include",
    ]

    temporary_path = output_path.with_suffix(
        ".csv.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for assignment in assignments:
            csv_row = dict(assignment)

            for key in [
                "assigned_score",
                "raw_top1_score",
                "raw_top2_score",
                "raw_margin",
            ]:
                csv_row[key] = (
                    f"{float(csv_row[key]):.8f}"
                )

            writer.writerow(csv_row)

    temporary_path.replace(output_path)

    selected_records = [
        assignment
        for assignment in assignments
        if assignment["auto_include"] == "Y"
    ]

    selected_records.sort(
        key=lambda assignment: (
            assignment["auto_student_id"],
            float(
                assignment[
                    "timestamp_seconds"
                ]
            ),
        )
    )

    review_path = (
        ROOT
        / "outputs"
        / "reports"
        / "auto_calibration_review.jpg"
    )

    create_review_sheet(
        selected_records=selected_records,
        context_dir=context_dir,
        aligned_dir=aligned_dir,
        output_path=review_path,
    )

    forced_assignments = [
        assignment
        for assignment in selected_records
        if assignment["assigned_rank"] > 1
    ]

    report = {
        "present_student_ids": (
            present_student_ids
        ),
        "absent_student_ids": ["STU_001"],
        "total_candidates": len(assignments),
        "selected_candidates": len(
            selected_records
        ),
        "forced_selected_assignments": len(
            forced_assignments
        ),
        "warnings": warnings,
        "selected_per_student": {
            student_id: sum(
                assignment["auto_include"] == "Y"
                for assignment
                in assignments_by_student[
                    student_id
                ]
            )
            for student_id in present_student_ids
        },
        "output_csv": str(
            output_path.relative_to(ROOT)
        ),
        "review_image": str(
            review_path.relative_to(ROOT)
        ),
    }

    report_path = (
        ROOT
        / "outputs"
        / "reports"
        / "auto_calibration_report.json"
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
    print("AUTO-LABELLING SUMMARY")
    print("=" * 70)
    print(
        f"Total candidates          : "
        f"{len(assignments)}"
    )
    print(
        f"Automatically selected    : "
        f"{len(selected_records)}"
    )
    print(
        f"Forced selected assignments: "
        f"{len(forced_assignments)}"
    )
    print()
    print("Selected per student:")

    for student_id in present_student_ids:
        count = report[
            "selected_per_student"
        ][student_id]

        scores = [
            assignment["assigned_score"]
            for assignment
            in assignments_by_student[student_id]
            if assignment["auto_include"] == "Y"
        ]

        minimum_score = (
            min(scores)
            if scores
            else None
        )

        print(
            f"  {student_id}: {count} "
            f"(min score={minimum_score})"
        )

    print()

    if warnings:
        print("Warnings:")

        for warning in warnings:
            print(f"  {warning}")

        print()

    print(f"Labels CSV : {output_path}")
    print(f"Review     : {review_path}")
    print(f"Report     : {report_path}")
    print()
    print(
        "No Excel editing is required."
    )

if __name__ == "__main__":
    main()