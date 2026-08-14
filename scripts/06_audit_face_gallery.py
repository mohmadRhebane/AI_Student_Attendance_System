import csv
import json
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


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))

    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector.")

    return vector / norm


def load_image(image_path: Path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_image(image_path: Path, image: np.ndarray):
    image_path.parent.mkdir(parents=True, exist_ok=True)

    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError(f"Cannot encode {image_path}")

    encoded.tofile(str(image_path))


def pair_to_json(pair: dict) -> dict:
    return {
        "similarity": round(pair["similarity"], 6),
        "first_student_id": pair["first_student_id"],
        "first_image": pair["first_image"],
        "second_student_id": pair["second_student_id"],
        "second_image": pair["second_image"],
    }


def create_pair_sheet(
    pairs: list,
    aligned_dir: Path,
    output_path: Path,
    title: str,
):
    selected_pairs = pairs[:10]

    if not selected_pairs:
        return

    width = 440
    title_height = 45
    row_height = 155
    face_size = 112

    sheet = np.full(
        (
            title_height + row_height * len(selected_pairs),
            width,
            3,
        ),
        245,
        dtype=np.uint8,
    )

    cv2.putText(
        sheet,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    for row_index, pair in enumerate(selected_pairs):
        y = title_height + row_index * row_height

        first_image = load_image(
            aligned_dir / pair["first_image"]
        )
        second_image = load_image(
            aligned_dir / pair["second_image"]
        )

        if first_image is None or second_image is None:
            continue

        first_image = cv2.resize(
            first_image,
            (face_size, face_size),
        )
        second_image = cv2.resize(
            second_image,
            (face_size, face_size),
        )

        sheet[y + 5:y + 117, 10:122] = first_image
        sheet[y + 5:y + 117, 318:430] = second_image

        cv2.putText(
            sheet,
            pair["first_student_id"],
            (10, y + 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            pair["second_student_id"],
            (318, y + 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            f"score",
            (180, y + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            sheet,
            f"{pair['similarity']:.4f}",
            (165, y + 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )

    save_image(output_path, sheet)


def summarize(values: list):
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "mean": None,
            "maximum": None,
        }

    array = np.asarray(values, dtype=np.float32)

    return {
        "count": int(array.size),
        "minimum": round(float(array.min()), 6),
        "mean": round(float(array.mean()), 6),
        "maximum": round(float(array.max()), 6),
    }


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    gallery_path = ROOT / config["data"]["gallery_npz"]
    aligned_dir = ROOT / config["data"]["aligned_previews"]

    gallery = np.load(
        gallery_path,
        allow_pickle=False,
    )

    embeddings = gallery["image_embeddings"].astype(
        np.float32
    )
    student_ids = gallery["image_student_ids"].astype(
        str
    ).tolist()
    full_names = gallery["image_full_names"].astype(
        str
    ).tolist()
    image_filenames = gallery["image_filenames"].astype(
        str
    ).tolist()

    if embeddings.ndim != 2 or embeddings.shape[1] != 512:
        raise RuntimeError(
            f"Invalid gallery shape: {embeddings.shape}"
        )

    similarity_matrix = embeddings @ embeddings.T

    same_student_pairs = []
    cross_student_pairs = []
    pair_scores_by_student = defaultdict(list)

    for first in range(len(embeddings)):
        for second in range(first + 1, len(embeddings)):
            score = float(
                similarity_matrix[first, second]
            )

            pair = {
                "similarity": score,
                "first_index": first,
                "second_index": second,
                "first_student_id": student_ids[first],
                "first_image": image_filenames[first],
                "second_student_id": student_ids[second],
                "second_image": image_filenames[second],
            }

            if student_ids[first] == student_ids[second]:
                same_student_pairs.append(pair)
                pair_scores_by_student[
                    student_ids[first]
                ].append(score)
            else:
                cross_student_pairs.append(pair)

    same_student_pairs.sort(
        key=lambda item: item["similarity"]
    )

    cross_student_pairs.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    indices_by_student = defaultdict(list)

    for index, student_id in enumerate(student_ids):
        indices_by_student[student_id].append(index)

    leave_one_out_rows = []
    correct_count = 0
    evaluated_count = 0
    unevaluated_count = 0
    true_scores = []
    best_impostor_scores = []

    ordered_students = list(indices_by_student.keys())

    for query_index, query_embedding in enumerate(embeddings):
        true_student_id = student_ids[query_index]
        candidate_scores = {}

        for candidate_student_id in ordered_students:
            candidate_indices = list(
                indices_by_student[candidate_student_id]
            )

            if candidate_student_id == true_student_id:
                candidate_indices = [
                    index
                    for index in candidate_indices
                    if index != query_index
                ]

            if not candidate_indices:
                continue

            candidate_template = normalize_vector(
                embeddings[candidate_indices].mean(axis=0)
            )

            candidate_scores[candidate_student_id] = float(
                query_embedding @ candidate_template
            )

        true_score = candidate_scores.get(true_student_id)

        if true_score is None:
            unevaluated_count += 1

            leave_one_out_rows.append(
                {
                    "student_id": true_student_id,
                    "full_name": full_names[query_index],
                    "image_filename": image_filenames[query_index],
                    "status": "NOT_EVALUATED",
                    "predicted_student_id": "",
                    "true_score": "",
                    "best_impostor_score": "",
                    "margin": "",
                    "correct": "",
                    "reason": "student_has_only_one_image",
                }
            )

            continue

        ranked_candidates = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        predicted_student_id, predicted_score = (
            ranked_candidates[0]
        )

        impostor_candidates = [
            (student_id, score)
            for student_id, score in ranked_candidates
            if student_id != true_student_id
        ]

        best_impostor_id, best_impostor_score = (
            impostor_candidates[0]
        )

        margin = true_score - best_impostor_score
        correct = predicted_student_id == true_student_id

        evaluated_count += 1
        correct_count += int(correct)
        true_scores.append(true_score)
        best_impostor_scores.append(best_impostor_score)

        leave_one_out_rows.append(
            {
                "student_id": true_student_id,
                "full_name": full_names[query_index],
                "image_filename": image_filenames[query_index],
                "status": "EVALUATED",
                "predicted_student_id": predicted_student_id,
                "true_score": f"{true_score:.6f}",
                "best_impostor_student_id": best_impostor_id,
                "best_impostor_score": (
                    f"{best_impostor_score:.6f}"
                ),
                "margin": f"{margin:.6f}",
                "correct": str(correct),
                "reason": (
                    "correct"
                    if correct
                    else "wrong_top1_identity"
                ),
            }
        )

    accuracy = (
        correct_count / evaluated_count
        if evaluated_count
        else 0.0
    )

    minimum_true_score = (
        min(true_scores)
        if true_scores
        else None
    )

    maximum_best_impostor = (
        max(best_impostor_scores)
        if best_impostor_scores
        else None
    )

    leave_one_out_gap = (
        minimum_true_score - maximum_best_impostor
        if (
            minimum_true_score is not None
            and maximum_best_impostor is not None
        )
        else None
    )

    per_student_statistics = {}

    for student_id in ordered_students:
        values = pair_scores_by_student[student_id]

        per_student_statistics[student_id] = {
            "image_count": len(
                indices_by_student[student_id]
            ),
            "same_student_similarity": summarize(values),
        }

    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    strongest_cross_path = (
        reports_dir
        / "strongest_cross_student_pairs.jpg"
    )

    weakest_same_path = (
        reports_dir
        / "weakest_same_student_pairs.jpg"
    )

    create_pair_sheet(
        pairs=cross_student_pairs,
        aligned_dir=aligned_dir,
        output_path=strongest_cross_path,
        title="Strongest cross-student pairs",
    )

    create_pair_sheet(
        pairs=same_student_pairs,
        aligned_dir=aligned_dir,
        output_path=weakest_same_path,
        title="Weakest same-student pairs",
    )

    csv_path = (
        reports_dir
        / "leave_one_out_results.csv"
    )

    fieldnames = [
        "student_id",
        "full_name",
        "image_filename",
        "status",
        "predicted_student_id",
        "true_score",
        "best_impostor_student_id",
        "best_impostor_score",
        "margin",
        "correct",
        "reason",
    ]

    with csv_path.open(
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
        writer.writerows(leave_one_out_rows)

    report = {
        "gallery": str(gallery_path.relative_to(ROOT)),
        "images": len(embeddings),
        "students": len(ordered_students),
        "pair_analysis": {
            "same_student": summarize(
                [
                    pair["similarity"]
                    for pair in same_student_pairs
                ]
            ),
            "different_students": summarize(
                [
                    pair["similarity"]
                    for pair in cross_student_pairs
                ]
            ),
            "weakest_same_student_pairs": [
                pair_to_json(pair)
                for pair in same_student_pairs[:10]
            ],
            "strongest_cross_student_pairs": [
                pair_to_json(pair)
                for pair in cross_student_pairs[:10]
            ],
        },
        "leave_one_out": {
            "evaluated": evaluated_count,
            "not_evaluated": unevaluated_count,
            "correct": correct_count,
            "incorrect": evaluated_count - correct_count,
            "top1_accuracy": round(accuracy, 6),
            "minimum_true_score": (
                round(minimum_true_score, 6)
                if minimum_true_score is not None
                else None
            ),
            "maximum_best_impostor_score": (
                round(maximum_best_impostor, 6)
                if maximum_best_impostor is not None
                else None
            ),
            "separation_gap": (
                round(leave_one_out_gap, 6)
                if leave_one_out_gap is not None
                else None
            ),
        },
        "per_student_statistics": (
            per_student_statistics
        ),
        "outputs": {
            "leave_one_out_csv": str(
                csv_path.relative_to(ROOT)
            ),
            "strongest_cross_student_sheet": str(
                strongest_cross_path.relative_to(ROOT)
            ),
            "weakest_same_student_sheet": str(
                weakest_same_path.relative_to(ROOT)
            ),
        },
    }

    report_path = reports_dir / "gallery_audit.json"

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

    print("=" * 70)
    print("FACE GALLERY AUDIT")
    print("=" * 70)
    print(f"Images                    : {len(embeddings)}")
    print(f"Students                  : {len(ordered_students)}")
    print(f"Evaluated images          : {evaluated_count}")
    print(f"Not evaluated             : {unevaluated_count}")
    print(f"Correct Top-1             : {correct_count}")
    print(
        f"Incorrect Top-1           : "
        f"{evaluated_count - correct_count}"
    )
    print(f"Leave-One-Out accuracy    : {accuracy:.2%}")
    print(
        f"Minimum true score        : "
        f"{minimum_true_score}"
    )
    print(
        f"Maximum best impostor     : "
        f"{maximum_best_impostor}"
    )
    print(
        f"Separation gap            : "
        f"{leave_one_out_gap}"
    )

    print()
    print("Incorrect recognition cases:")

    incorrect_rows = [
        row
        for row in leave_one_out_rows
        if row.get("correct") == "False"
    ]

    if not incorrect_rows:
        print("  None")
    else:
        for row in incorrect_rows:
            print(
                f"  {row['student_id']} | "
                f"{row['image_filename']} -> "
                f"{row['predicted_student_id']} | "
                f"true={row['true_score']} | "
                f"impostor={row['best_impostor_score']} | "
                f"margin={row['margin']}"
            )

    print()
    print("Strongest cross-student pairs:")

    for pair in cross_student_pairs[:10]:
        print(
            f"  {pair['similarity']:.6f} | "
            f"{pair['first_student_id']} "
            f"({pair['first_image']}) <-> "
            f"{pair['second_student_id']} "
            f"({pair['second_image']})"
        )

    print()
    print(f"Audit report       : {report_path}")
    print(f"Leave-One-Out CSV  : {csv_path}")
    print(f"Cross-student sheet: {strongest_cross_path}")
    print(f"Same-student sheet : {weakest_same_path}")


if __name__ == "__main__":
    main()