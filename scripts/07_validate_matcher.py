import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.face_matcher import (
    FaceGalleryMatcher,
)


def classify_record(
    record,
    min_score,
    min_margin,
):
    if record["top_score"] < min_score:
        return "UNKNOWN"

    if record["margin"] < min_margin:
        return "AMBIGUOUS"

    return "MATCH"


def evaluate_configuration(
    records,
    min_score,
    min_margin,
):
    correct_matches = 0
    false_matches = 0
    ambiguous = 0
    unknown = 0

    for record in records:
        decision = classify_record(
            record=record,
            min_score=min_score,
            min_margin=min_margin,
        )

        if decision == "MATCH":
            if record["top_student_id"] == record[
                "true_student_id"
            ]:
                correct_matches += 1
            else:
                false_matches += 1

        elif decision == "AMBIGUOUS":
            ambiguous += 1

        else:
            unknown += 1

    return {
        "min_score": round(float(min_score), 2),
        "min_margin": round(float(min_margin), 2),
        "correct_matches": correct_matches,
        "false_matches": false_matches,
        "ambiguous": ambiguous,
        "unknown": unknown,
        "accepted": correct_matches + false_matches,
        "total": len(records),
        "correct_accept_rate": round(
            correct_matches / len(records),
            6,
        ),
    }


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    matcher_config = config["matcher"]
    gallery_path = ROOT / config["data"]["gallery_npz"]

    matcher = FaceGalleryMatcher(
        gallery_path=gallery_path,
        top_k=matcher_config["top_k"],
        min_score=matcher_config["min_score"],
        min_margin=matcher_config["min_margin"],
        max_candidates=matcher_config[
            "max_candidates"
        ],
    )

    gallery = np.load(
        gallery_path,
        allow_pickle=False,
    )

    embeddings = gallery[
        "image_embeddings"
    ].astype(np.float32)

    student_ids = gallery[
        "image_student_ids"
    ].astype(str).tolist()

    full_names = gallery[
        "image_full_names"
    ].astype(str).tolist()

    image_filenames = gallery[
        "image_filenames"
    ].astype(str).tolist()

    evaluable_records = []
    skipped_records = []

    for query_index, query_embedding in enumerate(
        embeddings
    ):
        true_student_id = student_ids[query_index]

        own_reference_count = (
            len(
                matcher.indices_by_student[
                    true_student_id
                ]
            )
            - 1
        )

        if own_reference_count == 0:
            skipped_records.append(
                {
                    "student_id": true_student_id,
                    "image_filename": (
                        image_filenames[query_index]
                    ),
                    "reason": (
                        "student_has_only_one_image"
                    ),
                }
            )
            continue

        candidates = matcher.rank_candidates(
            query_embedding=query_embedding,
            exclude_image_index=query_index,
        )

        best = candidates[0]
        second = candidates[1]

        margin = best.score - second.score

        true_candidate = next(
            candidate
            for candidate in candidates
            if candidate.student_id
            == true_student_id
        )

        evaluable_records.append(
            {
                "query_index": query_index,
                "true_student_id": true_student_id,
                "full_name": full_names[query_index],
                "image_filename": (
                    image_filenames[query_index]
                ),
                "top_student_id": best.student_id,
                "top_score": best.score,
                "second_student_id": (
                    second.student_id
                ),
                "second_score": second.score,
                "margin": margin,
                "true_score": true_candidate.score,
                "top_reference_images": (
                    best.top_reference_images
                ),
                "top_reference_scores": (
                    best.top_reference_scores
                ),
            }
        )

    raw_top1_correct = sum(
        record["top_student_id"]
        == record["true_student_id"]
        for record in evaluable_records
    )

    configured_evaluation = evaluate_configuration(
        records=evaluable_records,
        min_score=matcher.min_score,
        min_margin=matcher.min_margin,
    )

    grid_results = []

    score_thresholds = np.arange(
        0.45,
        0.851,
        0.01,
    )

    margin_thresholds = np.arange(
        0.00,
        0.201,
        0.01,
    )

    for min_score in score_thresholds:
        for min_margin in margin_thresholds:
            grid_results.append(
                evaluate_configuration(
                    records=evaluable_records,
                    min_score=min_score,
                    min_margin=min_margin,
                )
            )

    safe_configurations = [
        result
        for result in grid_results
        if result["false_matches"] == 0
    ]

    safe_configurations.sort(
        key=lambda result: (
            -result["correct_matches"],
            result["ambiguous"]
            + result["unknown"],
            -result["min_margin"],
            -result["min_score"],
        )
    )

    recommended = (
        safe_configurations[0]
        if safe_configurations
        else None
    )

    output_rows = []

    for record in evaluable_records:
        configured_decision = classify_record(
            record,
            matcher.min_score,
            matcher.min_margin,
        )

        output_rows.append(
            {
                "true_student_id": (
                    record["true_student_id"]
                ),
                "full_name": record["full_name"],
                "image_filename": (
                    record["image_filename"]
                ),
                "top_student_id": (
                    record["top_student_id"]
                ),
                "top_score": (
                    f"{record['top_score']:.6f}"
                ),
                "second_student_id": (
                    record["second_student_id"]
                ),
                "second_score": (
                    f"{record['second_score']:.6f}"
                ),
                "margin": (
                    f"{record['margin']:.6f}"
                ),
                "true_score": (
                    f"{record['true_score']:.6f}"
                ),
                "top1_correct": str(
                    record["top_student_id"]
                    == record["true_student_id"]
                ),
                "configured_decision": (
                    configured_decision
                ),
                "top_reference_images": (
                    "|".join(
                        record[
                            "top_reference_images"
                        ]
                    )
                ),
                "top_reference_scores": (
                    "|".join(
                        f"{score:.6f}"
                        for score in record[
                            "top_reference_scores"
                        ]
                    )
                ),
            }
        )

    reports_dir = ROOT / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        reports_dir
        / "matcher_validation.csv"
    )

    fieldnames = [
        "true_student_id",
        "full_name",
        "image_filename",
        "top_student_id",
        "top_score",
        "second_student_id",
        "second_score",
        "margin",
        "true_score",
        "top1_correct",
        "configured_decision",
        "top_reference_images",
        "top_reference_scores",
    ]

    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "strategy": "top_k_mean",
        "top_k": matcher.top_k,
        "evaluated": len(evaluable_records),
        "skipped": skipped_records,
        "raw_top1": {
            "correct": raw_top1_correct,
            "incorrect": (
                len(evaluable_records)
                - raw_top1_correct
            ),
            "accuracy": round(
                raw_top1_correct
                / len(evaluable_records),
                6,
            ),
        },
        "configured_thresholds": {
            "min_score": matcher.min_score,
            "min_margin": matcher.min_margin,
            "result": configured_evaluation,
        },
        "recommended_training_configuration": (
            recommended
        ),
        "top_safe_configurations": (
            safe_configurations[:20]
        ),
    }

    report_path = (
        reports_dir
        / "matcher_validation.json"
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

    print("=" * 70)
    print("MULTI-TEMPLATE MATCHER VALIDATION")
    print("=" * 70)
    print(
        f"Strategy                   : "
        f"Top-{matcher.top_k} mean"
    )
    print(
        f"Evaluated                  : "
        f"{len(evaluable_records)}"
    )
    print(
        f"Skipped                    : "
        f"{len(skipped_records)}"
    )
    print(
        f"Raw Top-1 correct          : "
        f"{raw_top1_correct}"
    )
    print(
        f"Raw Top-1 incorrect        : "
        f"{len(evaluable_records) - raw_top1_correct}"
    )
    print(
        f"Raw Top-1 accuracy         : "
        f"{raw_top1_correct / len(evaluable_records):.2%}"
    )

    print()
    print("Configured decision policy:")
    print(
        f"  Minimum score            : "
        f"{matcher.min_score}"
    )
    print(
        f"  Minimum margin           : "
        f"{matcher.min_margin}"
    )
    print(
        f"  Correct matches          : "
        f"{configured_evaluation['correct_matches']}"
    )
    print(
        f"  False matches            : "
        f"{configured_evaluation['false_matches']}"
    )
    print(
        f"  Ambiguous                : "
        f"{configured_evaluation['ambiguous']}"
    )
    print(
        f"  Unknown                  : "
        f"{configured_evaluation['unknown']}"
    )

    print()
    print("Incorrect raw Top-1 cases:")

    incorrect_records = [
        record
        for record in evaluable_records
        if record["top_student_id"]
        != record["true_student_id"]
    ]

    if not incorrect_records:
        print("  None")
    else:
        for record in incorrect_records:
            print(
                f"  {record['true_student_id']} | "
                f"{record['image_filename']} -> "
                f"{record['top_student_id']} | "
                f"top={record['top_score']:.6f} | "
                f"true={record['true_score']:.6f} | "
                f"margin={record['margin']:.6f}"
            )

    print()
    print("Recommended safe training configuration:")

    if recommended is None:
        print("  No zero-false-match configuration found.")
    else:
        print(
            f"  Minimum score            : "
            f"{recommended['min_score']}"
        )
        print(
            f"  Minimum margin           : "
            f"{recommended['min_margin']}"
        )
        print(
            f"  Correct matches          : "
            f"{recommended['correct_matches']}"
        )
        print(
            f"  False matches            : "
            f"{recommended['false_matches']}"
        )
        print(
            f"  Ambiguous                : "
            f"{recommended['ambiguous']}"
        )
        print(
            f"  Unknown                  : "
            f"{recommended['unknown']}"
        )

    print()
    print(f"CSV report : {csv_path}")
    print(f"JSON report: {report_path}")


if __name__ == "__main__":
    main()