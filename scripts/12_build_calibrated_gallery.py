import csv
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
    l2_normalize,
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


def calculate_cross_student_pairs(
    embeddings,
    student_ids,
    filenames,
    sources,
):
    similarity_matrix = embeddings @ embeddings.T
    pairs = []

    for first in range(len(embeddings)):
        for second in range(
            first + 1,
            len(embeddings),
        ):
            if (
                student_ids[first]
                == student_ids[second]
            ):
                continue

            pairs.append(
                {
                    "similarity": round(
                        float(
                            similarity_matrix[
                                first,
                                second,
                            ]
                        ),
                        6,
                    ),
                    "first_student_id": (
                        student_ids[first]
                    ),
                    "first_filename": (
                        filenames[first]
                    ),
                    "first_source": (
                        sources[first]
                    ),
                    "second_student_id": (
                        student_ids[second]
                    ),
                    "second_filename": (
                        filenames[second]
                    ),
                    "second_source": (
                        sources[second]
                    ),
                }
            )

    pairs.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return pairs[:20]


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    data_config = config["data"]
    recognizer_config = config["recognizer"]

    original_gallery_path = (
        ROOT / data_config["gallery_npz"]
    )

    labels_path = (
        ROOT
        / "data"
        / "generated"
        / "auto_calibration_labels.csv"
    )

    calibration_images_dir = (
        ROOT
        / "data"
        / "previews"
        / "calibration_candidates"
        / "aligned"
    )

    output_gallery_path = (
        ROOT
        / data_config["calibrated_gallery_npz"]
    )

    output_manifest_path = (
        ROOT
        / data_config[
            "calibrated_gallery_manifest"
        ]
    )

    output_report_path = (
        ROOT
        / data_config[
            "calibrated_gallery_report"
        ]
    )

    original_gallery = np.load(
        original_gallery_path,
        allow_pickle=False,
    )

    original_embeddings = normalize_rows(
        original_gallery["image_embeddings"]
    )

    original_student_ids = (
        original_gallery[
            "image_student_ids"
        ].astype(str).tolist()
    )

    original_full_names = (
        original_gallery[
            "image_full_names"
        ].astype(str).tolist()
    )

    original_filenames = (
        original_gallery[
            "image_filenames"
        ].astype(str).tolist()
    )

    ordered_student_ids = (
        original_gallery[
            "student_ids"
        ].astype(str).tolist()
    )

    ordered_student_names = (
        original_gallery[
            "student_names"
        ].astype(str).tolist()
    )

    student_name_map = dict(
        zip(
            ordered_student_ids,
            ordered_student_names,
        )
    )

    with labels_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        label_rows = list(
            csv.DictReader(file)
        )

    selected_rows = [
        row
        for row in label_rows
        if row["auto_include"].strip().upper()
        == "Y"
    ]

    if not selected_rows:
        raise RuntimeError(
            "No automatically selected "
            "calibration images were found."
        )

    invalid_student_ids = sorted(
        {
            row["auto_student_id"]
            for row in selected_rows
            if row["auto_student_id"]
            not in student_name_map
        }
    )

    if invalid_student_ids:
        raise RuntimeError(
            f"Unknown calibration student IDs: "
            f"{invalid_student_ids}"
        )

    if any(
        row["auto_student_id"] == "STU_001"
        for row in selected_rows
    ):
        raise RuntimeError(
            "STU_001 is absent and must not have "
            "camera calibration images."
        )

    calibration_faces = []

    for row in selected_rows:
        image_path = (
            calibration_images_dir
            / row["aligned_filename"]
        )

        image = load_image(image_path)

        if image is None:
            raise RuntimeError(
                f"Cannot read calibration face: "
                f"{image_path}"
            )

        if image.shape != (112, 112, 3):
            raise RuntimeError(
                f"Invalid calibration face shape "
                f"{image.shape}: {image_path}"
            )

        calibration_faces.append(image)

    recognizer = ArcFaceRecognizer(
        model_path=(
            ROOT
            / config["models"]["recognizer"][
                "path"
            ]
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
    print("CALIBRATED FACE GALLERY BUILDER")
    print("=" * 70)
    print(
        f"Original embeddings   : "
        f"{len(original_embeddings)}"
    )
    print(
        f"Calibration embeddings: "
        f"{len(calibration_faces)}"
    )
    print(
        f"Providers             : "
        f"{recognizer.providers}"
    )
    print()

    print("Warming up ArcFace...")
    recognizer.warm_up()

    print(
        "Generating camera calibration "
        "embeddings..."
    )

    start_time = time.perf_counter()

    calibration_embeddings = (
        recognizer.embed_faces(
            calibration_faces
        )
    )

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    calibration_student_ids = [
        row["auto_student_id"]
        for row in selected_rows
    ]

    calibration_full_names = [
        student_name_map[
            row["auto_student_id"]
        ]
        for row in selected_rows
    ]

    calibration_filenames = [
        row["aligned_filename"]
        for row in selected_rows
    ]

    calibration_candidate_ids = [
        row["candidate_id"]
        for row in selected_rows
    ]

    combined_embeddings = np.concatenate(
        [
            original_embeddings,
            calibration_embeddings,
        ],
        axis=0,
    ).astype(np.float32)

    combined_embeddings = normalize_rows(
        combined_embeddings
    )

    combined_student_ids = (
        original_student_ids
        + calibration_student_ids
    )

    combined_full_names = (
        original_full_names
        + calibration_full_names
    )

    combined_filenames = (
        original_filenames
        + calibration_filenames
    )

    combined_sources = (
        ["enrollment"] * len(
            original_embeddings
        )
        + ["camera_calibration"] * len(
            calibration_embeddings
        )
    )

    combined_candidate_ids = (
        [""] * len(original_embeddings)
        + calibration_candidate_ids
    )

    student_templates = []
    student_image_counts = []

    for student_id in ordered_student_ids:
        indices = [
            index
            for index, current_student_id
            in enumerate(combined_student_ids)
            if current_student_id == student_id
        ]

        if not indices:
            raise RuntimeError(
                f"No embeddings for {student_id}."
            )

        template = l2_normalize(
            combined_embeddings[
                indices
            ].mean(
                axis=0,
                keepdims=True,
            )
        )[0]

        student_templates.append(template)
        student_image_counts.append(
            len(indices)
        )

    student_templates = np.stack(
        student_templates,
        axis=0,
    ).astype(np.float32)

    output_gallery_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_gallery_path = (
        output_gallery_path.with_name(
            output_gallery_path.stem
            + ".tmp.npz"
        )
    )

    np.savez_compressed(
        temporary_gallery_path,
        schema_version=np.asarray("1.1"),
        model_name=original_gallery[
            "model_name"
        ],
        model_sha256=original_gallery[
            "model_sha256"
        ],
        embedding_dimension=np.asarray(
            combined_embeddings.shape[1],
            dtype=np.int32,
        ),
        image_embeddings=combined_embeddings,
        image_student_ids=np.asarray(
            combined_student_ids,
            dtype=np.str_,
        ),
        image_full_names=np.asarray(
            combined_full_names,
            dtype=np.str_,
        ),
        image_filenames=np.asarray(
            combined_filenames,
            dtype=np.str_,
        ),
        image_sources=np.asarray(
            combined_sources,
            dtype=np.str_,
        ),
        image_candidate_ids=np.asarray(
            combined_candidate_ids,
            dtype=np.str_,
        ),
        student_templates=student_templates,
        student_ids=np.asarray(
            ordered_student_ids,
            dtype=np.str_,
        ),
        student_names=np.asarray(
            ordered_student_names,
            dtype=np.str_,
        ),
        student_image_counts=np.asarray(
            student_image_counts,
            dtype=np.int32,
        ),
    )

    temporary_gallery_path.replace(
        output_gallery_path
    )

    output_manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_fieldnames = [
        "embedding_index",
        "student_id",
        "full_name",
        "source",
        "image_filename",
        "candidate_id",
    ]

    temporary_manifest_path = (
        output_manifest_path.with_suffix(
            ".csv.tmp"
        )
    )

    with temporary_manifest_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=manifest_fieldnames,
        )

        writer.writeheader()

        for index in range(
            len(combined_embeddings)
        ):
            writer.writerow(
                {
                    "embedding_index": index,
                    "student_id": (
                        combined_student_ids[index]
                    ),
                    "full_name": (
                        combined_full_names[index]
                    ),
                    "source": (
                        combined_sources[index]
                    ),
                    "image_filename": (
                        combined_filenames[index]
                    ),
                    "candidate_id": (
                        combined_candidate_ids[
                            index
                        ]
                    ),
                }
            )

    temporary_manifest_path.replace(
        output_manifest_path
    )

    source_counts = Counter(
        combined_sources
    )

    calibration_counts = Counter(
        calibration_student_ids
    )

    highest_cross_pairs = (
        calculate_cross_student_pairs(
            embeddings=combined_embeddings,
            student_ids=combined_student_ids,
            filenames=combined_filenames,
            sources=combined_sources,
        )
    )

    report = {
        "schema_version": "1.1",
        "original_gallery": str(
            original_gallery_path.relative_to(
                ROOT
            )
        ),
        "calibrated_gallery": str(
            output_gallery_path.relative_to(
                ROOT
            )
        ),
        "original_embeddings": len(
            original_embeddings
        ),
        "calibration_embeddings": len(
            calibration_embeddings
        ),
        "total_embeddings": len(
            combined_embeddings
        ),
        "total_students": len(
            ordered_student_ids
        ),
        "embedding_dimension": int(
            combined_embeddings.shape[1]
        ),
        "arcface_time_ms": round(
            elapsed_ms,
            3,
        ),
        "source_counts": dict(source_counts),
        "calibration_per_student": {
            student_id: calibration_counts.get(
                student_id,
                0,
            )
            for student_id in ordered_student_ids
        },
        "total_per_student": {
            ordered_student_ids[index]: (
                student_image_counts[index]
            )
            for index in range(
                len(ordered_student_ids)
            )
        },
        "highest_cross_student_pairs": (
            highest_cross_pairs
        ),
        "manifest": str(
            output_manifest_path.relative_to(
                ROOT
            )
        ),
    }

    output_report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_report_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    norms = np.linalg.norm(
        combined_embeddings,
        axis=1,
    )

    print()
    print("=" * 70)
    print("CALIBRATED GALLERY SUMMARY")
    print("=" * 70)
    print(
        f"Original embeddings   : "
        f"{len(original_embeddings)}"
    )
    print(
        f"Camera embeddings     : "
        f"{len(calibration_embeddings)}"
    )
    print(
        f"Total embeddings      : "
        f"{len(combined_embeddings)}"
    )
    print(
        f"Students              : "
        f"{len(ordered_student_ids)}"
    )
    print(
        f"Embedding shape       : "
        f"{combined_embeddings.shape}"
    )
    print(
        f"Template shape        : "
        f"{student_templates.shape}"
    )
    print(
        f"Norm range            : "
        f"{norms.min():.8f} - "
        f"{norms.max():.8f}"
    )
    print(
        f"ArcFace time          : "
        f"{elapsed_ms:.3f} ms"
    )
    print(f"Gallery               : {output_gallery_path}")
    print(f"Manifest              : {output_manifest_path}")
    print(f"Report                : {output_report_path}")
    print()
    print("CALIBRATED GALLERY BUILD PASSED")


if __name__ == "__main__":
    main()