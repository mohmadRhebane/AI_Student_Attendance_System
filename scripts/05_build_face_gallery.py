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

from smart_attendance_ai.arcface_recognizer import (
    ArcFaceRecognizer,
    l2_normalize,
)


def load_image(image_path: Path):
    try:
        encoded = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except Exception:
        return None


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest().upper()


def vector_to_json(vector: np.ndarray) -> str:
    return "[" + ",".join(
        f"{float(value):.8f}"
        for value in vector
    ) + "]"


def write_csv_atomic(
    output_path: Path,
    fieldnames: list,
    rows: list,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.tmp")

    with temporary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    temporary_path.replace(output_path)


def summarize_scores(values: list):
    if not values:
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }

    scores = np.asarray(values, dtype=np.float32)

    return {
        "count": int(scores.size),
        "minimum": round(float(scores.min()), 6),
        "p05": round(float(np.percentile(scores, 5)), 6),
        "mean": round(float(scores.mean()), 6),
        "median": round(float(np.median(scores)), 6),
        "p95": round(float(np.percentile(scores, 95)), 6),
        "p99": round(float(np.percentile(scores, 99)), 6),
        "maximum": round(float(scores.max()), 6),
    }


def calculate_similarity_analysis(
    embeddings: np.ndarray,
    student_ids: list,
    image_filenames: list,
):
    similarity_matrix = embeddings @ embeddings.T

    genuine_scores = []
    impostor_scores = []
    cross_student_pairs = []

    for first in range(len(embeddings)):
        for second in range(first + 1, len(embeddings)):
            score = float(
                similarity_matrix[first, second]
            )

            if student_ids[first] == student_ids[second]:
                genuine_scores.append(score)
            else:
                impostor_scores.append(score)

                cross_student_pairs.append(
                    {
                        "similarity": round(score, 6),
                        "first_student_id": student_ids[first],
                        "first_image": image_filenames[first],
                        "second_student_id": student_ids[second],
                        "second_image": image_filenames[second],
                    }
                )

    cross_student_pairs.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return {
        "genuine_same_student": summarize_scores(
            genuine_scores
        ),
        "impostor_different_students": summarize_scores(
            impostor_scores
        ),
        "top_cross_student_pairs": cross_student_pairs[:10],
    }


def main():
    config_path = ROOT / "configs" / "ai.yaml"

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    model_config = config["models"]["recognizer"]
    recognizer_config = config["recognizer"]
    data_config = config["data"]

    model_path = ROOT / model_config["path"]
    expected_hash = model_config["sha256"].upper()
    actual_hash = calculate_sha256(model_path)

    if actual_hash != expected_hash:
        raise RuntimeError(
            "Recognition model SHA256 does not match.\n"
            f"Expected: {expected_hash}\n"
            f"Actual  : {actual_hash}"
        )

    students_csv = ROOT / data_config["students_csv"]
    aligned_dir = ROOT / data_config["aligned_previews"]

    gallery_path = ROOT / data_config["gallery_npz"]
    face_embeddings_csv = (
        ROOT / data_config["face_embeddings_csv"]
    )
    student_templates_csv = (
        ROOT / data_config["student_templates_csv"]
    )
    report_path = ROOT / data_config["gallery_report"]

    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with students_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    aligned_faces = []
    image_student_ids = []
    image_full_names = []
    image_filenames = []

    for row in rows:
        student_id = row["student_id"].strip()
        full_name = row["full_name"].strip()
        image_filename = row["image_filename"].strip()

        aligned_path = aligned_dir / image_filename
        aligned_face = load_image(aligned_path)

        if aligned_face is None:
            raise RuntimeError(
                f"Aligned face is missing or unreadable: "
                f"{aligned_path}"
            )

        if aligned_face.shape != (112, 112, 3):
            raise RuntimeError(
                f"Expected aligned face size 112x112, but "
                f"{image_filename} has shape "
                f"{aligned_face.shape}."
            )

        aligned_faces.append(aligned_face)
        image_student_ids.append(student_id)
        image_full_names.append(full_name)
        image_filenames.append(image_filename)

    recognizer = ArcFaceRecognizer(
        model_path=model_path,
        input_size=(
            recognizer_config["input_width"],
            recognizer_config["input_height"],
        ),
        embedding_size=recognizer_config["embedding_size"],
        batch_size=recognizer_config["batch_size"],
        normalize=recognizer_config["l2_normalize"],
    )

    print("=" * 70)
    print("ARCFACE FACE GALLERY BUILDER")
    print("=" * 70)
    print(f"Model       : {model_path}")
    print(f"Model SHA256: {actual_hash}")
    print(f"Providers   : {recognizer.providers}")
    print(f"Faces       : {len(aligned_faces)}")
    print(f"Batch size  : {recognizer.batch_size}")
    print()

    print("Warming up ArcFace on GPU...")
    recognizer.warm_up()

    print("Generating embeddings...")

    start_time = time.perf_counter()

    image_embeddings = recognizer.embed_faces(
        aligned_faces
    )

    embedding_time_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    embedding_norms = np.linalg.norm(
        image_embeddings,
        axis=1,
    )

    if not np.allclose(
        embedding_norms,
        1.0,
        atol=1e-5,
    ):
        raise RuntimeError(
            "One or more embeddings are not L2-normalized."
        )

    ordered_student_ids = list(
        dict.fromkeys(image_student_ids)
    )

    student_names = []
    student_templates = []
    student_image_counts = []

    for student_id in ordered_student_ids:
        indices = [
            index
            for index, current_student_id
            in enumerate(image_student_ids)
            if current_student_id == student_id
        ]

        student_embeddings = image_embeddings[indices]

        template = l2_normalize(
            student_embeddings.mean(
                axis=0,
                keepdims=True,
            )
        )[0]

        matching_names = {
            image_full_names[index]
            for index in indices
        }

        if len(matching_names) != 1:
            raise RuntimeError(
                f"Student {student_id} has inconsistent names: "
                f"{matching_names}"
            )

        student_names.append(next(iter(matching_names)))
        student_templates.append(template)
        student_image_counts.append(len(indices))

    student_templates = np.stack(
        student_templates,
        axis=0,
    ).astype(np.float32)

    similarity_analysis = calculate_similarity_analysis(
        embeddings=image_embeddings,
        student_ids=image_student_ids,
        image_filenames=image_filenames,
    )

    model_name = recognizer_config["model_name"]
    embedding_dimension = int(
        image_embeddings.shape[1]
    )

    temporary_gallery_path = gallery_path.with_name(
        gallery_path.stem + ".tmp.npz"
    )

    np.savez_compressed(
        temporary_gallery_path,
        schema_version=np.asarray("1.0"),
        model_name=np.asarray(model_name),
        model_sha256=np.asarray(actual_hash),
        embedding_dimension=np.asarray(
            embedding_dimension,
            dtype=np.int32,
        ),
        image_embeddings=image_embeddings,
        image_student_ids=np.asarray(
            image_student_ids,
            dtype=np.str_,
        ),
        image_full_names=np.asarray(
            image_full_names,
            dtype=np.str_,
        ),
        image_filenames=np.asarray(
            image_filenames,
            dtype=np.str_,
        ),
        student_templates=student_templates,
        student_ids=np.asarray(
            ordered_student_ids,
            dtype=np.str_,
        ),
        student_names=np.asarray(
            student_names,
            dtype=np.str_,
        ),
        student_image_counts=np.asarray(
            student_image_counts,
            dtype=np.int32,
        ),
    )

    temporary_gallery_path.replace(gallery_path)

    face_embedding_rows = []

    for index, embedding in enumerate(image_embeddings):
        face_embedding_rows.append(
            {
                "student_id": image_student_ids[index],
                "full_name": image_full_names[index],
                "image_filename": image_filenames[index],
                "model_name": model_name,
                "model_sha256": actual_hash,
                "embedding_dimension": embedding_dimension,
                "embedding_norm": (
                    f"{float(embedding_norms[index]):.8f}"
                ),
                "embedding_json": vector_to_json(
                    embedding
                ),
            }
        )

    write_csv_atomic(
        face_embeddings_csv,
        fieldnames=[
            "student_id",
            "full_name",
            "image_filename",
            "model_name",
            "model_sha256",
            "embedding_dimension",
            "embedding_norm",
            "embedding_json",
        ],
        rows=face_embedding_rows,
    )

    student_template_rows = []

    for index, template in enumerate(student_templates):
        student_template_rows.append(
            {
                "student_id": ordered_student_ids[index],
                "full_name": student_names[index],
                "model_name": model_name,
                "model_sha256": actual_hash,
                "embedding_dimension": embedding_dimension,
                "image_count": student_image_counts[index],
                "template_norm": (
                    f"{float(np.linalg.norm(template)):.8f}"
                ),
                "embedding_json": vector_to_json(
                    template
                ),
            }
        )

    write_csv_atomic(
        student_templates_csv,
        fieldnames=[
            "student_id",
            "full_name",
            "model_name",
            "model_sha256",
            "embedding_dimension",
            "image_count",
            "template_norm",
            "embedding_json",
        ],
        rows=student_template_rows,
    )

    batch_sizes = []

    for start in range(
        0,
        len(aligned_faces),
        recognizer.batch_size,
    ):
        batch_sizes.append(
            min(
                recognizer.batch_size,
                len(aligned_faces) - start,
            )
        )

    report = {
        "schema_version": "1.0",
        "model_name": model_name,
        "model_path": str(model_path.relative_to(ROOT)),
        "model_sha256": actual_hash,
        "providers": recognizer.providers,
        "embedding_dimension": embedding_dimension,
        "batch_size": recognizer.batch_size,
        "executed_batch_sizes": batch_sizes,
        "total_images": len(image_embeddings),
        "total_students": len(student_templates),
        "embedding_time_ms": round(
            embedding_time_ms,
            3,
        ),
        "average_embedding_time_ms": round(
            embedding_time_ms / len(image_embeddings),
            3,
        ),
        "embedding_norm_min": round(
            float(embedding_norms.min()),
            8,
        ),
        "embedding_norm_max": round(
            float(embedding_norms.max()),
            8,
        ),
        "images_per_student": {
            ordered_student_ids[index]: (
                student_image_counts[index]
            )
            for index in range(len(ordered_student_ids))
        },
        "similarity_analysis": similarity_analysis,
        "outputs": {
            "gallery_npz": str(
                gallery_path.relative_to(ROOT)
            ),
            "face_embeddings_csv": str(
                face_embeddings_csv.relative_to(ROOT)
            ),
            "student_templates_csv": str(
                student_templates_csv.relative_to(ROOT)
            ),
        },
    }

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    genuine = similarity_analysis[
        "genuine_same_student"
    ]

    impostor = similarity_analysis[
        "impostor_different_students"
    ]

    print()
    print("=" * 70)
    print("FACE GALLERY SUMMARY")
    print("=" * 70)
    print(f"Images                 : {len(image_embeddings)}")
    print(f"Students               : {len(student_templates)}")
    print(f"Embedding shape        : {image_embeddings.shape}")
    print(f"Template shape         : {student_templates.shape}")
    print(f"Executed batches       : {batch_sizes}")
    print(f"Total GPU embedding ms : {embedding_time_ms:.3f}")
    print(
        f"Average ms/face        : "
        f"{embedding_time_ms / len(image_embeddings):.3f}"
    )
    print(
        f"Embedding norms        : "
        f"{embedding_norms.min():.8f} - "
        f"{embedding_norms.max():.8f}"
    )
    print()
    print("Similarity statistics:")
    print(
        f"  Genuine pairs        : {genuine['count']}"
    )
    print(
        f"  Genuine min/mean/max : "
        f"{genuine['minimum']} / "
        f"{genuine['mean']} / "
        f"{genuine['maximum']}"
    )
    print(
        f"  Impostor pairs       : {impostor['count']}"
    )
    print(
        f"  Impostor min/mean/max: "
        f"{impostor['minimum']} / "
        f"{impostor['mean']} / "
        f"{impostor['maximum']}"
    )

    print()
    print("Highest cross-student similarities:")

    for pair in similarity_analysis[
        "top_cross_student_pairs"
    ][:5]:
        print(
            f"  {pair['similarity']:.6f} | "
            f"{pair['first_student_id']} "
            f"({pair['first_image']}) <-> "
            f"{pair['second_student_id']} "
            f"({pair['second_image']})"
        )

    print()
    print(f"Gallery     : {gallery_path}")
    print(f"Image CSV   : {face_embeddings_csv}")
    print(f"Template CSV: {student_templates_csv}")
    print(f"Report      : {report_path}")
    print()
    print("FACE GALLERY BUILD PASSED")


if __name__ == "__main__":
    main()