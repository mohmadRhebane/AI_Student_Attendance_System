import cv2
import numpy as np


ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_similarity_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_points, dtype=np.float64)
    destination = np.asarray(destination_points, dtype=np.float64)

    if source.shape != (5, 2):
        raise ValueError(
            f"Expected source landmarks shape (5, 2), got {source.shape}."
        )

    if destination.shape != (5, 2):
        raise ValueError(
            f"Expected destination landmarks shape (5, 2), "
            f"got {destination.shape}."
        )

    if not np.isfinite(source).all():
        raise ValueError("Source landmarks contain invalid values.")

    if not np.isfinite(destination).all():
        raise ValueError("Destination landmarks contain invalid values.")

    number_of_points = source.shape[0]

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)

    centered_source = source - source_mean
    centered_destination = destination - destination_mean

    source_variance = (
        np.sum(centered_source * centered_source) / number_of_points
    )

    if source_variance <= 1e-12:
        raise ValueError("Face landmarks are degenerate.")

    covariance = (
        centered_destination.T @ centered_source
    ) / number_of_points

    left_matrix, singular_values, right_matrix_transposed = np.linalg.svd(
        covariance
    )

    correction = np.ones(2, dtype=np.float64)

    if (
        np.linalg.det(left_matrix)
        * np.linalg.det(right_matrix_transposed)
        < 0
    ):
        correction[-1] = -1.0

    rotation = (
        left_matrix
        @ np.diag(correction)
        @ right_matrix_transposed
    )

    scale = (
        np.sum(singular_values * correction)
        / source_variance
    )

    translation = (
        destination_mean
        - scale * (rotation @ source_mean)
    )

    transform = np.zeros((2, 3), dtype=np.float32)
    transform[:, :2] = (scale * rotation).astype(np.float32)
    transform[:, 2] = translation.astype(np.float32)

    if not np.isfinite(transform).all():
        raise ValueError("Calculated alignment transform is invalid.")

    return transform


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> tuple:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    landmarks = np.asarray(landmarks, dtype=np.float32)

    if landmarks.shape != (5, 2):
        raise ValueError(
            f"Expected landmarks shape (5, 2), got {landmarks.shape}."
        )

    destination = ARCFACE_TEMPLATE_112.copy()

    if output_size != 112:
        destination *= float(output_size) / 112.0

    transform = estimate_similarity_transform(
        landmarks,
        destination,
    )

    aligned_face = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    if aligned_face.shape != (output_size, output_size, 3):
        raise RuntimeError(
            f"Unexpected aligned face shape: {aligned_face.shape}"
        )

    return aligned_face, transform


def calculate_face_quality(aligned_face: np.ndarray) -> dict:
    if aligned_face is None or aligned_face.size == 0:
        raise ValueError("Aligned face is empty.")

    gray = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY)

    blur_score = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    brightness = float(gray.mean())
    contrast = float(gray.std())

    dark_ratio = float(np.mean(gray < 30))
    bright_ratio = float(np.mean(gray > 225))

    return {
        "blur_score": round(blur_score, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "dark_ratio": round(dark_ratio, 6),
        "bright_ratio": round(bright_ratio, 6),
    }