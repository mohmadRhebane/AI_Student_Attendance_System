from dataclasses import dataclass

import cv2
import numpy as np

from smart_attendance_ai.gpu_runtime import create_gpu_session


@dataclass
class FaceDetection:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


def distance_to_bbox(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    x1 = points[:, 0] - distances[:, 0]
    y1 = points[:, 1] - distances[:, 1]
    x2 = points[:, 0] + distances[:, 2]
    y2 = points[:, 1] + distances[:, 3]
    return np.stack([x1, y1, x2, y2], axis=1)


def distance_to_landmarks(points: np.ndarray, distances: np.ndarray) -> np.ndarray:
    landmarks = np.empty_like(distances, dtype=np.float32)

    for index in range(0, distances.shape[1], 2):
        landmarks[:, index] = points[:, 0] + distances[:, index]
        landmarks[:, index + 1] = points[:, 1] + distances[:, index + 1]

    return landmarks


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        xx1 = np.maximum(x1[current], x1[remaining])
        yy1 = np.maximum(y1[current], y1[remaining])
        xx2 = np.minimum(x2[current], x2[remaining])
        yy2 = np.minimum(y2[current], y2[remaining])

        widths = np.maximum(0.0, xx2 - xx1)
        heights = np.maximum(0.0, yy2 - yy1)
        intersection = widths * heights

        union = areas[current] + areas[remaining] - intersection
        iou = intersection / np.maximum(union, 1e-12)

        order = remaining[iou <= threshold]

    return keep


class SCRFDDetector:
    def __init__(
        self,
        model_path,
        input_size=(640, 640),
        score_threshold=0.50,
        nms_threshold=0.40,
        provider="cuda",
        allow_cpu_fallback=False,
        device_id=0,
    ):
        self.input_width = int(input_size[0])
        self.input_height = int(input_size[1])
        self.score_threshold = float(
            score_threshold
        )
        self.nms_threshold = float(
            nms_threshold
        )

        self.strides = (8, 16, 32)

        self.requested_provider = str(
            provider
        ).lower()

        self.allow_cpu_fallback = bool(
            allow_cpu_fallback
        )

        self.device_id = int(
            device_id
        )

        self.session = create_gpu_session(
            model_path,
            provider=self.requested_provider,
            allow_cpu_fallback=(
                self.allow_cpu_fallback
            ),
            device_id=self.device_id,
        )

        self.input_name = (
            self.session
            .get_inputs()[0]
            .name
        )

        outputs = self.session.get_outputs()

        if len(outputs) != 9:
            raise RuntimeError(
                "Expected 9 SCRFD outputs, "
                f"but model has {len(outputs)} outputs."
            )

    @property
    def providers(self):
        return self.session.get_providers()

    def preprocess(self, image: np.ndarray):
        if image is None or image.size == 0:
            raise ValueError("Input image is empty.")

        original_height, original_width = image.shape[:2]

        resize_scale = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )

        resized_width = max(1, int(round(original_width * resize_scale)))
        resized_height = max(1, int(round(original_height * resize_scale)))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        canvas = np.zeros(
            (self.input_height, self.input_width, 3),
            dtype=np.uint8,
        )

        canvas[:resized_height, :resized_width] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 128.0,
            size=(self.input_width, self.input_height),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
            crop=False,
        )

        return blob.astype(np.float32), float(resize_scale)

    def detect(self, image: np.ndarray) -> list:
        blob, resize_scale = self.preprocess(image)

        outputs = self.session.run(
            None,
            {self.input_name: blob},
        )

        score_outputs = outputs[0:3]
        bbox_outputs = outputs[3:6]
        landmark_outputs = outputs[6:9]

        all_boxes = []
        all_scores = []
        all_landmarks = []

        for stride, score_output, bbox_output, landmark_output in zip(
            self.strides,
            score_outputs,
            bbox_outputs,
            landmark_outputs,
        ):
            scores = np.asarray(score_output, dtype=np.float32).reshape(-1)
            bbox_distances = (
                np.asarray(bbox_output, dtype=np.float32).reshape(-1, 4) * stride
            )
            landmark_distances = (
                np.asarray(landmark_output, dtype=np.float32).reshape(-1, 10)
                * stride
            )

            feature_height = self.input_height // stride
            feature_width = self.input_width // stride
            locations_count = feature_height * feature_width

            if scores.size % locations_count != 0:
                raise RuntimeError(
                    f"Unexpected SCRFD output size at stride {stride}: "
                    f"{scores.size} scores."
                )

            number_of_anchors = scores.size // locations_count

            grid_x, grid_y = np.meshgrid(
                np.arange(feature_width),
                np.arange(feature_height),
            )

            anchor_centers = np.stack(
                [grid_x, grid_y],
                axis=-1,
            ).astype(np.float32)

            anchor_centers = (anchor_centers * stride).reshape(-1, 2)

            if number_of_anchors > 1:
                anchor_centers = np.repeat(
                    anchor_centers,
                    number_of_anchors,
                    axis=0,
                )

            selected = np.where(scores >= self.score_threshold)[0]

            if selected.size == 0:
                continue

            selected_centers = anchor_centers[selected]
            selected_scores = scores[selected]
            selected_bbox_distances = bbox_distances[selected]
            selected_landmark_distances = landmark_distances[selected]

            boxes = distance_to_bbox(
                selected_centers,
                selected_bbox_distances,
            )

            landmarks = distance_to_landmarks(
                selected_centers,
                selected_landmark_distances,
            ).reshape(-1, 5, 2)

            boxes /= resize_scale
            landmarks /= resize_scale

            all_boxes.append(boxes)
            all_scores.append(selected_scores)
            all_landmarks.append(landmarks)

        if not all_boxes:
            return []

        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        landmarks = np.concatenate(all_landmarks, axis=0)

        image_height, image_width = image.shape[:2]

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width - 1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height - 1)

        landmarks[:, :, 0] = np.clip(
            landmarks[:, :, 0],
            0,
            image_width - 1,
        )

        landmarks[:, :, 1] = np.clip(
            landmarks[:, :, 1],
            0,
            image_height - 1,
        )

        keep = nms(boxes, scores, self.nms_threshold)

        detections = []

        for index in keep:
            detections.append(
                FaceDetection(
                    bbox=boxes[index].astype(np.float32),
                    score=float(scores[index]),
                    landmarks=landmarks[index].astype(np.float32),
                )
            )

        detections.sort(key=lambda item: item.score, reverse=True)
        return detections