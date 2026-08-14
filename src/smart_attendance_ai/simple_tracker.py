from dataclasses import dataclass

import numpy as np


@dataclass
class FaceTrack:
    track_id: int
    bbox: np.ndarray
    landmarks: np.ndarray
    detection_score: float
    first_seen_frame: int
    last_seen_frame: int
    hits: int = 1

    @property
    def width(self):
        return float(
            self.bbox[2] - self.bbox[0]
        )

    @property
    def height(self):
        return float(
            self.bbox[3] - self.bbox[1]
        )

    @property
    def center(self):
        return np.array(
            [
                (
                    self.bbox[0]
                    + self.bbox[2]
                ) / 2.0,
                (
                    self.bbox[1]
                    + self.bbox[3]
                ) / 2.0,
            ],
            dtype=np.float32,
        )


def calculate_iou(first_bbox, second_bbox):
    first = np.asarray(
        first_bbox,
        dtype=np.float32,
    )

    second = np.asarray(
        second_bbox,
        dtype=np.float32,
    )

    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)

    intersection = (
        intersection_width
        * intersection_height
    )

    first_area = max(
        0.0,
        first[2] - first[0],
    ) * max(
        0.0,
        first[3] - first[1],
    )

    second_area = max(
        0.0,
        second[2] - second[0],
    ) * max(
        0.0,
        second[3] - second[1],
    )

    union = (
        first_area
        + second_area
        - intersection
    )

    if union <= 1e-12:
        return 0.0

    return float(intersection / union)


class SimpleFaceTracker:
    def __init__(
        self,
        minimum_iou=0.15,
        maximum_center_distance=1.20,
        maximum_age_frames=30,
        bbox_smoothing=0.25,
    ):
        self.minimum_iou = float(
            minimum_iou
        )

        self.maximum_center_distance = float(
            maximum_center_distance
        )

        self.maximum_age_frames = int(
            maximum_age_frames
        )

        self.bbox_smoothing = float(
            bbox_smoothing
        )

        self.tracks = {}
        self.next_track_id = 1
        self.total_created_tracks = 0

    def _remove_expired_tracks(
        self,
        frame_index,
    ):
        expired_ids = [
            track_id
            for track_id, track
            in self.tracks.items()
            if (
                frame_index
                - track.last_seen_frame
                > self.maximum_age_frames
            )
        ]

        for track_id in expired_ids:
            del self.tracks[track_id]

    def _pair_score(
        self,
        track,
        detection,
    ):
        iou = calculate_iou(
            track.bbox,
            detection.bbox,
        )

        detection_center = np.array(
            [
                (
                    detection.bbox[0]
                    + detection.bbox[2]
                ) / 2.0,
                (
                    detection.bbox[1]
                    + detection.bbox[3]
                ) / 2.0,
            ],
            dtype=np.float32,
        )

        center_distance = float(
            np.linalg.norm(
                track.center
                - detection_center
            )
        )

        normalization_size = max(
            track.width,
            track.height,
            detection.width,
            detection.height,
            1.0,
        )

        normalized_distance = (
            center_distance
            / normalization_size
        )

        eligible = (
            iou >= self.minimum_iou
            or normalized_distance
            <= self.maximum_center_distance
        )

        if not eligible:
            return None

        proximity = max(
            0.0,
            1.0
            - (
                normalized_distance
                / self.maximum_center_distance
            ),
        )

        score = iou + 0.25 * proximity

        return score

    def update(
        self,
        detections,
        frame_index,
    ):
        self._remove_expired_tracks(
            frame_index
        )

        track_ids = list(
            self.tracks.keys()
        )

        possible_pairs = []

        for track_id in track_ids:
            track = self.tracks[track_id]

            for detection_index, detection in (
                enumerate(detections)
            ):
                score = self._pair_score(
                    track,
                    detection,
                )

                if score is not None:
                    possible_pairs.append(
                        (
                            score,
                            track_id,
                            detection_index,
                        )
                    )

        possible_pairs.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        matched_track_ids = set()
        matched_detection_indices = set()

        for (
            score,
            track_id,
            detection_index,
        ) in possible_pairs:
            if track_id in matched_track_ids:
                continue

            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            track = self.tracks[track_id]
            detection = detections[
                detection_index
            ]

            old_weight = self.bbox_smoothing
            new_weight = 1.0 - old_weight

            track.bbox = (
                old_weight * track.bbox
                + new_weight * detection.bbox
            ).astype(np.float32)

            track.landmarks = (
                detection.landmarks.copy()
            )

            track.detection_score = float(
                detection.score
            )

            track.last_seen_frame = frame_index
            track.hits += 1

            matched_track_ids.add(track_id)
            matched_detection_indices.add(
                detection_index
            )

        for detection_index, detection in (
            enumerate(detections)
        ):
            if (
                detection_index
                in matched_detection_indices
            ):
                continue

            track_id = self.next_track_id
            self.next_track_id += 1
            self.total_created_tracks += 1

            self.tracks[track_id] = FaceTrack(
                track_id=track_id,
                bbox=detection.bbox.copy(),
                landmarks=(
                    detection.landmarks.copy()
                ),
                detection_score=float(
                    detection.score
                ),
                first_seen_frame=frame_index,
                last_seen_frame=frame_index,
            )

        return list(self.tracks.values())

    def get_active_tracks(
        self,
        frame_index,
    ):
        self._remove_expired_tracks(
            frame_index
        )

        return list(self.tracks.values())