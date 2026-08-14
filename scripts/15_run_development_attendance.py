import csv
import json
import sys
import time
import uuid
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
from smart_attendance_ai.face_alignment import (
    align_face,
    calculate_face_quality,
)
from smart_attendance_ai.face_matcher import (
    FaceGalleryMatcher,
)
from smart_attendance_ai.scrfd_detector import (
    SCRFDDetector,
)
from smart_attendance_ai.simple_tracker import (
    SimpleFaceTracker,
)
from smart_attendance_ai.temporal_identity import (
    TemporalIdentityResolver,
)


def write_csv_atomic(path, fieldnames, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(".csv.tmp")

    with temporary.open(
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

    temporary.replace(path)


def main():
    with (ROOT / "configs" / "ai.yaml").open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    detector_config = config["detector"]
    recognizer_config = config["recognizer"]
    matcher_config = config["matcher"]
    tracking_config = config["tracking"]
    quality_config = config["video_recognition"]
    attendance_config = config["attendance"]

    video_path = (
        ROOT
        / config["video"]["development_path"]
    )

    active_gallery_path = (
        ROOT
        / config["runtime"]["active_gallery"]
    )

    output_video_path = (
        ROOT
        / "outputs"
        / "videos"
        / "development_attendance.mp4"
    )

    attendance_csv_path = (
        ROOT
        / "outputs"
        / "reports"
        / "development_attendance.csv"
    )

    events_path = (
        ROOT
        / "outputs"
        / "events"
        / "development_attendance_events.jsonl"
    )

    summary_path = (
        ROOT
        / "outputs"
        / "reports"
        / "development_attendance_summary.json"
    )

    output_video_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    events_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = SCRFDDetector(
        model_path=(
            ROOT
            / config["models"]["detector"]["path"]
        ),
        input_size=(
            detector_config["input_width"],
            detector_config["input_height"],
        ),
        score_threshold=detector_config[
            "score_threshold"
        ],
        nms_threshold=detector_config[
            "nms_threshold"
        ],
    )

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

    matcher = FaceGalleryMatcher(
        gallery_path=active_gallery_path,
        top_k=matcher_config["top_k"],
        min_score=matcher_config["min_score"],
        min_margin=matcher_config["min_margin"],
        max_candidates=matcher_config[
            "max_candidates"
        ],
    )

    tracker = SimpleFaceTracker(
        minimum_iou=tracking_config[
            "minimum_iou"
        ],
        maximum_center_distance=tracking_config[
            "maximum_center_distance"
        ],
        maximum_age_frames=tracking_config[
            "maximum_age_frames"
        ],
        bbox_smoothing=tracking_config[
            "bbox_smoothing"
        ],
    )

    resolver = TemporalIdentityResolver(
        history_size=attendance_config[
            "identity_history_size"
        ],
        minimum_observations=attendance_config[
            "minimum_identity_observations"
        ],
        minimum_support_ratio=attendance_config[
            "minimum_support_ratio"
        ],
        minimum_average_score=attendance_config[
            "minimum_average_score"
        ],
        minimum_average_margin=attendance_config[
            "minimum_average_margin"
        ],
    )

    gallery = np.load(
        active_gallery_path,
        allow_pickle=False,
    )

    student_ids = gallery[
        "student_ids"
    ].astype(str).tolist()

    student_names = gallery[
        "student_names"
    ].astype(str).tolist()

    student_name_map = dict(
        zip(student_ids, student_names)
    )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Cannot create output video."
        )

    detection_interval = int(
        tracking_config[
            "detection_interval_frames"
        ]
    )

    recognition_interval = int(
        attendance_config[
            "recognition_interval_frames"
        ]
    )

    minimum_track_hits = int(
        attendance_config[
            "minimum_track_hits"
        ]
    )

    last_recognition_frame = {}
    attendance = {}
    events = []

    detector_times = []
    recognizer_times = []
    recognition_faces = 0

    frame_index = 0
    processing_start = time.perf_counter()

    print("=" * 70)
    print("SMART ATTENDANCE PIPELINE")
    print("=" * 70)
    print(f"Session           : {attendance_config['session_id']}")
    print(f"Video             : {video_path}")
    print(f"Active gallery    : {active_gallery_path}")
    print(f"Frames            : {total_frames}")
    print(f"Detection every   : {detection_interval} frames")
    print(f"Recognition every : {recognition_interval} frames")
    print(f"Detector          : {detector.providers}")
    print(f"Recognizer        : {recognizer.providers}")
    print()

    success, warmup_frame = capture.read()

    if not success:
        capture.release()
        writer.release()
        raise RuntimeError(
            "Cannot read first frame."
        )

    print("Warming up GPU models...")
    detector.detect(warmup_frame)
    recognizer.warm_up()
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        success, frame = capture.read()

        if not success:
            break

        timestamp_seconds = frame_index / fps

        if frame_index % detection_interval == 0:
            start = time.perf_counter()
            detections = detector.detect(frame)

            detector_times.append(
                (
                    time.perf_counter()
                    - start
                ) * 1000.0
            )

            tracks = tracker.update(
                detections,
                frame_index,
            )

            recognition_tracks = []
            aligned_faces = []

            for track in tracks:
                if (
                    track.last_seen_frame
                    != frame_index
                ):
                    continue

                if track.hits < minimum_track_hits:
                    continue

                last_frame = last_recognition_frame.get(
                    track.track_id,
                    -recognition_interval,
                )

                if (
                    frame_index - last_frame
                    < recognition_interval
                ):
                    continue

                minimum_side = min(
                    track.width,
                    track.height,
                )

                if (
                    track.detection_score
                    < quality_config[
                        "min_detection_score"
                    ]
                ):
                    continue

                if (
                    minimum_side
                    < quality_config[
                        "min_face_size_original"
                    ]
                ):
                    continue

                try:
                    aligned, _ = align_face(
                        frame,
                        track.landmarks,
                        output_size=112,
                    )

                    quality = (
                        calculate_face_quality(
                            aligned
                        )
                    )

                except Exception:
                    continue

                if (
                    quality["blur_score"]
                    < quality_config[
                        "min_blur_score"
                    ]
                ):
                    continue

                if not (
                    quality_config[
                        "min_brightness"
                    ]
                    <= quality["brightness"]
                    <= quality_config[
                        "max_brightness"
                    ]
                ):
                    continue

                recognition_tracks.append(track)
                aligned_faces.append(aligned)

            if aligned_faces:
                start = time.perf_counter()

                embeddings = recognizer.embed_faces(
                    aligned_faces
                )

                recognizer_times.append(
                    (
                        time.perf_counter()
                        - start
                    ) * 1000.0
                )

                recognition_faces += len(
                    aligned_faces
                )

                for track, embedding in zip(
                    recognition_tracks,
                    embeddings,
                ):
                    last_recognition_frame[
                        track.track_id
                    ] = frame_index

                    result = matcher.match(
                        embedding
                    )

                    state, confirmed_now = (
                        resolver.add_result(
                            track_id=track.track_id,
                            frame_index=frame_index,
                            timestamp_seconds=(
                                timestamp_seconds
                            ),
                            match_result=result,
                        )
                    )

                    if confirmed_now:
                        student_id = (
                            state.confirmed_student_id
                        )

                        if student_id not in attendance:
                            attendance[student_id] = {
                                "student_id": student_id,
                                "full_name": (
                                    student_name_map.get(
                                        student_id,
                                        "",
                                    )
                                ),
                                "first_seen_seconds": min(
                                    item.timestamp_seconds
                                    for item
                                    in state.history
                                    if (
                                        item.student_id
                                        == student_id
                                    )
                                ),
                                "confirmed_at_seconds": (
                                    timestamp_seconds
                                ),
                                "last_seen_seconds": (
                                    timestamp_seconds
                                ),
                                "best_score": (
                                    state.average_score
                                ),
                                "observation_count": (
                                    state.winner_observations
                                ),
                                "track_ids": {
                                    track.track_id
                                },
                            }

                            event = {
                                "event_id": str(
                                    uuid.uuid4()
                                ),
                                "event_type": (
                                    "attendance_confirmed"
                                ),
                                "session_id": (
                                    attendance_config[
                                        "session_id"
                                    ]
                                ),
                                "student_id": (
                                    student_id
                                ),
                                "full_name": (
                                    student_name_map.get(
                                        student_id,
                                        "",
                                    )
                                ),
                                "track_id": (
                                    track.track_id
                                ),
                                "timestamp_seconds": round(
                                    timestamp_seconds,
                                    3,
                                ),
                                "average_score": round(
                                    state.average_score,
                                    6,
                                ),
                                "average_margin": round(
                                    state.average_margin,
                                    6,
                                ),
                            }

                            events.append(event)

                    if state.confirmed_student_id:
                        student_id = (
                            state.confirmed_student_id
                        )

                        if student_id in attendance:
                            record = attendance[
                                student_id
                            ]

                            record[
                                "last_seen_seconds"
                            ] = timestamp_seconds

                            record["best_score"] = max(
                                record["best_score"],
                                state.average_score,
                            )

                            record["observation_count"] = max(
                                record[
                                    "observation_count"
                                ],
                                state.winner_observations,
                            )

                            record["track_ids"].add(
                                track.track_id
                            )

        else:
            tracks = tracker.get_active_tracks(
                frame_index
            )

        for track in tracks:
            stale_frames = (
                frame_index
                - track.last_seen_frame
            )

            if stale_frames > (
                detection_interval * 2
            ):
                continue

            state = resolver.get_state(
                track.track_id
            )

            x1, y1, x2, y2 = (
                track.bbox.astype(int)
            )

            if state.confirmed_student_id:
                color = (0, 220, 0)

                label = (
                    f"{state.confirmed_student_id} "
                    f"T{track.track_id} "
                    f"{state.average_score:.2f}"
                )

            elif state.latest_status == "AMBIGUOUS":
                color = (0, 200, 255)

                label = (
                    f"AMB T{track.track_id}"
                )

            elif state.latest_status == "UNKNOWN":
                color = (0, 0, 220)

                label = (
                    f"UNKNOWN T{track.track_id}"
                )

            else:
                color = (180, 180, 180)

                label = (
                    f"PENDING T{track.track_id} "
                    f"{state.winner_observations}/"
                    f"{attendance_config['minimum_identity_observations']}"
                )

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 44),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            (
                f"SMART ATTENDANCE | "
                f"time={timestamp_seconds:.1f}s "
                f"present={len(attendance)} "
                f"active_tracks={len(tracks)}"
            ),
            (12, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)
        frame_index += 1

        if (
            frame_index % 300 == 0
            or frame_index == total_frames
        ):
            print(
                f"Processed "
                f"{frame_index}/{total_frames} | "
                f"present={len(attendance)}"
            )

    capture.release()
    writer.release()

    processing_seconds = (
        time.perf_counter()
        - processing_start
    )

    attendance_rows = []

    for student_id in student_ids:
        record = attendance.get(student_id)

        if record is None:
            attendance_rows.append(
                {
                    "session_id": (
                        attendance_config[
                            "session_id"
                        ]
                    ),
                    "student_id": student_id,
                    "full_name": (
                        student_name_map.get(
                            student_id,
                            "",
                        )
                    ),
                    "status": "ABSENT",
                    "first_seen_seconds": "",
                    "confirmed_at_seconds": "",
                    "last_seen_seconds": "",
                    "best_score": "",
                    "observation_count": 0,
                    "track_ids": "",
                }
            )
        else:
            attendance_rows.append(
                {
                    "session_id": (
                        attendance_config[
                            "session_id"
                        ]
                    ),
                    "student_id": student_id,
                    "full_name": record[
                        "full_name"
                    ],
                    "status": "PRESENT",
                    "first_seen_seconds": round(
                        record[
                            "first_seen_seconds"
                        ],
                        3,
                    ),
                    "confirmed_at_seconds": round(
                        record[
                            "confirmed_at_seconds"
                        ],
                        3,
                    ),
                    "last_seen_seconds": round(
                        record[
                            "last_seen_seconds"
                        ],
                        3,
                    ),
                    "best_score": round(
                        record["best_score"],
                        6,
                    ),
                    "observation_count": record[
                        "observation_count"
                    ],
                    "track_ids": "|".join(
                        str(track_id)
                        for track_id in sorted(
                            record["track_ids"]
                        )
                    ),
                }
            )

    write_csv_atomic(
        attendance_csv_path,
        fieldnames=[
            "session_id",
            "student_id",
            "full_name",
            "status",
            "first_seen_seconds",
            "confirmed_at_seconds",
            "last_seen_seconds",
            "best_score",
            "observation_count",
            "track_ids",
        ],
        rows=attendance_rows,
    )

    temporary_events_path = (
        events_path.with_suffix(".jsonl.tmp")
    )

    with temporary_events_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for event in events:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temporary_events_path.replace(events_path)

    present_ids = sorted(attendance.keys())
    absent_ids = sorted(
        set(student_ids) - set(present_ids)
    )

    expected_present = set(
        config["validation"][
            "expected_present"
        ]
    )

    expected_absent = set(
        config["validation"][
            "expected_absent"
        ]
    )

    summary = {
        "session_id": attendance_config[
            "session_id"
        ],
        "source_video": str(
            video_path.relative_to(ROOT)
        ),
        "active_gallery": str(
            active_gallery_path.relative_to(
                ROOT
            )
        ),
        "frames_processed": frame_index,
        "processing_seconds": round(
            processing_seconds,
            3,
        ),
        "processing_fps": round(
            frame_index / processing_seconds,
            3,
        ),
        "tracks_created": (
            tracker.total_created_tracks
        ),
        "recognition_faces": recognition_faces,
        "present_student_ids": present_ids,
        "absent_student_ids": absent_ids,
        "missed_expected_present": sorted(
            expected_present - set(present_ids)
        ),
        "false_present": sorted(
            expected_absent & set(present_ids)
        ),
        "attendance_csv": str(
            attendance_csv_path.relative_to(
                ROOT
            )
        ),
        "events_jsonl": str(
            events_path.relative_to(ROOT)
        ),
        "annotated_video": str(
            output_video_path.relative_to(
                ROOT
            )
        ),
    }

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 70)
    print("ATTENDANCE SUMMARY")
    print("=" * 70)
    print(f"Present              : {present_ids}")
    print(f"Absent               : {absent_ids}")
    print(
        f"Missed expected      : "
        f"{summary['missed_expected_present']}"
    )
    print(
        f"False present        : "
        f"{summary['false_present']}"
    )
    print(
        f"Processing FPS       : "
        f"{summary['processing_fps']}"
    )
    print(f"Attendance CSV       : {attendance_csv_path}")
    print(f"Events JSONL         : {events_path}")
    print(f"Annotated video      : {output_video_path}")
    print(f"Summary JSON         : {summary_path}")


if __name__ == "__main__":
    main()  