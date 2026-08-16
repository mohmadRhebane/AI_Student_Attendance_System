"""Run a dynamic attendance session from a file, camera, or stream."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Union

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.attendance_schemas import OutputOptions, SessionConfig


VideoSource = Union[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a dynamic attendance session without changing ai.yaml. "
            "The source, roster, session, and outputs are supplied at runtime."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Video path, camera index such as 0, or RTSP/HTTP stream URL.",
    )
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--course-id")
    parser.add_argument("--camera-id")
    parser.add_argument(
        "--gallery",
        default="data/generated/face_gallery_calibrated.npz",
        help="Gallery path, absolute or relative to the project root.",
    )
    parser.add_argument(
        "--roster",
        nargs="+",
        help=(
            "Student IDs separated by spaces or commas, for example "
            "--roster STU_001 STU_002 or --roster STU_001,STU_002."
        ),
    )
    parser.add_argument(
        "--roster-file",
        help="Optional JSON list or text file containing student IDs.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dynamic",
        help="Output directory, absolute or relative to the project root.",
    )
    parser.add_argument(
        "--output-prefix",
        help="Output filename prefix. Defaults to session-id.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Stop after this many frames. Zero means no frame limit.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop after this many source seconds. Zero means no time limit.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=150,
        help="Print progress every N frames.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write an annotated output video. Disabled by default.",
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument("--no-summary", action="store_true")

    args = parser.parse_args()

    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative.")
    if args.max_seconds < 0:
        parser.error("--max-seconds cannot be negative.")
    if args.progress_every <= 0:
        parser.error("--progress-every must be greater than zero.")
    if not args.roster and not args.roster_file:
        parser.error("Provide --roster or --roster-file.")

    return args


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def parse_source(source_value: str) -> tuple[VideoSource, str, bool]:
    stripped = source_value.strip()
    if not stripped:
        raise ValueError("Source cannot be empty.")

    if stripped.isdigit():
        camera_index = int(stripped)
        return camera_index, str(camera_index), True

    lowered = stripped.lower()
    if lowered.startswith(("rtsp://", "rtsps://", "http://", "https://")):
        return stripped, stripped, True

    source_path = resolve_path(stripped)
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {source_path}")

    return str(source_path), str(source_path), False


def normalize_student_ids(values: Iterable[str]) -> List[str]:
    normalized = []
    seen = set()

    for value in values:
        for item in str(value).replace(";", ",").split(","):
            student_id = item.strip()
            if not student_id or student_id in seen:
                continue
            seen.add(student_id)
            normalized.append(student_id)

    return normalized


def load_roster(
    roster_values: Optional[List[str]],
    roster_file_value: Optional[str],
) -> List[str]:
    values: List[str] = list(roster_values or [])

    if roster_file_value:
        roster_path = resolve_path(roster_file_value)
        if not roster_path.exists():
            raise FileNotFoundError(
                f"Roster file does not exist: {roster_path}"
            )

        text = roster_path.read_text(encoding="utf-8-sig")
        if roster_path.suffix.lower() == ".json":
            payload = json.loads(text)
            if not isinstance(payload, list):
                raise ValueError("Roster JSON must contain a list.")
            values.extend(str(item) for item in payload)
        else:
            values.extend(
                line.strip()
                for line in text.splitlines()
                if line.strip()
            )

    roster = normalize_student_ids(values)
    if not roster:
        raise ValueError("The resolved roster is empty.")

    return roster


def write_csv_atomic(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_events_jsonl(path: Path, events) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", encoding="utf-8") as file:
        for event in events:
            payload = {
                "event_id": event.event_id,
                "event_type": event.event_type,
                "session_id": event.session_id,
                "student_id": event.student_id,
                "full_name": event.full_name,
                "track_id": event.track_id,
                "frame_index": event.frame_index,
                "timestamp_seconds": event.timestamp_seconds,
                "score": event.score,
                "margin": event.margin,
                "created_at": event.created_at.isoformat(),
            }
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    temporary.replace(path)

#
#
#def draw_frame(
#     frame,
#     frame_result,
# ):
#     annotated = frame.copy()

#     for track in frame_result.tracks:

#         x1, y1, x2, y2 = [
#             int(round(value))
#             for value in track.bbox
#         ]

#         status = getattr(
#             track.status,
#             "value",
#             str(track.status),
#         )

#         # ======================================
#         # Label
#         # ======================================

#         if track.student_id:

#             name = (
#                 track.full_name
#                 or track.student_id
#             )

#             label = (
#                 f"{track.student_id}"
#                 f" | {name}"
#                 f" | {track.score:.2f}"
#             )

#         else:

#             label = (
#                 f"{status}"
#                 f" | {track.score:.2f}"
#             )

#         # ======================================
#         # BBox
#         # ======================================

#         cv2.rectangle(
#             annotated,
#             (x1, y1),
#             (x2, y2),
#             (255, 255, 255),
#             2,
#         )

#         # ======================================
#         # Background for label
#         # ======================================

#         font = (
#             cv2.FONT_HERSHEY_SIMPLEX
#         )

#         font_scale = 0.55
#         thickness = 2

#         (
#             text_width,
#             text_height,
#         ), baseline = (
#             cv2.getTextSize(
#                 label,
#                 font,
#                 font_scale,
#                 thickness,
#             )
#         )

#         label_y = max(
#             y1 - 10,
#             text_height + 10,
#         )

#         cv2.rectangle(
#             annotated,

#             (
#                 x1,
#                 label_y
#                 - text_height
#                 - 8,
#             ),

#             (
#                 x1
#                 + text_width
#                 + 10,

#                 label_y
#                 + baseline,
#             ),

#             (0, 0, 0),

#             -1,
#         )

#         cv2.putText(
#             annotated,
#             label,

#             (
#                 x1 + 5,
#                 label_y - 3,
#             ),

#             font,
#             font_scale,

#             (255, 255, 255),

#             thickness,

#             cv2.LINE_AA,
#         )

#     return annotated
#
#
def draw_frame(frame, frame_result):
    annotated = frame.copy()

    for track in frame_result.tracks:
        x1, y1, x2, y2 = [int(round(value)) for value in track.bbox]
        label = f"T{track.track_id} {track.status.value}"
        if track.student_id:
            label += f" {track.student_id}"
        if track.score:
            label += f" {track.score:.3f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated


def attendance_rows(engine: AttendanceEngine, roster: List[str]) -> List[dict]:
    rows = []

    for student_id in roster:
        record = engine.attendance.get(student_id)
        rows.append(
            {
                "session_id": engine.session_config.session_id,
                "class_id": engine.session_config.class_id,
                "course_id": engine.session_config.course_id or "",
                "student_id": student_id,
                "full_name": engine.student_name_map.get(student_id, ""),
                "status": "PRESENT" if record else "ABSENT",
                "first_seen_seconds": (
                    round(record.first_seen_seconds, 3)
                    if record and record.first_seen_seconds is not None
                    else ""
                ),
                "confirmed_at_seconds": (
                    round(record.confirmed_at_seconds, 3)
                    if record and record.confirmed_at_seconds is not None
                    else ""
                ),
                "last_detected_seconds": (
                    round(record.last_detected_seconds, 3)
                    if record and record.last_detected_seconds is not None
                    else ""
                ),
                "last_recognized_seconds": (
                    round(record.last_recognized_seconds, 3)
                    if record and record.last_recognized_seconds is not None
                    else ""
                ),
                "best_score": (
                    round(record.best_score, 6)
                    if record and record.best_score is not None
                    else ""
                ),
                "best_margin": (
                    round(record.best_margin, 6)
                    if record and record.best_margin is not None
                    else ""
                ),
                "observation_count": record.observation_count if record else 0,
                "track_ids": (
                    ",".join(str(value) for value in sorted(record.track_ids))
                    if record
                    else ""
                ),
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    roster = load_roster(args.roster, args.roster_file)
    capture_source, source_display, is_live = parse_source(args.source)
    gallery_path = resolve_path(args.gallery)
    output_directory = resolve_path(args.output_dir)
    output_prefix = args.output_prefix or args.session_id

    output_options = OutputOptions(
        save_csv=not args.no_csv,
        save_events=not args.no_events,
        save_summary=not args.no_summary,
        save_video=args.save_video,
        output_directory=output_directory,
        output_prefix=output_prefix,
    )
    session = SessionConfig(
        session_id=args.session_id,
        class_id=args.class_id,
        course_id=args.course_id,
        camera_id=args.camera_id,
        source=(capture_source if isinstance(capture_source, int) else args.source),
        gallery_path=gallery_path,
        roster_student_ids=roster,
        output=output_options,
    )

    engine = AttendanceEngine(root=ROOT)
    engine.start_session(session)

    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open source: {source_display}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    timestamp_fps = fps if fps > 0 else 30.0
    writer = None
    interrupted = False
    processing_started = time.perf_counter()

    reports_directory = output_directory / "reports"
    videos_directory = output_directory / "videos"
    csv_path = reports_directory / f"{output_prefix}_attendance.csv"
    events_path = reports_directory / f"{output_prefix}_events.jsonl"
    summary_path = reports_directory / f"{output_prefix}_summary.json"
    video_path = videos_directory / f"{output_prefix}_attendance.mp4"

    print("=" * 78)
    print("DYNAMIC ATTENDANCE RUNNER")
    print("=" * 78)
    print(f"Session ID      : {session.session_id}")
    print(f"Class ID        : {session.class_id}")
    print(f"Course ID       : {session.course_id or '-'}")
    print(f"Source          : {source_display}")
    print(f"Source type     : {'live' if is_live else 'video file'}")
    print(f"Source FPS      : {fps:.3f}")
    print(f"Roster count    : {len(roster)}")
    print(f"Gallery         : {gallery_path}")
    print(f"Output directory: {output_directory}")
    print(f"Save video      : {args.save_video}")
    print("Warming up GPU models...")

    success, first_frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError("Cannot read the first frame from the source.")

    engine.warm_up(first_frame)

    pending_frame = first_frame
    if not is_live:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pending_frame = None

    frame_index = 0

    try:
        while True:
            if pending_frame is not None:
                frame = pending_frame
                pending_frame = None
                success = True
            else:
                success, frame = capture.read()

            if not success:
                break

            if is_live:
                timestamp_seconds = time.perf_counter() - processing_started
            else:
                timestamp_seconds = frame_index / timestamp_fps

            if args.max_frames and frame_index >= args.max_frames:
                break
            if args.max_seconds and timestamp_seconds >= args.max_seconds:
                break

            result = engine.process_frame(
                frame=frame,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
            )

            if args.save_video:
                if writer is None:
                    videos_directory.mkdir(parents=True, exist_ok=True)
                    height, width = frame.shape[:2]
                    writer_fps = fps if fps > 0 else 30.0
                    writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        writer_fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Cannot create output video: {video_path}"
                        )

                writer.write(draw_frame(frame, result))

            if result.newly_confirmed_student_ids:
                print(
                    f"Frame {frame_index:6d} | confirmed: "
                    + ", ".join(result.newly_confirmed_student_ids)
                )

            if frame_index % args.progress_every == 0:
                print(
                    f"Frame {frame_index:6d} | "
                    f"detected={result.detected_faces:2d} | "
                    f"tracks={result.active_tracks:2d} | "
                    f"recognized={result.recognized_faces:2d} | "
                    f"present={len(engine.attendance):2d}"
                )

            frame_index += 1

    except KeyboardInterrupt:
        interrupted = True
        print("\nSession interrupted by the user.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    processing_seconds = time.perf_counter() - processing_started
    runtime_info = engine.get_runtime_info()
    processing_fps = (
        runtime_info["frames_processed"] / processing_seconds
        if processing_seconds > 0
        else 0.0
    )

    rows = attendance_rows(engine, roster)
    present_ids = sorted(engine.attendance.keys())
    absent_ids = sorted(set(roster) - set(present_ids))

    if output_options.save_csv:
        write_csv_atomic(csv_path, list(rows[0].keys()), rows)
    if output_options.save_events:
        write_events_jsonl(events_path, engine.events)

    summary = {
        "session_id": session.session_id,
        "class_id": session.class_id,
        "course_id": session.course_id,
        "camera_id": session.camera_id,
        "source": source_display,
        "source_type": "live" if is_live else "video_file",
        "gallery_path": str(gallery_path),
        "status": "CANCELLED" if interrupted else "COMPLETED",
        "roster_count": len(roster),
        "present_count": len(present_ids),
        "absent_count": len(absent_ids),
        "present_student_ids": present_ids,
        "absent_student_ids": absent_ids,
        "frames_processed": runtime_info["frames_processed"],
        "detection_calls": runtime_info["detection_calls"],
        "recognition_calls": runtime_info["recognition_calls"],
        "recognition_faces": runtime_info["recognition_faces"],
        "alignment_failures": runtime_info["alignment_failures"],
        "source_fps": round(fps, 3),
        "processing_seconds": round(processing_seconds, 3),
        "processing_fps": round(processing_fps, 3),
        "events_count": len(engine.events),
        "outputs": {
            "attendance_csv": str(csv_path) if output_options.save_csv else None,
            "events_jsonl": (
                str(events_path) if output_options.save_events else None
            ),
            "summary_json": (
                str(summary_path) if output_options.save_summary else None
            ),
            "annotated_video": (
                str(video_path) if output_options.save_video else None
            ),
        },
    }

    if output_options.save_summary:
        write_json_atomic(summary_path, summary)

    print()
    print("=" * 78)
    print("DYNAMIC ATTENDANCE SUMMARY")
    print("=" * 78)
    print(f"Status             : {summary['status']}")
    print(f"Frames processed   : {summary['frames_processed']}")
    print(f"Processing FPS     : {summary['processing_fps']}")
    print(f"Recognition faces  : {summary['recognition_faces']}")
    print(f"Alignment failures : {summary['alignment_failures']}")
    print(f"Present count      : {summary['present_count']}")
    print(f"Absent count       : {summary['absent_count']}")
    print(f"Present students   : {present_ids}")
    print(f"Absent students    : {absent_ids}")
    if output_options.save_csv:
        print(f"Attendance CSV     : {csv_path}")
    if output_options.save_events:
        print(f"Events JSONL       : {events_path}")
    if output_options.save_summary:
        print(f"Summary JSON       : {summary_path}")
    if output_options.save_video:
        print(f"Annotated video    : {video_path}")


if __name__ == "__main__":
    main()