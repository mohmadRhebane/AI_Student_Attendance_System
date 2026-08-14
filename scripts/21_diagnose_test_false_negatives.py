"""Diagnose false-negative attendance results without changing AI thresholds."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.attendance_engine import AttendanceEngine
from smart_attendance_ai.attendance_schemas import OutputOptions, SessionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze students that were marked absent even though they are "
            "known to be present. This script does not change ai.yaml or "
            "matching decisions."
        )
    )
    parser.add_argument(
        "--source",
        default="data/raw/videos/test.mp4",
    )
    parser.add_argument(
        "--session-id",
        default="TEST_FALSE_NEGATIVE_DIAGNOSTIC_001",
    )
    parser.add_argument(
        "--class-id",
        default="CLASS_A",
    )
    parser.add_argument(
        "--course-id",
        default="COURSE_101",
    )
    parser.add_argument(
        "--gallery",
        default="data/generated/face_gallery_calibrated.npz",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["STU_006", "STU_010"],
    )
    parser.add_argument(
        "--roster",
        nargs="+",
        default=[f"STU_{number:03d}" for number in range(1, 16)],
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/diagnostics",
    )
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target_student_id",
        "frame_index",
        "timestamp_seconds",
        "track_id",
        "target_rank",
        "target_score",
        "top_student_id",
        "top_score",
        "second_score",
        "margin",
        "match_status",
        "target_is_top1",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_target(
    target_id: str,
    rows: list,
    confirmed_students: set,
    min_score: float,
    min_margin: float,
    minimum_observations: int,
) -> dict:
    target_rows = [
        row for row in rows
        if row["target_student_id"] == target_id
    ]
    top1_rows = [row for row in target_rows if row["target_is_top1"]]
    top1_match_rows = [
        row for row in top1_rows if row["match_status"] == "MATCH"
    ]
    top1_ambiguous_rows = [
        row for row in top1_rows if row["match_status"] == "AMBIGUOUS"
    ]
    top1_unknown_rows = [
        row for row in top1_rows if row["match_status"] == "UNKNOWN"
    ]

    match_counts_by_track = defaultdict(int)
    all_top1_counts_by_track = defaultdict(int)
    for row in top1_rows:
        all_top1_counts_by_track[str(row["track_id"])] += 1
    for row in top1_match_rows:
        match_counts_by_track[str(row["track_id"])] += 1

    max_candidate_score = max(
        (float(row["target_score"]) for row in target_rows),
        default=0.0,
    )
    max_top1_score = max(
        (float(row["top_score"]) for row in top1_rows),
        default=0.0,
    )
    max_top1_margin = max(
        (float(row["margin"]) for row in top1_rows),
        default=0.0,
    )
    best_rank = min(
        (int(row["target_rank"]) for row in target_rows),
        default=0,
    )
    max_match_observations_same_track = max(
        match_counts_by_track.values(),
        default=0,
    )

    if target_id in confirmed_students:
        likely_reason = "confirmed"
    elif not top1_rows:
        likely_reason = (
            "The target never became the best candidate for any recognized face."
        )
    elif not top1_match_rows:
        if max_top1_score < min_score:
            likely_reason = (
                "The target became top-1, but its score stayed below min_score."
            )
        elif max_top1_margin < min_margin:
            likely_reason = (
                "The target became top-1, but the margin was too small."
            )
        else:
            likely_reason = (
                "The target was top-1 but never produced an accepted MATCH."
            )
    elif max_match_observations_same_track < minimum_observations:
        likely_reason = (
            "Accepted MATCH observations existed, but they were insufficient "
            "or fragmented across tracks for temporal confirmation."
        )
    else:
        likely_reason = (
            "Enough MATCH observations may exist, but temporal support/history "
            "conditions prevented confirmation. Inspect per-track rows."
        )

    return {
        "student_id": target_id,
        "confirmed": target_id in confirmed_students,
        "candidate_observations": len(target_rows),
        "best_candidate_rank": best_rank,
        "top1_observations": len(top1_rows),
        "top1_match_observations": len(top1_match_rows),
        "top1_ambiguous_observations": len(top1_ambiguous_rows),
        "top1_unknown_observations": len(top1_unknown_rows),
        "distinct_top1_tracks": len(all_top1_counts_by_track),
        "distinct_match_tracks": len(match_counts_by_track),
        "max_match_observations_same_track": max_match_observations_same_track,
        "max_candidate_score": round(max_candidate_score, 6),
        "max_top1_score": round(max_top1_score, 6),
        "max_top1_margin": round(max_top1_margin, 6),
        "match_counts_by_track": dict(
            sorted(
                match_counts_by_track.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "likely_reason": likely_reason,
    }


def main() -> None:
    args = parse_args()
    source_path = resolve_path(args.source)
    gallery_path = resolve_path(args.gallery)
    output_dir = resolve_path(args.output_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Video does not exist: {source_path}")
    if not gallery_path.exists():
        raise FileNotFoundError(f"Gallery does not exist: {gallery_path}")

    roster = list(dict.fromkeys(str(item).strip() for item in args.roster))
    targets = list(dict.fromkeys(str(item).strip() for item in args.targets))
    unknown_targets = sorted(set(targets) - set(roster))
    if unknown_targets:
        raise ValueError(
            "Targets must belong to the roster: " + ", ".join(unknown_targets)
        )

    session = SessionConfig(
        session_id=args.session_id,
        class_id=args.class_id,
        course_id=args.course_id,
        source=str(source_path),
        gallery_path=gallery_path,
        roster_student_ids=roster,
        output=OutputOptions(
            save_csv=False,
            save_events=False,
            save_summary=False,
            save_video=False,
        ),
    )

    engine = AttendanceEngine(root=ROOT)
    engine.start_session(session)

    # Returning more candidates changes diagnostics only, not the matching
    # decision, score, margin, or temporal confirmation logic.
    engine.matcher.max_candidates = len(roster)

    observation_rows = []
    original_add_result = engine.resolver.add_result

    def diagnostic_add_result(
        track_id,
        frame_index,
        timestamp_seconds,
        match_result,
    ):
        candidate_rank = {
            candidate.student_id: index + 1
            for index, candidate in enumerate(match_result.candidates)
        }
        candidate_score = {
            candidate.student_id: float(candidate.score)
            for candidate in match_result.candidates
        }

        for target_id in targets:
            rank = candidate_rank.get(target_id)
            if rank is None:
                continue
            observation_rows.append(
                {
                    "target_student_id": target_id,
                    "frame_index": int(frame_index),
                    "timestamp_seconds": round(float(timestamp_seconds), 3),
                    "track_id": int(track_id),
                    "target_rank": int(rank),
                    "target_score": round(candidate_score[target_id], 6),
                    "top_student_id": match_result.student_id,
                    "top_score": round(float(match_result.score), 6),
                    "second_score": round(float(match_result.second_score), 6),
                    "margin": round(float(match_result.margin), 6),
                    "match_status": match_result.status,
                    "target_is_top1": match_result.student_id == target_id,
                }
            )

        return original_add_result(
            track_id=track_id,
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            match_result=match_result,
        )

    engine.resolver.add_result = diagnostic_add_result

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    success, warmup_frame = capture.read()
    if not success:
        capture.release()
        raise RuntimeError("Cannot read first video frame.")

    engine.warm_up(warmup_frame)
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("=" * 78)
    print("FALSE NEGATIVE DIAGNOSTIC")
    print("=" * 78)
    print(f"Source   : {source_path}")
    print(f"Targets  : {targets}")
    print(f"Roster   : {len(roster)} students")
    print("Decisions: unchanged")
    print()

    frame_index = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            result = engine.process_frame(
                frame=frame,
                frame_index=frame_index,
                timestamp_seconds=frame_index / fps,
            )

            if result.newly_confirmed_student_ids:
                print(
                    f"Frame {frame_index:5d} confirmed: "
                    + ", ".join(result.newly_confirmed_student_ids)
                )
            elif frame_index % args.progress_every == 0:
                print(
                    f"Frame {frame_index:5d} | present={len(engine.attendance):2d} "
                    f"| recognized={result.recognized_faces:2d}"
                )

            frame_index += 1
    finally:
        capture.release()

    confirmed_students = set(engine.attendance.keys())
    matcher_config = engine.config["matcher"]
    attendance_config = engine.config["attendance"]

    target_summaries = [
        summarize_target(
            target_id=target_id,
            rows=observation_rows,
            confirmed_students=confirmed_students,
            min_score=float(matcher_config["min_score"]),
            min_margin=float(matcher_config["min_margin"]),
            minimum_observations=int(
                attendance_config["minimum_identity_observations"]
            ),
        )
        for target_id in targets
    ]

    csv_path = output_dir / f"{args.session_id}_target_observations.csv"
    json_path = output_dir / f"{args.session_id}_diagnostic_summary.json"
    write_csv(csv_path, observation_rows)

    payload = {
        "session_id": args.session_id,
        "source": str(source_path),
        "frames_processed": frame_index,
        "ground_truth": {
            "all_roster_students_present": True,
            "expected_present_student_ids": roster,
            "expected_absent_student_ids": [],
        },
        "actual": {
            "confirmed_student_ids": sorted(confirmed_students),
            "missing_student_ids": sorted(set(roster) - confirmed_students),
        },
        "thresholds_unchanged": {
            "min_score": float(matcher_config["min_score"]),
            "min_margin": float(matcher_config["min_margin"]),
            "minimum_identity_observations": int(
                attendance_config["minimum_identity_observations"]
            ),
            "minimum_support_ratio": float(
                attendance_config["minimum_support_ratio"]
            ),
            "minimum_average_score": float(
                attendance_config["minimum_average_score"]
            ),
            "minimum_average_margin": float(
                attendance_config["minimum_average_margin"]
            ),
        },
        "targets": target_summaries,
        "outputs": {
            "observations_csv": str(csv_path),
            "summary_json": str(json_path),
        },
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 78)
    print(f"Frames processed : {frame_index}")
    print(f"Confirmed        : {sorted(confirmed_students)}")
    print(f"Missing          : {sorted(set(roster) - confirmed_students)}")
    print()

    for item in target_summaries:
        print(f"{item['student_id']}")
        print(f"  confirmed                    : {item['confirmed']}")
        print(f"  best candidate rank           : {item['best_candidate_rank']}")
        print(f"  top-1 observations            : {item['top1_observations']}")
        print(f"  accepted MATCH observations   : {item['top1_match_observations']}")
        print(f"  AMBIGUOUS top-1 observations  : {item['top1_ambiguous_observations']}")
        print(f"  UNKNOWN top-1 observations    : {item['top1_unknown_observations']}")
        print(f"  distinct MATCH tracks         : {item['distinct_match_tracks']}")
        print(
            "  max MATCH observations/track : "
            f"{item['max_match_observations_same_track']}"
        )
        print(f"  max candidate score           : {item['max_candidate_score']}")
        print(f"  max top-1 score               : {item['max_top1_score']}")
        print(f"  max top-1 margin              : {item['max_top1_margin']}")
        print(f"  likely reason                 : {item['likely_reason']}")
        print()

    print(f"Observations CSV: {csv_path}")
    print(f"Summary JSON    : {json_path}")
    print()
    print("FALSE NEGATIVE DIAGNOSTIC COMPLETED")


if __name__ == "__main__":
    main()