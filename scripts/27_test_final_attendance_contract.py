"""Dynamic test for the final backend-facing attendance contract."""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.ai_contracts import (
    AttendanceSessionAIRequest,
    AttendanceSessionAIResult,
    AttendanceSessionSnapshotAIResult,
)
from smart_attendance_ai.ai_facade import SmartAttendanceAI


def parse_json_object(value: Optional[str]) -> Dict[str, object]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("JSON value must be an object.")
    return parsed


def parse_ids(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None

    normalized = value.strip()

    if normalized.upper() in {
        "NONE",
        "EMPTY",
        "NO_STUDENTS",
        "-",
    }:
        return []

    return [
        item.strip()
        for item in normalized.split(",")
        if item.strip()
    ]


def source_value(value: str):
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the final typed attendance AI contract."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--class-id", required=True)
    parser.add_argument("--roster", nargs="+", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--course-id")
    parser.add_argument("--camera-id")
    parser.add_argument("--gallery")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--metadata-json")
    parser.add_argument("--extensions-json")
    parser.add_argument("--expected-present")
    parser.add_argument("--expected-absent")
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    session_id = args.session_id or (
        "FINAL_AI_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    )

    metadata = parse_json_object(args.metadata_json)
    extensions = parse_json_object(args.extensions_json)
    expected_present = parse_ids(args.expected_present)
    expected_absent = parse_ids(args.expected_absent)

    request = AttendanceSessionAIRequest(
        session_id=session_id,
        class_id=args.class_id,
        course_id=args.course_id,
        camera_id=args.camera_id,
        source=source_value(args.source),
        roster_student_ids=args.roster,
        gallery_path=Path(args.gallery) if args.gallery else None,
        max_frames=args.max_frames,
        max_seconds=args.max_seconds,
        metadata=metadata,
        extensions=extensions,
    )

    output_path = (
        Path(args.output_json)
        if args.output_json
        else ROOT / "outputs" / "contracts" / f"{session_id}_result.json"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    print("=" * 70)
    print("FINAL ATTENDANCE AI CONTRACT TEST")
    print("=" * 70)
    print(f"Session ID          : {request.session_id}")
    print(f"Source              : {request.source}")
    print(f"Roster count        : {len(request.roster_student_ids)}")
    print(f"Max frames          : {request.max_frames}")
    print(f"Max seconds         : {request.max_seconds}")

    ai = SmartAttendanceAI(project_root=ROOT)
    try:
        started = ai.start_attendance(request)
        assert isinstance(started, AttendanceSessionSnapshotAIResult)
        print(f"Start status        : {started.status}")
        print(f"Contract version    : {started.contract_version}")
        print(f"Gallery version     : {started.gallery_version_id}")

        terminal = ai.wait_for_attendance(
            request.session_id,
            timeout=args.timeout,
        )
        assert isinstance(terminal, AttendanceSessionSnapshotAIResult)

        result = ai.get_attendance_result(request.session_id)
        if result is None:
            raise RuntimeError("Final attendance result is unavailable.")
        assert isinstance(result, AttendanceSessionAIResult)

        roster = set(request.roster_student_ids)
        result_students = {record.student_id for record in result.attendance}
        present = set(result.present_student_ids)
        absent = set(result.absent_student_ids)

        assert terminal.status == result.status
        assert result_students == roster
        assert present.isdisjoint(absent)
        assert present | absent == roster
        assert len(result.attendance) == len(roster)
        assert result.metrics.frames_processed == terminal.frames_processed
        assert result.metadata == metadata
        assert result.extensions == extensions

        # Backwards-compatible dict-style access remains available.
        assert result["session_id"] == result.session_id
        assert terminal["status"] == terminal.status

        if expected_present is not None:
            assert sorted(expected_present) == result.present_student_ids
        if expected_absent is not None:
            assert sorted(expected_absent) == result.absent_student_ids

        payload = result.to_dict()
        json.dumps(payload, ensure_ascii=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

        db_attendance = list(result.iter_attendance_records())
        db_events = list(result.iter_event_records())

        print()
        print("=" * 70)
        print("FINAL CONTRACT SUMMARY")
        print("=" * 70)
        print(f"Terminal status     : {result.status}")
        print(f"Frames processed    : {result.metrics.frames_processed}")
        print(f"Present count       : {result.present_count}")
        print(f"Absent count        : {result.absent_count}")
        print(f"Present students    : {result.present_student_ids}")
        print(f"Absent students     : {result.absent_student_ids}")
        print(f"Attendance records  : {len(db_attendance)}")
        print(f"Event records       : {len(db_events)}")
        print(f"Source type         : {result.source_type}")
        print(f"Gallery SHA256      : {result.gallery_sha256}")
        print(f"JSON output         : {output_path}")
        print()
        print("FINAL ATTENDANCE AI CONTRACT TEST PASSED")

    finally:
        ai.shutdown(wait=True)


if __name__ == "__main__":
    main()