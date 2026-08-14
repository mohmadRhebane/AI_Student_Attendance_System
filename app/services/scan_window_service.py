from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import List


@dataclass(frozen=True)
class ScanWindow:
    scan_number: int
    start_seconds: int
    duration_seconds: int
    end_seconds: int


def _round_interval_to_minute(
        total_duration_seconds: int,
        scan_count: int,
) -> int:
    raw_minutes = (
        Decimal(total_duration_seconds)
        / Decimal(scan_count + 1)
        / Decimal(60)
    )

    rounded_minutes = int(
        raw_minutes.quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )

    if rounded_minutes < 1:
        rounded_minutes = 1

    return rounded_minutes * 60


def build_scan_windows(
        video_duration_seconds: int,
        scan_count: int,
        scan_duration_seconds: int,
) -> List[ScanWindow]:
    if video_duration_seconds <= 0:
        raise ValueError(
            "video_duration_seconds must be greater than zero."
        )

    if scan_count <= 0:
        raise ValueError(
            "scan_count must be greater than zero."
        )

    if scan_duration_seconds <= 0:
        raise ValueError(
            "scan_duration_seconds must be greater than zero."
        )

    interval_seconds = _round_interval_to_minute(
        total_duration_seconds=video_duration_seconds,
        scan_count=scan_count,
    )

    if scan_duration_seconds >= interval_seconds:
        raise ValueError(
            "Scan duration is too long for the calculated interval."
        )

    windows = []

    for scan_number in range(
        1,
        scan_count + 1,
    ):
        start_seconds = (
            interval_seconds * scan_number
        )

        end_seconds = (
            start_seconds
            + scan_duration_seconds
        )

        if end_seconds >= video_duration_seconds:
            raise ValueError(
                "Calculated scan window reaches or exceeds "
                "the end of the video."
            )

        windows.append(
            ScanWindow(
                scan_number=scan_number,
                start_seconds=start_seconds,
                duration_seconds=scan_duration_seconds,
                end_seconds=end_seconds,
            )
        )

    return windows