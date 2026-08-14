"""Regression tests for temporal identity evidence handling."""

import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from smart_attendance_ai.temporal_identity import TemporalIdentityResolver


def result(status, student_id="", score=0.0, margin=0.0):
    return SimpleNamespace(
        status=status,
        student_id=student_id,
        score=score,
        margin=margin,
    )


def add(resolver, track_id, index, match_result):
    return resolver.add_result(
        track_id=track_id,
        frame_index=index,
        timestamp_seconds=index / 30.0,
        match_result=match_result,
    )


def test_sparse_matches_survive_unknowns():
    resolver = TemporalIdentityResolver(
        history_size=8,
        minimum_observations=3,
        minimum_support_ratio=0.60,
        minimum_average_score=0.55,
        minimum_average_margin=0.08,
    )

    sequence = [
        result("MATCH", "STU_010", 0.61, 0.20),
        result("UNKNOWN", "STU_010", 0.52, 0.18),
        result("UNKNOWN", "STU_010", 0.51, 0.17),
        result("MATCH", "STU_010", 0.59, 0.16),
        result("UNKNOWN", "STU_010", 0.50, 0.14),
        result("UNKNOWN", "STU_010", 0.53, 0.19),
        result("MATCH", "STU_010", 0.58, 0.15),
    ]

    confirmed_now = False
    state = None
    for index, item in enumerate(sequence):
        state, confirmed_now = add(resolver, 24, index, item)

    assert confirmed_now
    assert state is not None
    assert state.confirmed_student_id == "STU_010"
    assert state.winner_observations == 3
    assert state.support_ratio == 1.0


def test_unknowns_alone_never_confirm():
    resolver = TemporalIdentityResolver()

    for index in range(20):
        state, confirmed_now = add(
            resolver,
            30,
            index,
            result("UNKNOWN", "STU_006", 0.53, 0.20),
        )
        assert not confirmed_now

    assert state.confirmed_student_id == ""
    assert state.winner_observations == 0


def test_competing_accepted_identities_reduce_support():
    resolver = TemporalIdentityResolver(
        history_size=8,
        minimum_observations=3,
        minimum_support_ratio=0.75,
        minimum_average_score=0.55,
        minimum_average_margin=0.08,
    )

    sequence = [
        result("MATCH", "STU_006", 0.60, 0.20),
        result("MATCH", "STU_014", 0.62, 0.18),
        result("MATCH", "STU_006", 0.59, 0.19),
        result("MATCH", "STU_014", 0.61, 0.17),
        result("MATCH", "STU_006", 0.58, 0.16),
    ]

    for index, item in enumerate(sequence):
        state, confirmed_now = add(resolver, 31, index, item)

    assert not confirmed_now
    assert state.confirmed_student_id == ""
    assert state.winner_observations == 3
    assert abs(state.support_ratio - 0.60) < 1e-9


def main():
    print("=" * 70)
    print("TEMPORAL EVIDENCE TEST")
    print("=" * 70)

    test_sparse_matches_survive_unknowns()
    print("Sparse accepted matches survive UNKNOWN observations: PASSED")

    test_unknowns_alone_never_confirm()
    print("UNKNOWN observations alone never confirm identity: PASSED")

    test_competing_accepted_identities_reduce_support()
    print("Competing MATCH identities reduce support: PASSED")

    print()
    print("TEMPORAL EVIDENCE TEST PASSED")


if __name__ == "__main__":
    main()