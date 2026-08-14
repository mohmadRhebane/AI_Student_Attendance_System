"""Temporal identity confirmation for tracked faces."""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple


@dataclass
class IdentityObservation:
    """One matcher observation for one face track."""

    frame_index: int
    timestamp_seconds: float
    status: str
    student_id: str
    score: float
    margin: float


@dataclass
class TrackIdentityState:
    """Accumulated temporal identity state for one tracker ID."""

    track_id: int

    # Recent raw observations, including MATCH, AMBIGUOUS and UNKNOWN.
    # This remains useful for diagnostics and first-seen information.
    history: Deque[IdentityObservation]

    # Recent accepted identity evidence only. UNKNOWN and AMBIGUOUS are
    # absence of accepted evidence; they must not erase earlier valid MATCH
    # observations for the same physical track.
    match_history: Deque[IdentityObservation]

    confirmed_student_id: str = ""
    confirmed_at_seconds: Optional[float] = None
    confirmed_at_frame: Optional[int] = None
    conflict_count: int = 0

    latest_status: str = "PENDING"
    latest_student_id: str = ""
    latest_score: float = 0.0
    latest_margin: float = 0.0

    winner_observations: int = 0
    support_ratio: float = 0.0
    average_score: float = 0.0
    average_margin: float = 0.0


class TemporalIdentityResolver:
    """
    Confirm an identity from repeated accepted matcher evidence.

    The resolver keeps two histories:

    * ``history`` stores all recent matcher outcomes for diagnostics.
    * ``match_history`` stores only accepted MATCH outcomes and is used for
      identity consensus.

    This distinction is important for crowded or oblique camera views where
    a correct face may intermittently fall below ``min_score``. Such UNKNOWN
    observations are not evidence for another identity, so they should not
    dilute or evict accepted evidence from the consensus calculation.
    """

    def __init__(
        self,
        history_size: int = 8,
        minimum_observations: int = 3,
        minimum_support_ratio: float = 0.60,
        minimum_average_score: float = 0.55,
        minimum_average_margin: float = 0.08,
    ) -> None:
        self.history_size = int(history_size)
        self.minimum_observations = int(minimum_observations)
        self.minimum_support_ratio = float(minimum_support_ratio)
        self.minimum_average_score = float(minimum_average_score)
        self.minimum_average_margin = float(minimum_average_margin)

        if self.history_size <= 0:
            raise ValueError("history_size must be greater than zero.")
        if self.minimum_observations <= 0:
            raise ValueError(
                "minimum_observations must be greater than zero."
            )
        if not 0.0 <= self.minimum_support_ratio <= 1.0:
            raise ValueError(
                "minimum_support_ratio must be between 0 and 1."
            )

        self.states: Dict[int, TrackIdentityState] = {}

    def get_state(self, track_id: int) -> TrackIdentityState:
        track_id = int(track_id)

        if track_id not in self.states:
            self.states[track_id] = TrackIdentityState(
                track_id=track_id,
                history=deque(maxlen=self.history_size),
                match_history=deque(maxlen=self.history_size),
            )

        return self.states[track_id]

    def add_result(
        self,
        track_id: int,
        frame_index: int,
        timestamp_seconds: float,
        match_result,
    ) -> Tuple[TrackIdentityState, bool]:
        state = self.get_state(track_id)

        observation = IdentityObservation(
            frame_index=int(frame_index),
            timestamp_seconds=float(timestamp_seconds),
            status=str(match_result.status),
            student_id=str(match_result.student_id),
            score=float(match_result.score),
            margin=float(match_result.margin),
        )

        # Preserve every recent outcome for diagnostics/UI state.
        state.history.append(observation)

        # Only accepted matcher decisions constitute identity evidence.
        if observation.status == "MATCH" and observation.student_id:
            state.match_history.append(observation)

        state.latest_status = observation.status
        state.latest_student_id = observation.student_id
        state.latest_score = observation.score
        state.latest_margin = observation.margin

        confirmed_now = False

        if state.confirmed_student_id:
            if (
                observation.status == "MATCH"
                and observation.student_id
                and observation.student_id != state.confirmed_student_id
            ):
                state.conflict_count += 1

            return state, confirmed_now

        matched_observations = list(state.match_history)

        if not matched_observations:
            return state, confirmed_now

        student_counts = Counter(
            item.student_id for item in matched_observations
        )
        winner_id, winner_count = student_counts.most_common(1)[0]

        winner_items = [
            item
            for item in matched_observations
            if item.student_id == winner_id
        ]

        # Support measures agreement among accepted identity decisions.
        # UNKNOWN and AMBIGUOUS observations are not votes for a competing
        # identity and therefore are intentionally excluded here.
        support_ratio = winner_count / len(matched_observations)
        average_score = sum(item.score for item in winner_items) / len(
            winner_items
        )
        average_margin = sum(item.margin for item in winner_items) / len(
            winner_items
        )

        state.winner_observations = winner_count
        state.support_ratio = support_ratio
        state.average_score = average_score
        state.average_margin = average_margin

        can_confirm = (
            winner_count >= self.minimum_observations
            and support_ratio >= self.minimum_support_ratio
            and average_score >= self.minimum_average_score
            and average_margin >= self.minimum_average_margin
        )

        if can_confirm:
            state.confirmed_student_id = winner_id
            state.confirmed_at_seconds = float(timestamp_seconds)
            state.confirmed_at_frame = int(frame_index)
            confirmed_now = True

        return state, confirmed_now