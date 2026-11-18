"""HMM-style sequence smoothing and Viterbi decoding for arc candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pathsnap.candidates import ObservationCandidates, RoadCandidate
from pathsnap.data import TripData
from pathsnap.eda import haversine_m


@dataclass(frozen=True)
class DecodeResult:
    arc_ids: tuple[int | None, ...]
    candidate_only_arc_ids: tuple[int | None, ...]
    connected_transition_rate: float
    transition_count: int
    candidate_only_connected_transition_rate: float


def _gps_step_m(trip: TripData, right_index: int) -> float:
    left, right = trip.observations[right_index - 1], trip.observations[right_index]
    return haversine_m(left.longitude, left.latitude, right.longitude, right.latitude)


def _transition_log_probability(trip: TripData, previous: RoadCandidate,
                                 current: RoadCandidate, step_m: float,
                                 sigma_m: float) -> float:
    previous_arc, current_arc = trip.arcs[previous.arc_id], trip.arcs[current.arc_id]
    if previous.arc_id == current.arc_id:
        network_m = (current.projection_fraction - previous.projection_fraction) * previous.segment_length_m
        # A small backtrack can result from noisy projections; larger reversals
        # receive a penalty while remaining possible under the noise model.
        backtrack_penalty = max(0.0, -network_m) * 0.08
        network_m = max(0.0, network_m)
        connectivity_penalty = 0.0
    elif previous_arc.to_node == current_arc.from_node:
        network_m = ((1.0 - previous.projection_fraction) * previous.segment_length_m
                     + current.projection_fraction * current.segment_length_m)
        backtrack_penalty = 0.0
        connectivity_penalty = 0.0
    else:
        # Preserve a finite recovery path for gaps, noisy samples, and known
        # discontinuous route encodings; topologically connected transitions
        # are strongly preferred when their distances are plausible.
        network_m = (max(0.0, (1.0 - previous.projection_fraction) * previous.segment_length_m)
                     + max(0.0, current.projection_fraction * current.segment_length_m))
        backtrack_penalty = 0.0
        connectivity_penalty = 5.0
    distance_penalty = abs(network_m - step_m) / max(sigma_m, 1.0)
    return -(distance_penalty + backtrack_penalty + connectivity_penalty)


def _decode_segment(trip: TripData, observations: list[ObservationCandidates],
                    emission_rows: list[np.ndarray], sigma_m: float) -> list[int]:
    backpointers: list[np.ndarray] = []
    previous = np.log(np.maximum(emission_rows[0], 1e-12))
    for offset in range(1, len(observations)):
        left_candidates = observations[offset - 1].candidates
        right_candidates = observations[offset].candidates
        step_m = _gps_step_m(trip, observations[offset].observation_index)
        sigma = sigma_m + 0.25 * step_m
        transitions = np.empty((len(left_candidates), len(right_candidates)), dtype=float)
        for left_i, left_candidate in enumerate(left_candidates):
            for right_i, right_candidate in enumerate(right_candidates):
                transitions[left_i, right_i] = _transition_log_probability(
                    trip, left_candidate, right_candidate, step_m, sigma)
        joint = previous[:, None] + transitions
        backpointers.append(np.argmax(joint, axis=0))
        previous = np.max(joint, axis=0) + np.log(np.maximum(emission_rows[offset], 1e-12))
    state = int(np.argmax(previous))
    states = [state]
    for pointer in reversed(backpointers):
        state = int(pointer[state])
        states.append(state)
    return list(reversed(states))


def viterbi_decode(trip: TripData, generated: list[ObservationCandidates],
                   candidate_probabilities: list[np.ndarray], *, sigma_m: float = 25.0) -> DecodeResult:
    """Decode a most likely arc sequence; probabilities are per-candidate emissions.

    Empty-candidate observations remain `None` and split decoding into
    contiguous segments. Transition probabilities reward road-network
    connectivity and agreement between road distance and GPS displacement.
    """
    if len(generated) != len(candidate_probabilities):
        raise ValueError("one candidate probability vector is required per observation")
    total = len(trip.observations)
    decoded: list[int | None] = [None] * total
    candidate_only: list[int | None] = [None] * total
    for row, probabilities in zip(generated, candidate_probabilities):
        if len(row.candidates) != len(probabilities):
            raise ValueError("candidate probability count does not match candidate count")
        if not row.candidates:
            continue
        if any(not math.isfinite(float(value)) or value < 0 for value in probabilities):
            raise ValueError("emission probabilities must be finite and nonnegative")
        candidate_only[row.observation_index] = row.candidates[int(np.argmax(probabilities))].arc_id

    cursor = 0
    while cursor < len(generated):
        while cursor < len(generated) and not generated[cursor].candidates:
            cursor += 1
        start = cursor
        while cursor < len(generated) and generated[cursor].candidates:
            cursor += 1
        if start == cursor:
            continue
        segment_rows = generated[start:cursor]
        states = _decode_segment(trip, segment_rows, candidate_probabilities[start:cursor], sigma_m)
        for row, state in zip(segment_rows, states):
            decoded[row.observation_index] = row.candidates[state].arc_id

    def continuity(sequence: list[int | None]) -> tuple[float, int]:
        pairs = [(left, right) for left, right in zip(sequence, sequence[1:])
                 if left is not None and right is not None]
        if not pairs:
            return 0.0, 0
        connected = sum(int(trip.arcs[left].to_node == trip.arcs[right].from_node
                            or left == right) for left, right in pairs)
        return connected / len(pairs), len(pairs)

    rate, count = continuity(decoded)
    candidate_rate, _ = continuity(candidate_only)
    return DecodeResult(tuple(decoded), tuple(candidate_only), rate, count, candidate_rate)
