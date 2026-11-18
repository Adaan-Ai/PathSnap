"""Monotonic, geometry-based alignment of route arcs to GPS observations.

The output is an observation-level training target derived from the supplied
route sequence, not an independently verified per-observation ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from pathsnap.candidates import ObservationCandidates
from pathsnap.data import TripData


@dataclass(frozen=True)
class RouteAlignment:
    trip_id: str
    arc_ids_by_observation: tuple[int | None, ...]
    route_positions_by_observation: tuple[int | None, ...]
    aligned_observations: int
    observation_count: int
    alignment_segments: int
    unmatched_observations: tuple[int, ...]


@dataclass(frozen=True)
class _State:
    route_position: int
    arc_id: int
    cost: float
    previous_state: int | None


def align_route_to_track(
    trip: TripData,
    candidates: list[ObservationCandidates],
    *,
    sigma_m: float = 20.0,
    advance_penalty: float = 0.4,
    stay_penalty: float = 0.2,
) -> RouteAlignment:
    """Assign candidate arcs to observations with a route-order Viterbi pass.

    Each observation may stay on the current route arc or move forward through
    the supplied route sequence; backward route movement is disallowed. GPS to
    candidate distance supplies the emission cost. Observations with no route
    arc among their spatial candidates remain explicitly unmatched. The
    distance scale and transition penalties are initial analysis parameters,
    not tuned settings.
    """
    if trip.route_arc_ids is None:
        raise ValueError("Route labels were not loaded; call load_trip(..., include_route_labels=True)")
    if len(candidates) != len(trip.observations):
        raise ValueError("Candidate results must contain one row per GPS observation")
    if sigma_m <= 0 or advance_penalty < 0 or stay_penalty < 0:
        raise ValueError("sigma_m must be positive and penalties must be non-negative")

    route_positions: dict[int, list[int]] = {}
    for position, arc_id in enumerate(trip.route_arc_ids):
        route_positions.setdefault(arc_id, []).append(position)

    # Events contain only observations with at least one reference-route arc
    # among their candidates. Their original indices remain in the result.
    events: list[tuple[int, list[tuple[int, int, float]]]] = []
    for row in candidates:
        states: list[tuple[int, int, float]] = []
        for candidate in row.candidates:
            for route_position in route_positions.get(candidate.arc_id, ()):
                emission = (candidate.perpendicular_distance_m / sigma_m) ** 2 / 2.0
                states.append((route_position, candidate.arc_id, emission))
        if states:
            events.append((row.observation_index, states))

    output_arcs: list[int | None] = [None] * len(trip.observations)
    output_positions: list[int | None] = [None] * len(trip.observations)
    matched_indices = {event_index for event_index, _ in events}
    unmatched = tuple(index for index in range(len(trip.observations)) if index not in matched_indices)
    if not events:
        return RouteAlignment(trip.trip_id, tuple(output_arcs), tuple(output_positions), 0,
                              len(trip.observations), 0, unmatched)

    event_states: list[list[_State]] = []
    segment_ranges: list[tuple[int, int]] = []
    segment_start = 0
    for event_index, (observation_index, current_options) in enumerate(events):
        previous_states = event_states[-1] if event_index > segment_start else []
        previous_observation_index = events[event_index - 1][0] if event_index > segment_start else None
        dt = (trip.observations[observation_index].timestamp_s -
              trip.observations[previous_observation_index].timestamp_s) if previous_observation_index is not None else 0
        current_states: list[_State] = []
        reachable = False
        for route_position, arc_id, emission in current_options:
            best_cost = emission
            best_previous: int | None = None
            for previous_index, previous in enumerate(previous_states):
                advance = route_position - previous.route_position
                if advance < 0:
                    continue
                # Longer GPS gaps can plausibly cover more route arcs. The
                # transition remains monotonic, but the skip cost is relaxed.
                gap_credit = max(0, int(max(dt, 0.0) // 5.0))
                excess_advance = max(0, advance - 1 - gap_credit)
                transition = (stay_penalty if advance == 0 else advance_penalty + advance_penalty * excess_advance)
                total = previous.cost + transition + emission
                if total < best_cost or best_previous is None:
                    best_cost, best_previous = total, previous_index
            if previous_states and best_previous is None:
                continue
            if best_previous is not None:
                reachable = True
            current_states.append(_State(route_position, arc_id, best_cost, best_previous))
        if event_index > segment_start and not reachable:
            segment_ranges.append((segment_start, event_index - 1))
            segment_start = event_index
            current_states = [_State(position, arc_id, emission, None)
                              for position, arc_id, emission in current_options]
        event_states.append(current_states)
    segment_ranges.append((segment_start, len(events) - 1))

    for start, end in segment_ranges:
        final_state_index = min(range(len(event_states[end])), key=lambda i: event_states[end][i].cost)
        for event_index in range(end, start - 1, -1):
            state = event_states[event_index][final_state_index]
            observation_index = events[event_index][0]
            output_arcs[observation_index] = state.arc_id
            output_positions[observation_index] = state.route_position
            if event_index > start:
                if state.previous_state is None:
                    break
                final_state_index = state.previous_state

    aligned = sum(arc_id is not None for arc_id in output_arcs)
    return RouteAlignment(trip.trip_id, tuple(output_arcs), tuple(output_positions), aligned,
                          len(trip.observations), len(segment_ranges), unmatched)
