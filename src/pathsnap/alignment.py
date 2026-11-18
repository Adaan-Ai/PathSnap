"""Monotonic geometry-based alignment of route arcs to GPS observations.

The output is a weak observation-level training target derived from the route
sequence. The source dataset does not provide independently verified labels
per GPS observation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pathsnap.data import TripData

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class RouteAlignment:
    trip_id: str
    arc_ids_by_observation: tuple[int | None, ...]
    route_positions_by_observation: tuple[int | None, ...]
    distances_m: tuple[float | None, ...]
    aligned_observations: int
    observation_count: int
    alignment_segments: int
    unmatched_observations: tuple[int, ...]


@dataclass(frozen=True)
class _State:
    route_position: int
    arc_id: int
    distance_m: float
    cost: float
    previous_state: int | None


def _route_arc_distances(trip: TripData, *, chunk_size: int = 128) -> np.ndarray:
    """Compute GPS-to-reference-route segment distances in a local projection."""
    if trip.route_arc_ids is None:
        raise ValueError("Route labels were not loaded; call load_trip(..., include_route_labels=True)")
    if not trip.route_arc_ids or not trip.observations:
        return np.empty((len(trip.observations), 0), dtype=np.float64)

    center_lon = sum(node[0] for node in trip.nodes) / len(trip.nodes)
    center_lat = sum(node[1] for node in trip.nodes) / len(trip.nodes)
    x_scale = math.pi * EARTH_RADIUS_M / 180.0 * math.cos(math.radians(center_lat))
    y_scale = math.pi * EARTH_RADIUS_M / 180.0
    nodes = np.asarray([((lon - center_lon) * x_scale, (lat - center_lat) * y_scale)
                        for lon, lat in trip.nodes], dtype=np.float64)
    route_arcs = [trip.arcs[arc_id] for arc_id in trip.route_arc_ids]
    starts = nodes[[arc.from_node for arc in route_arcs]]
    vectors = nodes[[arc.to_node for arc in route_arcs]] - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    lengths_sq[lengths_sq == 0] = 1.0
    output = np.empty((len(trip.observations), len(route_arcs)), dtype=np.float64)

    for start_index in range(0, len(trip.observations), chunk_size):
        end_index = min(start_index + chunk_size, len(trip.observations))
        observations = trip.observations[start_index:end_index]
        points = np.asarray([((obs.longitude - center_lon) * x_scale,
                              (obs.latitude - center_lat) * y_scale) for obs in observations])
        relative = points[:, None, :] - starts[None, :, :]
        fraction = np.einsum("ijk,jk->ij", relative, vectors) / lengths_sq[None, :]
        np.clip(fraction, 0.0, 1.0, out=fraction)
        nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
        delta = points[:, None, :] - nearest
        output[start_index:end_index] = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta))
    return output


def align_route_to_track(
    trip: TripData,
    *,
    sigma_m: float = 20.0,
    max_alignment_distance_m: float = 250.0,
    states_per_observation: int = 8,
    advance_penalty: float = 0.4,
    stay_penalty: float = 0.2,
) -> RouteAlignment:
    """Align GPS samples to nearby reference-route arcs in chronological order.

    Emissions are GPS-to-route-segment distances. Candidate states are drawn
    from the route itself, independently of the prediction candidate list, so
    candidate miss rate can later be measured by comparing the aligned arc ID
    with generated prediction candidates. The route position is constrained to
    stay or move forward; large time gaps relax the cost for skipping route
    arcs. Points with no route arc within the configured limit remain unmatched.

    The distance threshold, emission scale, and transition penalties are
    explicit initial settings. The resulting labels are weak supervision and
    must be validated; they are not asserted to be physical ground truth.
    """
    if trip.route_arc_ids is None:
        raise ValueError("Route labels were not loaded; call load_trip(..., include_route_labels=True)")
    if sigma_m <= 0 or max_alignment_distance_m <= 0 or states_per_observation <= 0:
        raise ValueError("Distance settings and states_per_observation must be positive")
    if advance_penalty < 0 or stay_penalty < 0:
        raise ValueError("Transition penalties must be non-negative")

    distances = _route_arc_distances(trip)
    route_ids = trip.route_arc_ids
    events: list[tuple[int, list[tuple[int, int, float]]]] = []
    for observation_index, row in enumerate(distances):
        within = np.flatnonzero(row <= max_alignment_distance_m)
        if not len(within):
            continue
        if len(within) > states_per_observation:
            selected = np.argpartition(row[within], states_per_observation - 1)[:states_per_observation]
            within = within[selected]
        within = within[np.argsort(row[within], kind="stable")]
        states = [(int(position), int(route_ids[position]), float(row[position])) for position in within]
        events.append((observation_index, states))

    observation_count = len(trip.observations)
    output_arcs: list[int | None] = [None] * observation_count
    output_positions: list[int | None] = [None] * observation_count
    output_distances: list[float | None] = [None] * observation_count
    matched_indices = {index for index, _ in events}
    unmatched = tuple(index for index in range(observation_count) if index not in matched_indices)
    if not events:
        return RouteAlignment(trip.trip_id, tuple(output_arcs), tuple(output_positions),
                              tuple(output_distances), 0, observation_count, 0, unmatched)

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
        for route_position, arc_id, distance_m in current_options:
            emission = (distance_m / sigma_m) ** 2 / 2.0
            best_cost = emission
            best_previous: int | None = None
            for previous_index, previous in enumerate(previous_states):
                advance = route_position - previous.route_position
                if advance < 0:
                    continue
                gap_credit = max(0, int(max(dt, 0.0) // 5.0))
                excess_advance = max(0, advance - 1 - gap_credit)
                transition = (stay_penalty if advance == 0 else
                              advance_penalty + advance_penalty * excess_advance)
                total = previous.cost + transition + emission
                if total < best_cost or best_previous is None:
                    best_cost, best_previous = total, previous_index
            if previous_states and best_previous is None:
                continue
            if best_previous is not None:
                reachable = True
            current_states.append(_State(route_position, arc_id, distance_m, best_cost, best_previous))
        if event_index > segment_start and not reachable:
            segment_ranges.append((segment_start, event_index - 1))
            segment_start = event_index
            current_states = [_State(position, arc_id, distance,
                                     (distance / sigma_m) ** 2 / 2.0, None)
                              for position, arc_id, distance in current_options]
        event_states.append(current_states)
    segment_ranges.append((segment_start, len(events) - 1))

    for start, end in segment_ranges:
        state_index = min(range(len(event_states[end])), key=lambda i: event_states[end][i].cost)
        for event_index in range(end, start - 1, -1):
            state = event_states[event_index][state_index]
            observation_index = events[event_index][0]
            output_arcs[observation_index] = state.arc_id
            output_positions[observation_index] = state.route_position
            output_distances[observation_index] = state.distance_m
            if event_index > start:
                if state.previous_state is None:
                    break
                state_index = state.previous_state

    aligned = sum(arc_id is not None for arc_id in output_arcs)
    return RouteAlignment(trip.trip_id, tuple(output_arcs), tuple(output_positions),
                          tuple(output_distances), aligned, observation_count,
                          len(segment_ranges), unmatched)
