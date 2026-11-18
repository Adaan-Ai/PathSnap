"""Spatial-indexed road-arc candidate generation for one trip at a time."""

from __future__ import annotations

import math
from dataclasses import dataclass

from pathsnap.data import TripData

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class RoadCandidate:
    arc_id: int
    perpendicular_distance_m: float
    projection_fraction: float
    segment_length_m: float
    road_bearing_deg: float
    movement_bearing_deg: float | None
    direction_difference_deg: float | None


@dataclass(frozen=True)
class ObservationCandidates:
    observation_index: int
    timestamp_s: float
    candidates: tuple[RoadCandidate, ...]
    status: str


def _bearing(dx: float, dy: float) -> float:
    return math.degrees(math.atan2(dx, dy)) % 360.0


def _angle_difference(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


class CandidateGenerator:
    """Find nearest directed arcs within a radius using a local grid index.

    Distances use a trip-local equirectangular tangent-plane projection. Keep
    the radius configurable and re-check it on training trips before selecting
    final production settings. Arc IDs remain local to the supplied trip.
    """

    def __init__(self, radius_m: float = 50.0, max_candidates: int = 8, cell_size_m: float = 100.0) -> None:
        if radius_m <= 0 or max_candidates <= 0 or cell_size_m <= 0:
            raise ValueError("radius_m, max_candidates, and cell_size_m must be positive")
        self.radius_m = radius_m
        self.max_candidates = max_candidates
        self.cell_size_m = cell_size_m

    def generate(self, trip: TripData, observation_indices: list[int] | tuple[int, ...] | None = None) -> list[ObservationCandidates]:
        selected_indices = list(range(len(trip.observations))) if observation_indices is None else list(observation_indices)
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError("observation_indices must not contain duplicates")
        if any(index < 0 or index >= len(trip.observations) for index in selected_indices):
            raise IndexError("observation index out of range")
        if not trip.observations or not trip.nodes or not trip.arcs:
            return [ObservationCandidates(i, trip.observations[i].timestamp_s, (), "empty_network_or_track")
                    for i in selected_indices]

        center_lon = sum(node[0] for node in trip.nodes) / len(trip.nodes)
        center_lat = sum(node[1] for node in trip.nodes) / len(trip.nodes)
        x_scale = math.pi * EARTH_RADIUS_M / 180.0 * math.cos(math.radians(center_lat))
        y_scale = math.pi * EARTH_RADIUS_M / 180.0
        projected_nodes = [((lon - center_lon) * x_scale, (lat - center_lat) * y_scale)
                           for lon, lat in trip.nodes]

        grid: dict[tuple[int, int], list[int]] = {}
        wide_arcs: list[int] = []
        for arc in trip.arcs:
            x1, y1 = projected_nodes[arc.from_node]
            x2, y2 = projected_nodes[arc.to_node]
            c0, c1 = math.floor(min(x1, x2) / self.cell_size_m), math.floor(max(x1, x2) / self.cell_size_m)
            r0, r1 = math.floor(min(y1, y2) / self.cell_size_m), math.floor(max(y1, y2) / self.cell_size_m)
            cell_count = (c1 - c0 + 1) * (r1 - r0 + 1)
            if cell_count > 4096:
                wide_arcs.append(arc.arc_id)
                continue
            for cell_x in range(c0, c1 + 1):
                for cell_y in range(r0, r1 + 1):
                    grid.setdefault((cell_x, cell_y), []).append(arc.arc_id)

        reach = math.ceil(self.radius_m / self.cell_size_m) + 1
        results: list[ObservationCandidates] = []
        for index in selected_indices:
            observation = trip.observations[index]
            x, y = ((observation.longitude - center_lon) * x_scale,
                    (observation.latitude - center_lat) * y_scale)
            cell_x, cell_y = math.floor(x / self.cell_size_m), math.floor(y / self.cell_size_m)
            possible = set(wide_arcs)
            for cx in range(cell_x - reach, cell_x + reach + 1):
                for cy in range(cell_y - reach, cell_y + reach + 1):
                    possible.update(grid.get((cx, cy), ()))

            movement_bearing: float | None = None
            if index > 0:
                previous = trip.observations[index - 1]
                dt = observation.timestamp_s - previous.timestamp_s
                if dt > 0 and (previous.longitude != observation.longitude or previous.latitude != observation.latitude):
                    prev_x = (previous.longitude - center_lon) * x_scale
                    prev_y = (previous.latitude - center_lat) * y_scale
                    movement_bearing = _bearing(x - prev_x, y - prev_y)

            candidates: list[RoadCandidate] = []
            for arc_id in possible:
                arc = trip.arcs[arc_id]
                x1, y1 = projected_nodes[arc.from_node]
                x2, y2 = projected_nodes[arc.to_node]
                dx, dy = x2 - x1, y2 - y1
                length_sq = dx * dx + dy * dy
                fraction = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
                nearest_x, nearest_y = x1 + fraction * dx, y1 + fraction * dy
                distance = math.hypot(x - nearest_x, y - nearest_y)
                if distance > self.radius_m:
                    continue
                bearing = _bearing(dx, dy) if length_sq else 0.0
                difference = _angle_difference(movement_bearing, bearing) if movement_bearing is not None else None
                candidates.append(RoadCandidate(arc.arc_id, distance, fraction, math.sqrt(length_sq),
                                                bearing, movement_bearing, difference))
            candidates.sort(key=lambda candidate: (candidate.perpendicular_distance_m, candidate.arc_id))
            status = "ok" if candidates else "no_candidate_within_radius"
            results.append(ObservationCandidates(index, observation.timestamp_s,
                                                  tuple(candidates[:self.max_candidates]), status))
        return results


def candidate_features(trip: TripData, generated: list[ObservationCandidates]) -> list[dict[str, float | int | None]]:
    """Return a flat candidate table; labels are intentionally not synthesized."""
    rows: list[dict[str, float | int | None]] = []
    outgoing: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for road_arc in trip.arcs:
        outgoing[road_arc.from_node] = outgoing.get(road_arc.from_node, 0) + 1
        incoming[road_arc.to_node] = incoming.get(road_arc.to_node, 0) + 1
    for observation_candidates in generated:
        index = observation_candidates.observation_index
        observation = trip.observations[index]
        if index:
            previous = trip.observations[index - 1]
            delta_t = observation.timestamp_s - previous.timestamp_s
            lat_mid = math.radians((observation.latitude + previous.latitude) / 2)
            dx = math.radians(observation.longitude - previous.longitude) * EARTH_RADIUS_M * math.cos(lat_mid)
            dy = math.radians(observation.latitude - previous.latitude) * EARTH_RADIUS_M
            gps_step = math.hypot(dx, dy)
        else:
            delta_t, gps_step = None, None
        speed = gps_step / delta_t if gps_step is not None and delta_t and delta_t > 0 else None
        for candidate in observation_candidates.candidates:
            arc = trip.arcs[candidate.arc_id]
            rows.append({
                "trip_id": trip.trip_id,
                "observation_index": index,
                "arc_id": candidate.arc_id,
                "perpendicular_distance_m": candidate.perpendicular_distance_m,
                "projection_fraction": candidate.projection_fraction,
                "segment_length_m": candidate.segment_length_m,
                "road_bearing_deg": candidate.road_bearing_deg,
                "movement_bearing_deg": candidate.movement_bearing_deg,
                "direction_difference_deg": candidate.direction_difference_deg,
                "gps_step_distance_m": gps_step,
                "delta_t_s": delta_t,
                "gps_speed_mps": speed,
                "from_node_degree": outgoing.get(arc.from_node, 0),
                "to_node_degree": incoming.get(arc.to_node, 0),
            })
    return rows
