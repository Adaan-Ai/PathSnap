"""Strict, trip-scoped readers for PathSnap's whitespace data files."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrackObservation:
    longitude: float
    latitude: float
    timestamp_s: float


@dataclass(frozen=True)
class RoadArc:
    arc_id: int
    from_node: int
    to_node: int


@dataclass(frozen=True)
class TripData:
    trip_id: str
    observations: tuple[TrackObservation, ...]
    nodes: tuple[tuple[float, float], ...]
    arcs: tuple[RoadArc, ...]
    route_arc_ids: tuple[int, ...] | None


def _numeric_rows(path: Path, width: int) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if len(tokens) != width:
                raise ValueError(f"{path}:{line_number}: expected {width} columns, found {len(tokens)}")
            try:
                row = [float(token) for token in tokens]
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from error
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"{path}:{line_number}: non-finite value")
            rows.append(row)
    return rows


def load_trip(directory: Path, *, include_route_labels: bool = False) -> TripData:
    """Load and validate a trip; reference labels are opt-in by design."""
    trip_id = directory.name
    paths = {suffix: directory / f"{trip_id}{suffix}" for suffix in (".track", ".nodes", ".arcs", ".route")}
    missing = [str(path) for suffix, path in paths.items() if suffix != ".route" and not path.is_file()]
    if include_route_labels and not paths[".route"].is_file():
        missing.append(str(paths[".route"]))
    if missing:
        raise FileNotFoundError("Missing required trip file(s): " + ", ".join(missing))

    track_rows = _numeric_rows(paths[".track"], 3)
    node_rows = _numeric_rows(paths[".nodes"], 2)
    arc_rows = _numeric_rows(paths[".arcs"], 2)
    observations: list[TrackObservation] = []
    for index, (lon, lat, timestamp) in enumerate(track_rows):
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"{paths['.track']}:{index + 1}: coordinate outside longitude/latitude bounds")
        observations.append(TrackObservation(lon, lat, timestamp))
    nodes: list[tuple[float, float]] = []
    for index, (lon, lat) in enumerate(node_rows):
        if not (-180 <= lon <= 180 and -90 <= lat <= 90):
            raise ValueError(f"{paths['.nodes']}:{index + 1}: coordinate outside longitude/latitude bounds")
        nodes.append((lon, lat))
    arcs: list[RoadArc] = []
    for index, (start_raw, end_raw) in enumerate(arc_rows):
        if not start_raw.is_integer() or not end_raw.is_integer():
            raise ValueError(f"{paths['.arcs']}:{index + 1}: node IDs must be integers")
        start, end = int(start_raw), int(end_raw)
        if min(start, end) < 0 or max(start, end) >= len(nodes):
            raise ValueError(f"{paths['.arcs']}:{index + 1}: node ID is out of range")
        arcs.append(RoadArc(index, start, end))

    route: tuple[int, ...] | None = None
    if include_route_labels:
        route_rows = _numeric_rows(paths[".route"], 1)
        route_ids: list[int] = []
        for index, (arc_raw,) in enumerate(route_rows):
            if not arc_raw.is_integer() or arc_raw < 0 or arc_raw >= len(arcs):
                raise ValueError(f"{paths['.route']}:{index + 1}: route arc ID is invalid for this trip")
            route_ids.append(int(arc_raw))
        route = tuple(route_ids)
    return TripData(trip_id, tuple(observations), tuple(nodes), tuple(arcs), route)


def load_partition(root: Path, *, include_route_labels: bool = False) -> list[TripData]:
    """Load each immediate trip directory without mixing trip-local arc IDs."""
    if not root.is_dir():
        raise FileNotFoundError(f"Partition directory not found: {root}")
    return [load_trip(directory, include_route_labels=include_route_labels)
            for directory in sorted(path for path in root.iterdir() if path.is_dir())]
