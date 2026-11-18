"""Training-set inventory and exploratory data quality analysis.

The analysis deliberately operates on one declared partition at a time. It
does not infer a partition from trip numbers and never reads files outside the
directory passed on the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

SUFFIXES = (".track", ".nodes", ".arcs", ".route")
EARTH_RADIUS_M = 6_371_008.8


def read_rows(path: Path, expected_columns: int) -> tuple[list[list[float]], int]:
    """Read whitespace-separated numeric rows and count malformed lines."""
    rows: list[list[float]] = []
    malformed = 0
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != expected_columns:
                malformed += 1
                continue
            try:
                values = [float(value) for value in parts]
            except ValueError:
                malformed += 1
                continue
            if not all(math.isfinite(value) for value in values):
                malformed += 1
                continue
            rows.append(values)
    return rows, malformed


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Return great-circle distance in metres for longitude/latitude degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max": ordered[-1],
    }


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def analyze_trip(directory: Path) -> dict[str, object]:
    trip_id = directory.name
    files = {suffix: directory / f"{trip_id}{suffix}" for suffix in SUFFIXES}
    missing_files = [suffix for suffix, path in files.items() if not path.is_file()]
    record: dict[str, object] = {
        "trip_id": trip_id,
        "missing_files": ",".join(missing_files),
        "file_bytes": sum(path.stat().st_size for path in files.values() if path.is_file()),
    }
    if missing_files:
        record["error"] = "missing required file(s)"
        return record

    track, bad_track = read_rows(files[".track"], 3)
    nodes, bad_nodes = read_rows(files[".nodes"], 2)
    arcs_raw, bad_arcs = read_rows(files[".arcs"], 2)
    route_raw, bad_route = read_rows(files[".route"], 1)
    arcs: list[tuple[int, int]] = []
    bad_arc_ids = 0
    for start, end in arcs_raw:
        if start < 0 or end < 0 or not start.is_integer() or not end.is_integer():
            bad_arc_ids += 1
            continue
        arcs.append((int(start), int(end)))
    route: list[int] = []
    bad_route_ids = 0
    for (arc_id,) in route_raw:
        if arc_id < 0 or not arc_id.is_integer():
            bad_route_ids += 1
            continue
        route.append(int(arc_id))

    invalid_coords = sum(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat, _ in track)
    invalid_node_coords = sum(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat in nodes)
    time_gaps: list[float] = []
    movement_m: list[float] = []
    speeds: list[float] = []
    non_increasing = 0
    for previous, current in zip(track, track[1:]):
        lon1, lat1, time1 = previous
        lon2, lat2, time2 = current
        dt = time2 - time1
        time_gaps.append(dt)
        if dt <= 0:
            non_increasing += 1
        distance = haversine_m(lon1, lat1, lon2, lat2)
        movement_m.append(distance)
        if dt > 0:
            speeds.append(distance / dt)

    out_of_range_arcs = sum(start >= len(nodes) or end >= len(nodes) for start, end in arcs)
    out_of_range_route = sum(arc_id >= len(arcs_raw) for arc_id in route)
    node_degrees = [0] * len(nodes)
    graph = DisjointSet(len(nodes))
    for start, end in arcs:
        if start < len(nodes) and end < len(nodes):
            node_degrees[start] += 1
            node_degrees[end] += 1
            graph.union(start, end)
    components = len({graph.find(i) for i, degree in enumerate(node_degrees) if degree > 0})
    isolated_nodes = sum(degree == 0 for degree in node_degrees)
    discontinuities = 0
    valid_route = [arc_id for arc_id in route if arc_id < len(arcs_raw)]
    for left_id, right_id in zip(valid_route, valid_route[1:]):
        if left_id < len(arcs) and right_id < len(arcs):
            if arcs[left_id][1] != arcs[right_id][0]:
                discontinuities += 1

    valid_track = [(lon, lat) for lon, lat, _ in track if -180 <= lon <= 180 and -90 <= lat <= 90]
    record.update({
        "track_rows": len(track), "track_malformed_rows": bad_track,
        "node_rows": len(nodes), "node_malformed_rows": bad_nodes,
        "arc_rows": len(arcs_raw), "arc_malformed_rows": bad_arcs + bad_arc_ids,
        "route_rows": len(route_raw), "route_malformed_rows": bad_route + bad_route_ids,
        "invalid_track_coordinates": invalid_coords,
        "invalid_node_coordinates": invalid_node_coords,
        "non_increasing_timestamps": non_increasing,
        "non_1s_time_gaps": sum(abs(gap - 1.0) > 0.05 for gap in time_gaps),
        "gaps_over_5s": sum(gap > 5 for gap in time_gaps),
        "gps_gaps_over_5s": describe([gap for gap in time_gaps if gap > 5]),
        "time_gap_seconds": describe(time_gaps),
        "gps_step_distance_m": describe(movement_m),
        "gps_speed_mps": describe(speeds),
        "gps_extent": ({"min_lon": min(p[0] for p in valid_track), "max_lon": max(p[0] for p in valid_track),
                        "min_lat": min(p[1] for p in valid_track), "max_lat": max(p[1] for p in valid_track)}
                       if valid_track else None),
        "arcs_with_invalid_node_id": out_of_range_arcs,
        "route_ids_out_of_range": out_of_range_route,
        "isolated_nodes": isolated_nodes,
        "weak_components_with_edges": components,
        "route_directed_discontinuities": discontinuities,
    })
    return record


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fields = sorted({key for row in records for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                          for key, value in row.items()} for row in records)


def run(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    trip_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    records = [analyze_trip(directory) for directory in trip_dirs]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "training_trip_inventory.csv", records)

    numeric_fields = (
        "track_rows", "node_rows", "arc_rows", "route_rows", "file_bytes",
        "invalid_track_coordinates", "invalid_node_coordinates", "non_increasing_timestamps",
        "non_1s_time_gaps", "gaps_over_5s", "arcs_with_invalid_node_id",
        "route_ids_out_of_range", "isolated_nodes", "weak_components_with_edges",
        "route_directed_discontinuities",
    )
    aggregates: dict[str, object] = {}
    for field in numeric_fields:
        values = [float(row[field]) for row in records if isinstance(row.get(field), (int, float))]
        aggregates[field] = describe(values)
    availability = Counter(suffix for row in records for suffix in SUFFIXES
                           if suffix not in str(row.get("missing_files", "")))
    report = {
        "partition": "training",
        "dataset_directory_name": dataset_root.name,
        "trip_count": len(trip_dirs),
        "trips_with_errors": [row["trip_id"] for row in records if row.get("error")],
        "required_file_availability": {suffix: availability[suffix] for suffix in SUFFIXES},
        "trip_summaries": aggregates,
        "notes": [
            "Only the supplied training directory was read.",
            "One-second timestamp intervals are counted within a 0.05-second tolerance.",
            "GPS step distances use the haversine formula on a spherical Earth.",
            "Road connectivity is reported as weak/undirected connectivity; arc direction is retained for route continuity checks.",
            "Route-to-track correspondence is not assumed or synthesized.",
            "A route continuity mismatch is a diagnostic, not proof that a label is invalid.",
        ],
    }
    with (output_dir / "training_eda_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Path to the declared partition directory, e.g. train/")
    parser.add_argument("--output", type=Path, default=Path("reports/eda"), help="Output directory (default: reports/eda)")
    args = parser.parse_args()
    report = run(args.dataset, args.output)
    print(json.dumps({"trip_count": report["trip_count"], "summary": str(args.output / 'training_eda_summary.json'),
                      "inventory": str(args.output / 'training_trip_inventory.csv')}, indent=2))


if __name__ == "__main__":
    main()
