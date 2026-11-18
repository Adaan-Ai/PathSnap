"""Create training-only summary charts and one Folium network/track map."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import folium
import matplotlib
from matplotlib.collections import LineCollection

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathsnap.eda import read_rows


def create_visualizations(dataset: Path, inventory_csv: Path, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    with inventory_csv.open("r", newline="", encoding="utf-8") as stream:
        trips = list(csv.DictReader(stream))
    fields = ("track_rows", "node_rows", "arc_rows", "route_rows")
    labels = ("GPS observations", "Road nodes", "Road arcs", "Reference route arcs")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, field, label in zip(axes.flat, fields, labels):
        values = [int(row[field]) for row in trips if row.get(field)]
        axis.hist(values, bins=16, color="#3569a8", edgecolor="white")
        axis.set_title(label)
        axis.set_xlabel("Count per trip")
        axis.set_ylabel("Trips")
    fig.suptitle("PathSnap training partition: per-trip size distributions")
    fig.savefig(output / "training_size_distributions.png", dpi=160)
    plt.close(fig)

    # Select the smallest complete network with a usable GPS trajectory so the
    # interactive map remains practical to open while preserving real geometry.
    complete = [row for row in trips if not row.get("missing_files") and row.get("track_rows")]
    sample = min(complete, key=lambda row: int(row["arc_rows"]))
    trip_id = sample["trip_id"]
    folder = dataset / trip_id
    gps, _ = read_rows(folder / f"{trip_id}.track", 3)
    nodes, _ = read_rows(folder / f"{trip_id}.nodes", 2)
    arcs, _ = read_rows(folder / f"{trip_id}.arcs", 2)
    valid_gps = [(lon, lat) for lon, lat, _ in gps if -180 <= lon <= 180 and -90 <= lat <= 90]
    valid_nodes = [(lon, lat) for lon, lat in nodes if -180 <= lon <= 180 and -90 <= lat <= 90]
    if not valid_gps:
        raise ValueError(f"Trip {trip_id} has no valid GPS positions to map")

    center = [sum(point[1] for point in valid_gps) / len(valid_gps),
              sum(point[0] for point in valid_gps) / len(valid_gps)]
    # Local equirectangular coordinates in kilometres are used for the static
    # figure so its axes express physical distance, not raw degrees.
    meters_per_degree_lat = math.pi * 6_371_008.8 / 180
    meters_per_degree_lon = meters_per_degree_lat * math.cos(math.radians(center[0]))
    road_segments: list[list[tuple[float, float]]] = []
    for start_raw, end_raw in arcs:
        start, end = int(start_raw), int(end_raw)
        if start < len(valid_nodes) and end < len(valid_nodes):
            a_lon, a_lat = valid_nodes[start]
            b_lon, b_lat = valid_nodes[end]
            road_segments.append([
                ((a_lon - center[1]) * meters_per_degree_lon / 1000, (a_lat - center[0]) * meters_per_degree_lat / 1000),
                ((b_lon - center[1]) * meters_per_degree_lon / 1000, (b_lat - center[0]) * meters_per_degree_lat / 1000),
            ])
    fig, axis = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axis.add_collection(LineCollection(road_segments, colors="#8290a3", linewidths=0.45, alpha=0.45,
                                       label="Road network"))
    gps_x = [(lon - center[1]) * meters_per_degree_lon / 1000 for lon, _ in valid_gps]
    gps_y = [(lat - center[0]) * meters_per_degree_lat / 1000 for _, lat in valid_gps]
    axis.plot(gps_x, gps_y, color="#e4572e", linewidth=1.4, label="GPS observations")
    axis.scatter(gps_x[0], gps_y[0], color="#27864b", s=24, zorder=3, label="Start")
    axis.scatter(gps_x[-1], gps_y[-1], color="#b3222a", s=24, zorder=3, label="End")
    axis.autoscale()
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xlabel("East-west distance from trip center (km)")
    axis.set_ylabel("North-south distance from trip center (km)")
    axis.set_title(f"Training trip {trip_id}: GPS observations and road network")
    axis.legend(loc="best")
    fig.savefig(output / f"training_trip_{trip_id}_roads_and_gps.png", dpi=160)
    plt.close(fig)

    map_view = folium.Map(location=center, zoom_start=13, control_scale=True, tiles="OpenStreetMap")
    roads = folium.FeatureGroup(name=f"Road network ({len(arcs):,} arcs)", show=True)
    for start_raw, end_raw in arcs:
        start, end = int(start_raw), int(end_raw)
        if start < len(valid_nodes) and end < len(valid_nodes):
            a, b = valid_nodes[start], valid_nodes[end]
            folium.PolyLine([(a[1], a[0]), (b[1], b[0])], color="#75849b", weight=1, opacity=0.55).add_to(roads)
    roads.add_to(map_view)
    folium.PolyLine([(lat, lon) for lon, lat in valid_gps], color="#e4572e", weight=3,
                    tooltip=f"Observed GPS trajectory: trip {trip_id}").add_to(map_view)
    folium.CircleMarker((valid_gps[0][1], valid_gps[0][0]), radius=4, color="#27864b",
                        tooltip="Start observation").add_to(map_view)
    folium.CircleMarker((valid_gps[-1][1], valid_gps[-1][0]), radius=4, color="#b3222a",
                        tooltip="End observation").add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([(min(lat for _, lat in valid_gps), min(lon for lon, _ in valid_gps)),
                         (max(lat for _, lat in valid_gps), max(lon for lon, _ in valid_gps))])
    map_path = output / f"training_trip_{trip_id}_roads_and_gps.html"
    map_view.save(map_path)
    return map_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--inventory", type=Path, default=Path("reports/eda/training_trip_inventory.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/eda/visualizations"))
    args = parser.parse_args()
    map_path = create_visualizations(args.dataset, args.inventory, args.output)
    print(f"Wrote charts and interactive map to {args.output}; map: {map_path}")


if __name__ == "__main__":
    main()
