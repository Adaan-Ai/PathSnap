"""Generate an actual four-layer route map from one permitted dev trip."""
from pathlib import Path
import sys
import numpy as np
import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap.data import load_trip
from pathsnap.service import PathSnapApplication

trip_id = "00000000"
folder = ROOT / "student_test" / trip_id
trip = load_trip(folder, include_route_labels=True)
app = PathSnapApplication(ROOT / "student_test", ROOT / "models/pathsnap_rf.joblib", ROOT / "ui")
response = app.predict(trip_id)
predicted = [row["arc_id"] for row in response["predictions"] if row["arc_id"] is not None]
predicted_dedup = [arc for i, arc in enumerate(predicted) if i == 0 or arc != predicted[i-1]]
reference = list(trip.route_arc_ids or ())
nodes = list(trip.nodes)
network = [[nodes[a.from_node], nodes[a.to_node]] for a in trip.arcs]
gps = [(p.longitude, p.latitude) for p in trip.observations]

def lines(arc_ids):
    return [[nodes[trip.arcs[i].from_node], nodes[trip.arcs[i].to_node]] for i in arc_ids]

# Render a compact static view for the PDF using all trip graph arcs and the
# actual GPS/reference/predicted geometries. Coordinates stay local to the repo.
net_xy = [[(p[0], p[1]) for p in line] for line in network]
fig, ax = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
ax.add_collection(LineCollection(net_xy, colors="#8c98a5", linewidths=.28, alpha=.36, label="Road network"))
ax.add_collection(LineCollection(lines(reference), colors="#228b5a", linewidths=1.0, alpha=.80, label="Reference route (weak label)"))
ax.add_collection(LineCollection(lines(predicted_dedup), colors="#7b2cbf", linewidths=1.0, alpha=.85, label="Predicted route (RF + Viterbi)"))
ax.plot([p[0] for p in gps], [p[1] for p in gps], color="#d9468f", linewidth=.65, alpha=.75, label="Raw GPS trajectory")
center_lat = float(np.mean([p[1] for p in gps]))
ax.set_aspect(1 / max(.1, np.cos(np.deg2rad(center_lat))))
ax.autoscale()
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title(f"Student-development trip {trip_id}: GPS, road graph, reference and decoded route")
ax.grid(alpha=.15)
ax.legend(loc="best", fontsize=8)
out = ROOT / "reports/figures"
out.mkdir(parents=True, exist_ok=True)
png = out / f"student_development_trip_{trip_id}_routes.png"
fig.savefig(png, dpi=180)
plt.close(fig)

# Interactive Folium companion with independent layer controls.
latlon = lambda seq: [(lat, lon) for lon, lat in seq]
center = [center_lat, float(np.mean([p[0] for p in gps]))]
m = folium.Map(location=center, zoom_start=12, control_scale=True, tiles="OpenStreetMap")
fg = folium.FeatureGroup(name=f"Road network ({len(network):,} directed arcs)", show=True)
for edge in network:
    folium.PolyLine(latlon(edge), color="#8c98a5", weight=1, opacity=.35).add_to(fg)
fg.add_to(m)
for name, ids, color in [("Reference route (weak dev label)", reference, "#228b5a"),
                         ("Predicted RF + Viterbi route", predicted_dedup, "#7b2cbf")]:
    group = folium.FeatureGroup(name=name, show=True)
    for edge in lines(ids):
        folium.PolyLine(latlon(edge), color=color, weight=4, opacity=.85).add_to(group)
    group.add_to(m)
gps_group = folium.FeatureGroup(name=f"Raw GPS observations ({len(gps):,})", show=True)
folium.PolyLine(latlon(gps), color="#d9468f", weight=2, opacity=.7).add_to(gps_group)
for lon, lat in gps[::max(1, len(gps)//300)]:
    folium.CircleMarker((lat, lon), radius=2, color="#d9468f", fill=True, fill_opacity=.7).add_to(gps_group)
gps_group.add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
m.fit_bounds([(min(p[1] for p in gps), min(p[0] for p in gps)),
              (max(p[1] for p in gps), max(p[0] for p in gps))])
html_path = out / f"student_development_trip_{trip_id}_routes.html"
m.save(html_path)
print(f"trip={trip_id}; observations={len(gps)}; network_arcs={len(network)}; reference_arcs={len(reference)}; predicted_unique_arcs={len(predicted_dedup)}")
print(f"png={png}; folium={html_path}")
