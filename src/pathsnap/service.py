"""Local HTTP API and single-page UI for server-side map matching."""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np

from pathsnap.candidates import CandidateGenerator
from pathsnap.data import load_trip
from pathsnap.decoder import viterbi_decode


def _feature_row(trip: object, observation_index: int, candidate: object,
                 out_degree: dict[int, int], in_degree: dict[int, int]) -> list[float]:
    import math
    from pathsnap.eda import haversine_m

    angle = math.radians(candidate.road_bearing_deg)
    movement = candidate.movement_bearing_deg
    movement_angle = math.radians(movement) if movement is not None else None
    if observation_index > 0:
        current = trip.observations[observation_index]
        previous = trip.observations[observation_index - 1]
        delta_t = current.timestamp_s - previous.timestamp_s
        step_distance = haversine_m(previous.longitude, previous.latitude,
                                     current.longitude, current.latitude)
    else:
        delta_t, step_distance = 0.0, 0.0
    speed = step_distance / delta_t if delta_t > 0 else 0.0
    arc = trip.arcs[candidate.arc_id]
    return [candidate.perpendicular_distance_m, candidate.projection_fraction,
            math.log1p(candidate.segment_length_m), math.sin(angle), math.cos(angle),
            math.sin(movement_angle) if movement_angle is not None else 0.0,
            math.cos(movement_angle) if movement_angle is not None else 0.0,
            (candidate.direction_difference_deg or 0.0) / 180.0,
            math.log1p(max(0.0, step_distance)), math.log1p(max(0.0, delta_t)),
            math.log1p(max(0.0, speed)), math.log1p(out_degree.get(arc.from_node, 0)),
            math.log1p(in_degree.get(arc.to_node, 0))]


class PathSnapApplication:
    def __init__(self, dataset_root: Path, model_path: Path, ui_root: Path) -> None:
        self.dataset_root = dataset_root.resolve()
        self.artifact = joblib.load(model_path)
        self.ui_root = ui_root.resolve()
        settings = self.artifact["candidate_settings"]
        self.generator = CandidateGenerator(**{
            "radius_m": float(settings["radius_m"]),
            "max_candidates": int(settings["max_candidates"]),
        })

    def predict(self, trip_id: str) -> dict[str, object]:
        if not trip_id or trip_id in {".", ".."} or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in trip_id):
            raise ValueError("trip_id must contain only letters, numbers, underscores, or hyphens")
        directory = (self.dataset_root / trip_id).resolve()
        if directory.parent != self.dataset_root or not directory.is_dir():
            raise FileNotFoundError("Trip not found in the configured data directory")
        # Route labels are intentionally never read on the inference path.
        trip = load_trip(directory, include_route_labels=False)
        generated = self.generator.generate(trip)
        outgoing: dict[int, int] = {}
        incoming: dict[int, int] = {}
        for arc in trip.arcs:
            outgoing[arc.from_node] = outgoing.get(arc.from_node, 0) + 1
            incoming[arc.to_node] = incoming.get(arc.to_node, 0) + 1
        feature_rows: list[list[float]] = []
        owners: list[int] = []
        for position, row in enumerate(generated):
            for candidate in row.candidates:
                feature_rows.append(_feature_row(trip, row.observation_index, candidate,
                                                 outgoing, incoming))
                owners.append(position)
        probabilities_by_position = [[] for _ in generated]
        if feature_rows:
            scores = self.artifact["estimator"].predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1]
            for owner, score in zip(owners, scores):
                probabilities_by_position[owner].append(float(score))
        probabilities = [np.asarray(values, dtype=float) for values in probabilities_by_position]
        decoded = viterbi_decode(trip, generated, probabilities)
        return {
            "trip_id": trip.trip_id,
            "observations": len(trip.observations),
            "matched_observations": sum(arc is not None for arc in decoded.arc_ids),
            "candidate_coverage": sum(bool(row.candidates) for row in generated) / max(1, len(generated)),
            "route_continuity": decoded.connected_transition_rate,
            "transition_count": decoded.transition_count,
            "predictions": [
                {"observation_index": row.observation_index,
                 "timestamp_s": row.timestamp_s,
                 "arc_id": decoded.arc_ids[position],
                 "candidate_arc_ids": [candidate.arc_id for candidate in row.candidates],
                 "candidate_distances_m": [round(candidate.perpendicular_distance_m, 2) for candidate in row.candidates]}
                for position, row in enumerate(generated)
            ],
        }


def create_handler(application: PathSnapApplication):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/health":
                self._send(200, b'{"status":"ready"}', "application/json; charset=utf-8")
                return
            if route == "/":
                path = application.ui_root / "index.html"
                self._send(200, path.read_bytes(), "text/html; charset=utf-8")
                return
            if route in {"/app.js", "/styles.css"}:
                path = application.ui_root / route.lstrip("/")
                if path.is_file() and path.parent == application.ui_root:
                    self._send(200, path.read_bytes(), "text/javascript; charset=utf-8" if route.endswith(".js") else "text/css; charset=utf-8")
                    return
            self._send(404, b'{"error":"not found"}', "application/json; charset=utf-8")

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/predict":
                self._send(404, b'{"error":"not found"}', "application/json; charset=utf-8")
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size <= 0 or size > 2048:
                    raise ValueError("Request body must be between 1 and 2048 bytes")
                payload = json.loads(self.rfile.read(size))
                trip_id = payload.get("trip_id") if isinstance(payload, dict) else None
                if not isinstance(trip_id, str):
                    raise ValueError("Request JSON must contain a string trip_id")
                result = application.predict(trip_id)
                body = json.dumps(result, separators=(",", ":")).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            except FileNotFoundError as error:
                self._send(404, json.dumps({"error": str(error)}).encode(), "application/json; charset=utf-8")
            except (ValueError, json.JSONDecodeError) as error:
                self._send(400, json.dumps({"error": str(error)}).encode(), "application/json; charset=utf-8")
            except Exception:
                self._send(500, b'{"error":"prediction failed"}', "application/json; charset=utf-8")

        def log_message(self, format: str, *args: object) -> None:
            print("PathSnap API: " + format % args)

    return Handler


def main() -> None:
    data_root = Path(os.environ.get("PATHSNAP_DATA_ROOT", "data/student_test"))
    model_path = Path(os.environ.get("PATHSNAP_MODEL", "models/pathsnap_rf.joblib"))
    host = os.environ.get("PATHSNAP_HOST", "0.0.0.0")
    port = int(os.environ.get("PATHSNAP_PORT", "8000"))
    application = PathSnapApplication(data_root, model_path, Path(__file__).resolve().parents[2] / "ui")
    server = ThreadingHTTPServer((host, port), create_handler(application))
    print(f"PathSnap ready at http://{host}:{port}/; data root={application.dataset_root}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
