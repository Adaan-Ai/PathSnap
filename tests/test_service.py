from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import numpy as np

from pathsnap.service import PathSnapApplication, create_handler


class PositiveEstimator:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile(np.array([[0.1, 0.9]]), (len(features), 1))


class ServiceTests(unittest.TestCase):
    def test_api_predicts_without_route_file_and_serves_ui(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data_root = root / "data"
            trip = data_root / "demo-trip"
            trip.mkdir(parents=True)
            (trip / "demo-trip.track").write_text(
                "8.0 49.0 1\n8.00005 49.0 2\n8.0001 49.0 3\n", encoding="utf-8")
            (trip / "demo-trip.nodes").write_text("8.0 49.0\n8.0001 49.0\n", encoding="utf-8")
            (trip / "demo-trip.arcs").write_text("0 1\n", encoding="utf-8")
            model_file = root / "model.joblib"
            joblib.dump({"estimator": PositiveEstimator(),
                         "candidate_settings": {"radius_m": 50.0, "max_candidates": 8}}, model_file)
            ui = root / "ui"
            ui.mkdir()
            (ui / "index.html").write_text("<main>PathSnap UI</main>", encoding="utf-8")
            (ui / "app.js").write_text("", encoding="utf-8")
            (ui / "styles.css").write_text("", encoding="utf-8")
            app = PathSnapApplication(data_root, model_file, ui)
            server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f"http://127.0.0.1:{server.server_port}"
            try:
                with urlopen(base + "/health") as response:
                    self.assertEqual(json.loads(response.read()), {"status": "ready"})
                with urlopen(base + "/") as response:
                    self.assertIn(b"PathSnap UI", response.read())
                request = Request(base + "/predict", data=b'{"trip_id":"demo-trip"}',
                                  headers={"Content-Type": "application/json"}, method="POST")
                with urlopen(request) as response:
                    result = json.loads(response.read())
                self.assertEqual(result["trip_id"], "demo-trip")
                self.assertEqual(result["matched_observations"], 3)
                self.assertEqual(len(result["predictions"]), 3)
                self.assertNotIn("route", result)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
