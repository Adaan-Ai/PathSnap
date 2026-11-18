"""Exercise the trained model through the actual HTTP API on one trip."""
from __future__ import annotations

import argparse
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

from pathsnap.service import PathSnapApplication, create_handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True,
                        help="folder containing the selected development trip")
    parser.add_argument("--model", type=Path, default=Path("models/pathsnap_rf.joblib"))
    parser.add_argument("--trip-id", required=True)
    args = parser.parse_args()
    ui_root = Path(__file__).resolve().parents[1] / "ui"
    app = PathSnapApplication(args.dataset, args.model, ui_root)
    server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base + "/health", timeout=10) as response:
            health = json.loads(response.read())
        with urlopen(base + "/", timeout=10) as response:
            assert b"Route reconstruction" in response.read()
        request = Request(base + "/predict", data=json.dumps({"trip_id": args.trip_id}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
        if health.get("status") != "ready" or not result.get("predictions"):
            raise RuntimeError("API smoke test returned an incomplete response")
        if any("route" in row for row in result["predictions"]):
            raise RuntimeError("Prediction response unexpectedly contains route labels")
        print(json.dumps({"health": health, "trip_id": result["trip_id"],
                          "observations": result["observations"],
                          "matched_observations": result["matched_observations"],
                          "candidate_coverage": result["candidate_coverage"],
                          "route_continuity": result["route_continuity"],
                          "prediction_rows": len(result["predictions"])}, indent=2))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
