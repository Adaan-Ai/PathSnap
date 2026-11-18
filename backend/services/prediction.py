import os
from pathlib import Path

import joblib
import numpy as np

from pathsnap.data import load_trip
from pathsnap.candidates import CandidateGenerator
from experiments.train_compare import _feature_row

ROOT = Path(__file__).resolve().parents[2]

DATASET = Path(os.getenv(
    "PATHSNAP_DATASET",
    str(Path.home() / "Desktop/ML_hackathon/Student folder/train")
))

MODEL_PATH = ROOT / "backend/artifacts/random_forest.joblib"

_bundle = joblib.load(MODEL_PATH)
_model = _bundle["model"]
_model.n_jobs = 1
if _bundle["feature_count"] != 13:
    raise RuntimeError("Model feature count mismatch")

_generator = CandidateGenerator(
    radius_m=_bundle["radius_m"],
    max_candidates=_bundle["max_candidates"]
)


def available_trips():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    return sorted(
        p.name for p in DATASET.iterdir()
        if p.is_dir()
        and (p / f"{p.name}.track").is_file()
        and (p / f"{p.name}.nodes").is_file()
        and (p / f"{p.name}.arcs").is_file()
    )



def predict_trip(trip_id: str):
    trip_dir = DATASET / trip_id

    if trip_id not in available_trips():
        raise ValueError(f"Unknown trip_id: {trip_id}")

    trip = load_trip(trip_dir, include_route_labels=False)
    generated = _generator.generate(trip)

    out_degree = {}
    in_degree = {}

    for arc in trip.arcs:
        out_degree[arc.from_node] = out_degree.get(arc.from_node, 0) + 1
        in_degree[arc.to_node] = in_degree.get(arc.to_node, 0) + 1

    all_features = []
    candidate_groups = []

    # Build features for every candidate before prediction.
    for observation_result in generated:
        candidates = observation_result.candidates
        start = len(all_features)

        for candidate in candidates:
            all_features.append(
                _feature_row(
                    trip,
                    observation_result.observation_index,
                    candidate,
                    out_degree,
                    in_degree
                )
            )

        candidate_groups.append((start, len(all_features)))

    # Predict all candidates in one batch.
    if all_features:
        probabilities = _model.predict_proba(
            np.asarray(all_features, dtype=np.float32)
        )[:, 1]
    else:
        probabilities = np.asarray([])

    results = []

    for observation_result, (start, end) in zip(
        generated, candidate_groups
    ):
        candidates = observation_result.candidates

        if not candidates:
            results.append({
                "observation_index": observation_result.observation_index,
                "timestamp": observation_result.timestamp_s,
                "predicted_arc_id": None,
                "confidence": None,
                "status": observation_result.status
            })
            continue

        local_probabilities = probabilities[start:end]
        best_index = int(np.argmax(local_probabilities))
        best_candidate = candidates[best_index]

        results.append({
            "observation_index": observation_result.observation_index,
            "timestamp": observation_result.timestamp_s,
            "predicted_arc_id": best_candidate.arc_id,
            "confidence": round(
                float(local_probabilities[best_index]), 4
            ),
            "status": "ok"
        })

    matched = sum(
        result["predicted_arc_id"] is not None
        for result in results
    )

    return {
        "trip_id": trip_id,
        "model": "random_forest",
        "decoder": "classifier_only",
        "total_observations": len(results),
        "matched_observations": matched,
        "unmatched_observations": len(results) - matched,
        "predictions": results
    }
