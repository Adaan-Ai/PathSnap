"""Fit the selected Random Forest on weak labels from the train partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train_compare import SEED, _load_sampled_candidates


def build(dataset: Path, output: Path) -> dict[str, object]:
    x, y, _groups, _keys, sample_ids, *_ = _load_sampled_candidates(dataset, 50.0, 8)
    rows = np.flatnonzero(sample_ids == 0)
    estimator = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=2,
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    estimator.fit(x[rows], y[rows])
    artifact = {
        "estimator": estimator,
        "feature_count": int(x.shape[1]),
        "feature_order": [
            "perpendicular_distance_m", "projection_fraction", "log_segment_length_m",
            "road_bearing_sin", "road_bearing_cos", "movement_bearing_sin",
            "movement_bearing_cos", "direction_difference_scaled", "log_gps_step_distance_m",
            "log_delta_t_s", "log_gps_speed_mps", "log_from_node_degree", "log_to_node_degree",
        ],
        "candidate_settings": {"radius_m": 50.0, "max_candidates": 8},
        "training_partition": "train",
        "training_trip_count": int(len(np.unique(_groups[rows]))),
        "training_candidate_count": int(len(rows)),
        "target_semantics": "monotonic route-sequence alignment weak labels",
        "random_seed": SEED,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output, compress=3)
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(json.dumps({key: value for key, value in artifact.items() if key != "estimator"}, indent=2) + "\n",
                             encoding="utf-8")
    return {key: value for key, value in artifact.items() if key != "estimator"} | {"artifact": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/pathsnap_rf.joblib"))
    args = parser.parse_args()
    print(json.dumps(build(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
