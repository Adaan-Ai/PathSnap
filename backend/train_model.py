import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.train_compare import _load_sampled_candidates, _estimators

DATASET = Path.home() / "Desktop/ML_hackathon/Student folder/train"
OUTPUT = ROOT / "backend/artifacts/random_forest.joblib"

print("Preparing training candidates...", flush=True)

(
    X, y, groups, obs_keys, sample_ids, group_names,
    expected_validation, validation_misses, total_aligned,
    sampled_aligned_targets, sampled_candidate_misses,
    mean_alignment_coverage
) = _load_sampled_candidates(DATASET, radius_m=50.0, max_candidates=8)

train_mask = sample_ids == 0
X_train = X[train_mask]
y_train = y[train_mask]

print(f"Training rows: {len(X_train)}", flush=True)
print(f"Positive examples: {int(y_train.sum())}", flush=True)

model = _estimators()["random_forest"]
print("Training Random Forest...", flush=True)
model.fit(X_train, y_train)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
joblib.dump({
    "model": model,
    "feature_count": X_train.shape[1],
    "radius_m": 50.0,
    "max_candidates": 8,
    "model_name": "random_forest",
}, OUTPUT)

print(f"Saved model: {OUTPUT}")
print(f"Feature count: {X_train.shape[1]}")
print("Training complete.")
