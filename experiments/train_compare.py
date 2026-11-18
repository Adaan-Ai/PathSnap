"""Compare five candidate scorers with trip-grouped cross-validation.

Targets come from monotonic route-to-track weak alignment. They are not
independently verified per-observation ground truth. This experiment uses a
fixed, seeded observation sample per trip so kernel and neighbour models remain
tractable; it reports candidate miss rate and coverage alongside scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from pathsnap.alignment import align_route_to_track
from pathsnap.candidates import CandidateGenerator
from pathsnap.data import TripData, load_trip
from pathsnap.eda import haversine_m

SEED = 2026
TRAIN_OBSERVATIONS_PER_TRIP = 100
VALIDATION_OBSERVATIONS_PER_TRIP = 50
MODEL_FAMILIES = {
    "logistic_regression": "linear probabilistic classifier",
    "knn": "instance-based nearest neighbours",
    "random_forest": "bagged decision-tree ensemble",
    "hist_gradient_boosting": "boosted decision-tree ensemble",
    "rbf_svm": "RBF kernel classifier",
}


def _feature_row(trip: TripData, observation_index: int, candidate: object,
                 out_degree: dict[int, int], in_degree: dict[int, int]) -> list[float]:
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
    return [
        candidate.perpendicular_distance_m,
        candidate.projection_fraction,
        math.log1p(candidate.segment_length_m),
        math.sin(angle), math.cos(angle),
        math.sin(movement_angle) if movement_angle is not None else 0.0,
        math.cos(movement_angle) if movement_angle is not None else 0.0,
        (candidate.direction_difference_deg or 0.0) / 180.0,
        math.log1p(max(0.0, step_distance)),
        math.log1p(max(0.0, delta_t)),
        math.log1p(max(0.0, speed)),
        math.log1p(out_degree.get(arc.from_node, 0)),
        math.log1p(in_degree.get(arc.to_node, 0)),
    ]


def _load_sampled_candidates(dataset: Path, radius_m: float, max_candidates: int):
    generator = CandidateGenerator(radius_m=radius_m, max_candidates=max_candidates)
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    group_chunks: list[np.ndarray] = []
    observation_chunks: list[np.ndarray] = []
    sample_chunks: list[np.ndarray] = []  # 0=train sample, 1=validation sample
    group_names: list[str] = []
    expected_validation: dict[int, int] = {}
    validation_misses: dict[int, int] = {}
    weak_alignment_coverage: list[float] = []
    sampled_candidate_misses = 0
    sampled_aligned_targets = 0
    total_aligned = 0

    trip_dirs = sorted(path for path in dataset.iterdir() if path.is_dir())
    for group_id, directory in enumerate(trip_dirs):
        trip = load_trip(directory, include_route_labels=True)
        alignment = align_route_to_track(trip, max_alignment_distance_m=250.0)
        eligible = [index for index, target in enumerate(alignment.arc_ids_by_observation) if target is not None]
        weak_alignment_coverage.append(len(eligible) / max(1, len(trip.observations)))
        total_aligned += len(eligible)

        rng = np.random.default_rng(SEED + group_id)
        shuffled = np.asarray(eligible, dtype=np.int32)
        rng.shuffle(shuffled)
        train_indices = shuffled[:TRAIN_OBSERVATIONS_PER_TRIP].tolist()
        validation_indices = shuffled[TRAIN_OBSERVATIONS_PER_TRIP:
                                      TRAIN_OBSERVATIONS_PER_TRIP + VALIDATION_OBSERVATIONS_PER_TRIP].tolist()
        selected = train_indices + validation_indices
        validation_set = set(validation_indices)
        sampled_aligned_targets += len(selected)
        candidate_rows = generator.generate(trip, selected)

        out_degree: dict[int, int] = {}
        in_degree: dict[int, int] = {}
        for arc in trip.arcs:
            out_degree[arc.from_node] = out_degree.get(arc.from_node, 0) + 1
            in_degree[arc.to_node] = in_degree.get(arc.to_node, 0) + 1

        feature_rows: list[list[float]] = []
        labels: list[int] = []
        observation_keys: list[int] = []
        sample_ids: list[int] = []
        misses_validation = 0
        for result in candidate_rows:
            observation_index = result.observation_index
            target = alignment.arc_ids_by_observation[observation_index]
            if target is None:
                continue
            is_validation = observation_index in validation_set
            if is_validation:
                expected_validation[group_id] = expected_validation.get(group_id, 0) + 1
            candidate_ids = {candidate.arc_id for candidate in result.candidates}
            if target not in candidate_ids:
                sampled_candidate_misses += 1
                if is_validation:
                    misses_validation += 1
                continue
            obs_key = (group_id << 32) + observation_index
            for candidate in result.candidates:
                feature_rows.append(_feature_row(trip, observation_index, candidate, out_degree, in_degree))
                labels.append(int(candidate.arc_id == target))
                observation_keys.append(obs_key)
                sample_ids.append(1 if is_validation else 0)

        validation_misses[group_id] = misses_validation
        if feature_rows:
            feature_chunks.append(np.asarray(feature_rows, dtype=np.float32))
            label_chunks.append(np.asarray(labels, dtype=np.int8))
            group_chunks.append(np.full(len(labels), group_id, dtype=np.int16))
            observation_chunks.append(np.asarray(observation_keys, dtype=np.int64))
            sample_chunks.append(np.asarray(sample_ids, dtype=np.int8))
        group_names.append(trip.trip_id)
        if (group_id + 1) % 10 == 0:
            print(f"prepared candidate samples for {group_id + 1}/{len(trip_dirs)} training trips", flush=True)

    if not feature_chunks:
        raise RuntimeError("No aligned route labels intersected the prediction candidate sets")
    return (
        np.concatenate(feature_chunks), np.concatenate(label_chunks),
        np.concatenate(group_chunks), np.concatenate(observation_chunks),
        np.concatenate(sample_chunks), group_names, expected_validation,
        validation_misses, total_aligned, sampled_aligned_targets, sampled_candidate_misses,
        float(np.mean(weak_alignment_coverage)),
    )


def _estimators() -> dict[str, object]:
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED)),
        "knn": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance", n_jobs=-1)),
        "random_forest": RandomForestClassifier(
            n_estimators=120, max_depth=14, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=SEED),
        "rbf_svm": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, class_weight="balanced",
                                  cache_size=512, random_state=SEED)),
    }


def _scores(estimator: object, features: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(features)[:, 1]
        return probabilities, probabilities
    decisions = estimator.decision_function(features)
    return np.asarray(decisions), None


def _top1_accuracy(y: np.ndarray, scores: np.ndarray, obs_keys: np.ndarray) -> int:
    order = np.argsort(obs_keys, kind="stable")
    sorted_keys = obs_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(order)]
    correct = 0
    for start, end in zip(starts, ends):
        indices = order[start:end]
        correct += int(y[indices[np.argmax(scores[indices])]] == 1)
    return correct


def _plot_model_comparison(summary: list[dict[str, object]], output_path: Path) -> None:
    labels = [str(row["model"]).replace("_", " ") for row in summary]
    x_positions = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, metric, color, label in ((-0.18, "top1_accuracy_all_aligned", "#2a6fbb", "Top-1, all aligned samples"),
                                         (0.18, "candidate_macro_f1", "#e07a36", "Candidate macro F1")):
        means = [row[f"{metric}_mean"] for row in summary]
        spreads = [row[f"{metric}_std"] for row in summary]
        axis.bar(x_positions + offset, means, width=0.34, yerr=spreads, capsize=3,
                 color=color, label=label)
    axis.set_xticks(x_positions, labels, rotation=15, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("5-fold mean score (error bars: fold standard deviation)")
    axis.set_title("PathSnap candidate classifiers: training-trip grouped CV")
    axis.legend()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def compare_models(dataset: Path, output_dir: Path, *, radius_m: float = 50.0,
                   max_candidates: int = 8, folds: int = 5) -> dict[str, object]:
    (x, y, groups, obs_keys, sample_ids, group_names, expected_validation,
     validation_misses, total_aligned, sampled_aligned_targets, sampled_candidate_misses,
     mean_alignment_coverage) = _load_sampled_candidates(
         dataset, radius_m, max_candidates)
    splitter = GroupKFold(n_splits=folds)
    fold_records: list[dict[str, object]] = []
    for fold_number, (group_train, group_test) in enumerate(
            splitter.split(np.arange(len(group_names)), groups=np.arange(len(group_names))), start=1):
        train_rows = np.flatnonzero(np.isin(groups, group_train) & (sample_ids == 0))
        test_rows = np.flatnonzero(np.isin(groups, group_test) & (sample_ids == 1))
        if len(np.unique(y[train_rows])) < 2 or len(test_rows) == 0:
            raise RuntimeError(f"Fold {fold_number} lacks both training classes or held-out rows")
        y_train = y[train_rows]
        y_test = y[test_rows]
        negative_count, positive_count = np.bincount(y_train, minlength=2)
        class_weights = np.where(y_train == 1, len(y_train) / max(2 * positive_count, 1),
                                 len(y_train) / max(2 * negative_count, 1))
        test_group_ids = np.unique(groups[test_rows])
        expected_obs = sum(expected_validation.get(int(g), 0) for g in test_group_ids)
        missed_obs = sum(validation_misses.get(int(g), 0) for g in test_group_ids)
        candidate_covered_obs = len(np.unique(obs_keys[test_rows]))

        for model_name, estimator in _estimators().items():
            started = time.perf_counter()
            if model_name == "hist_gradient_boosting":
                estimator.fit(x[train_rows], y_train, sample_weight=class_weights)
            else:
                estimator.fit(x[train_rows], y_train)
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            scores, probabilities = _scores(estimator, x[test_rows])
            inference_seconds = time.perf_counter() - started
            predicted = scores >= (0.5 if probabilities is not None else 0.0)
            top1_correct = _top1_accuracy(y_test, scores, obs_keys[test_rows])
            fold_records.append({
                "model": model_name,
                "fold": fold_number,
                "candidate_macro_f1": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
                "candidate_accuracy": float(accuracy_score(y_test, predicted)),
                "top1_accuracy_candidate_covered": top1_correct / max(1, candidate_covered_obs),
                "top1_accuracy_all_aligned": top1_correct / max(1, expected_obs),
                "candidate_miss_rate": missed_obs / max(1, expected_obs),
                "log_loss": float(log_loss(y_test, probabilities, labels=[0, 1])) if probabilities is not None else None,
                "train_candidates": int(len(train_rows)),
                "validation_candidates": int(len(test_rows)),
                "validation_observations_aligned": int(expected_obs),
                "validation_observations_with_candidate_label": int(candidate_covered_obs),
                "train_seconds": fit_seconds,
                "inference_seconds": inference_seconds,
            })
            print(f"fold {fold_number}/{folds}: {model_name} fit in {fit_seconds:.2f}s, "
                  f"top-1={top1_correct / max(1, candidate_covered_obs):.3f}", flush=True)

    metric_names = ("candidate_macro_f1", "candidate_accuracy", "top1_accuracy_candidate_covered",
                    "top1_accuracy_all_aligned", "candidate_miss_rate", "log_loss",
                    "train_seconds", "inference_seconds")
    summary: list[dict[str, object]] = []
    for model_name in [record["model"] for record in fold_records[:5]]:
        model_folds = [record for record in fold_records if record["model"] == model_name]
        row: dict[str, object] = {"model": model_name, "algorithm_family": MODEL_FAMILIES[model_name],
                                  "folds": len(model_folds)}
        for metric in metric_names:
            values = [float(record[metric]) for record in model_folds if record[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        summary.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "model_comparison_folds.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fold_records[0]))
        writer.writeheader()
        writer.writerows(fold_records)
    report = {
        "partition": "training_grouped_cross_validation",
        "trip_count": len(group_names),
        "folds": folds,
        "seed": SEED,
        "candidate_settings": {"radius_m": radius_m, "max_candidates": max_candidates},
        "sample_design": {"train_observations_per_trip_max": TRAIN_OBSERVATIONS_PER_TRIP,
                          "validation_observations_per_trip_max": VALIDATION_OBSERVATIONS_PER_TRIP,
                          "evaluation_groups": "held-out trip IDs via GroupKFold",
                          "selection_seed": SEED},
        "target_semantics": "weak route-sequence-to-track alignment; not independent per-observation ground truth",
        "alignment_max_distance_m": 250.0,
        "mean_trip_alignment_coverage": mean_alignment_coverage,
        "aligned_training_observations": total_aligned,
        "sampled_aligned_training_observations": sampled_aligned_targets,
        "sampled_candidate_misses": sampled_candidate_misses,
        "sampled_candidate_miss_rate": sampled_candidate_misses / max(1, sampled_aligned_targets),
        "full_partition_aligned_observations": total_aligned,
        "models": summary,
        "fold_results": fold_records,
        "limitations": [
            "Cross-validation metrics are conditional on noisy route-derived weak labels.",
            "The validation metric uses a seeded 50-observation sample per held-out trip.",
            "Route-level continuity is not measured in this candidate-classifier experiment.",
            "The separate student-development partition was not supplied and is not evaluated here.",
            "Evaluator-testing data and labels are not read by this experiment.",
        ],
    }
    with (output_dir / "model_comparison_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    _plot_model_comparison(summary, output_dir / "model_comparison.png")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    report = compare_models(args.dataset, args.output, radius_m=args.radius_m,
                            max_candidates=args.max_candidates, folds=args.folds)
    print(json.dumps({"trips": report["trip_count"], "folds": report["folds"],
                      "sampled_candidate_miss_rate": report["sampled_candidate_miss_rate"],
                      "results": str(args.output / "model_comparison_summary.json")}, indent=2))


if __name__ == "__main__":
    main()
