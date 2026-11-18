"""Compare candidate classifiers on the permitted student-development trips.

Student-test routes are read by the scoring/alignment path only. Prediction
features and candidate generation use a route-free TripData instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

from pathsnap.alignment import align_route_to_track
from pathsnap.candidates import CandidateGenerator
from pathsnap.data import load_trip
from train_compare import (SEED, _estimators, _feature_row, _scores,
                           _load_sampled_candidates)


def evaluate(train_root: Path, student_test_root: Path, output_dir: Path, *,
             radius_m: float = 50.0, max_candidates: int = 8) -> dict[str, object]:
    (x_train, y_train, groups, obs_keys, sample_ids, group_names, expected_validation,
     validation_misses, total_aligned, sampled_aligned, sampled_misses,
     mean_alignment_coverage) = _load_sampled_candidates(train_root, radius_m, max_candidates)
    train_rows = np.flatnonzero(sample_ids == 0)
    if len(np.unique(y_train[train_rows])) != 2:
        raise RuntimeError("Training sample does not contain both candidate-label classes")

    models = _estimators()
    training_times: dict[str, float] = {}
    for name, estimator in models.items():
        labels = y_train[train_rows]
        negative_count, positive_count = np.bincount(labels, minlength=2)
        weights = np.where(labels == 1, len(labels) / max(2 * positive_count, 1),
                           len(labels) / max(2 * negative_count, 1))
        start = time.perf_counter()
        if name == "hist_gradient_boosting":
            estimator.fit(x_train[train_rows], labels, sample_weight=weights)
        else:
            estimator.fit(x_train[train_rows], labels)
        training_times[name] = time.perf_counter() - start

    generator = CandidateGenerator(radius_m=radius_m, max_candidates=max_candidates)
    trip_dirs = sorted(path for path in student_test_root.iterdir() if path.is_dir())
    foldlike_rows: list[dict[str, object]] = []
    for directory in trip_dirs:
        # This object is the full input to inference: it contains no .route data.
        inference_trip = load_trip(directory, include_route_labels=False)
        # Read labels separately for development scoring only.
        reference_trip = load_trip(directory, include_route_labels=True)
        alignment = align_route_to_track(reference_trip, max_alignment_distance_m=250.0)
        candidate_rows = generator.generate(inference_trip)
        out_degree: dict[int, int] = {}
        in_degree: dict[int, int] = {}
        for arc in inference_trip.arcs:
            out_degree[arc.from_node] = out_degree.get(arc.from_node, 0) + 1
            in_degree[arc.to_node] = in_degree.get(arc.to_node, 0) + 1

        feature_rows: list[list[float]] = []
        labels: list[int] = []
        observation_indices: list[int] = []
        candidate_arc_ids: list[int] = []
        missed_targets = 0
        aligned_count = 0
        for generated in candidate_rows:
            index = generated.observation_index
            target = alignment.arc_ids_by_observation[index]
            if target is None:
                continue
            aligned_count += 1
            available_ids = {candidate.arc_id for candidate in generated.candidates}
            if target not in available_ids:
                missed_targets += 1
            for candidate in generated.candidates:
                feature_rows.append(_feature_row(inference_trip, index, candidate, out_degree, in_degree))
                labels.append(int(candidate.arc_id == target) if target in available_ids else -1)
                observation_indices.append(index)
                candidate_arc_ids.append(candidate.arc_id)

        matrix = np.asarray(feature_rows, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int8)
        indices_array = np.asarray(observation_indices, dtype=np.int32)
        records_by_model: list[dict[str, object]] = []
        for name, estimator in models.items():
            start = time.perf_counter()
            scores, probabilities = _scores(estimator, matrix)
            inference_seconds = time.perf_counter() - start
            valid_mask = y >= 0
            valid_y = y[valid_mask]
            valid_scores = scores[valid_mask]
            predicted = valid_scores >= (0.5 if probabilities is not None else 0.0)
            valid_probabilities = probabilities[valid_mask] if probabilities is not None else None
            best_arc_by_observation: dict[int, int] = {}
            order = np.argsort(indices_array, kind="stable")
            sorted_keys = indices_array[order]
            starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1] if len(order) else np.asarray([], dtype=int)
            ends = np.r_[starts[1:], len(order)] if len(starts) else np.asarray([], dtype=int)
            for first, last in zip(starts, ends):
                row_indices = order[first:last]
                best = row_indices[np.argmax(scores[row_indices])]
                observation_index = int(indices_array[best])
                best_arc_by_observation[observation_index] = candidate_arc_ids[best]

            correct = sum(best_arc_by_observation.get(index) == target
                          for index, target in enumerate(alignment.arc_ids_by_observation)
                          if target is not None)

            paired = 0
            connected = 0
            for index in range(len(inference_trip.observations) - 1):
                if index not in best_arc_by_observation or index + 1 not in best_arc_by_observation:
                    continue
                paired += 1
                previous_arc = inference_trip.arcs[best_arc_by_observation[index]]
                next_arc = inference_trip.arcs[best_arc_by_observation[index + 1]]
                connected += int(previous_arc.to_node == next_arc.from_node)

            records_by_model.append({
                "model": name,
                "trip_id": inference_trip.trip_id,
                "aligned_observations": aligned_count,
                "candidate_misses": missed_targets,
                "candidate_miss_rate": missed_targets / max(1, aligned_count),
                "candidate_rows": len(y),
                "candidate_rows_scored": int(valid_mask.sum()),
                "candidate_macro_f1": float(f1_score(valid_y, predicted, average="macro", zero_division=0)) if len(valid_y) else None,
                "candidate_accuracy": float(accuracy_score(valid_y, predicted)) if len(valid_y) else None,
                "top1_accuracy_candidate_covered": correct / max(1, aligned_count - missed_targets),
                "top1_accuracy_all_aligned": correct / max(1, aligned_count),
                "log_loss": float(log_loss(valid_y, valid_probabilities, labels=[0, 1])) if valid_probabilities is not None and len(valid_y) else None,
                "classifier_route_continuity": connected / paired if paired else None,
                "continuity_pairs": paired,
                "inference_seconds": inference_seconds,
                "training_seconds": training_times[name],
            })
        foldlike_rows.extend(records_by_model)
        print(f"evaluated student-development trip {inference_trip.trip_id}", flush=True)

    model_summaries: list[dict[str, object]] = []
    for name, estimator in models.items():
        model_records = [row for row in foldlike_rows if row["model"] == name]
        aligned = sum(int(row["aligned_observations"]) for row in model_records)
        misses = sum(int(row["candidate_misses"]) for row in model_records)
        candidate_rows = sum(int(row["candidate_rows"]) for row in model_records)
        summary: dict[str, object] = {
            "model": name,
            "student_test_trip_count": len(model_records),
            "aligned_observations": aligned,
            "candidate_misses": misses,
            "candidate_miss_rate": misses / max(1, aligned),
            "candidate_rows": candidate_rows,
            "top1_accuracy_all_aligned": sum(float(row["top1_accuracy_all_aligned"]) * int(row["aligned_observations"])
                                              for row in model_records) / max(1, aligned),
            "top1_accuracy_candidate_covered": sum(float(row["top1_accuracy_candidate_covered"]) *
                                                   (int(row["aligned_observations"]) - int(row["candidate_misses"]))
                                                   for row in model_records) / max(1, aligned - misses),
            "candidate_macro_f1_mean_trip": float(np.mean([row["candidate_macro_f1"] for row in model_records
                                                           if row["candidate_macro_f1"] is not None])),
            "candidate_accuracy_mean_trip": float(np.mean([row["candidate_accuracy"] for row in model_records
                                                            if row["candidate_accuracy"] is not None])),
            "log_loss_mean_trip": float(np.mean([row["log_loss"] for row in model_records if row["log_loss"] is not None]))
                                 if any(row["log_loss"] is not None for row in model_records) else None,
            "classifier_route_continuity_mean_trip": float(np.mean([row["classifier_route_continuity"] for row in model_records
                                                                     if row["classifier_route_continuity"] is not None]))
                                                       if any(row["classifier_route_continuity"] is not None for row in model_records) else None,
            "training_seconds": training_times[name],
            "inference_seconds_total": sum(float(row["inference_seconds"]) for row in model_records),
        }
        model_summaries.append(summary)

    report = {
        "partition": "student_test_development_only",
        "candidate_settings": {"radius_m": radius_m, "max_candidates": max_candidates},
        "training_partition": "train",
        "training_seed": SEED,
        "training_observations_per_trip_max": 100,
        "alignment_max_distance_m": 250.0,
        "target_semantics": "weak route-sequence-to-track alignment; not independent per-observation ground truth",
        "models": model_summaries,
        "trip_results": foldlike_rows,
        "limitations": [
            "Student-test routes are read only for alignment and scoring; prediction inputs are loaded separately without route labels.",
            "Scores reflect weak route-derived targets and may understate errors from label ambiguity.",
            "Evaluator-testing trips and labels are not accessed.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "student_test_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "student_test_trip_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(foldlike_rows[0]))
        writer.writeheader()
        writer.writerows(foldlike_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--student-test", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    args = parser.parse_args()
    report = evaluate(args.train, args.student_test, args.output,
                      radius_m=args.radius_m, max_candidates=args.max_candidates)
    print(json.dumps({"student_test_trips": report["models"][0]["student_test_trip_count"],
                      "models": report["models"]}, indent=2))


if __name__ == "__main__":
    main()
