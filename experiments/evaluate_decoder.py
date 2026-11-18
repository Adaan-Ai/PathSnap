"""Measure classifier-only and HMM/Viterbi results on student-test trips."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pathsnap.alignment import align_route_to_track
from pathsnap.candidates import CandidateGenerator
from pathsnap.data import load_trip
from pathsnap.decoder import viterbi_decode
from train_compare import SEED, _feature_row, _load_sampled_candidates


def evaluate(train_root: Path, student_test_root: Path, output: Path,
             radius_m: float = 50.0, max_candidates: int = 8) -> dict[str, object]:
    x, y, _groups, _keys, sample_ids, *_ = _load_sampled_candidates(
        train_root, radius_m, max_candidates)
    training_rows = np.flatnonzero(sample_ids == 0)
    model = RandomForestClassifier(n_estimators=120, max_depth=14, min_samples_leaf=2,
                                   class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    start = time.perf_counter()
    model.fit(x[training_rows], y[training_rows])
    fit_seconds = time.perf_counter() - start
    generator = CandidateGenerator(radius_m=radius_m, max_candidates=max_candidates)
    trip_records: list[dict[str, object]] = []
    for directory in sorted(path for path in student_test_root.iterdir() if path.is_dir()):
        prediction_trip = load_trip(directory, include_route_labels=False)
        scoring_trip = load_trip(directory, include_route_labels=True)
        alignment = align_route_to_track(scoring_trip, max_alignment_distance_m=250.0)
        generated = generator.generate(prediction_trip)
        outgoing: dict[int, int] = {}
        incoming: dict[int, int] = {}
        for arc in prediction_trip.arcs:
            outgoing[arc.from_node] = outgoing.get(arc.from_node, 0) + 1
            incoming[arc.to_node] = incoming.get(arc.to_node, 0) + 1
        features: list[list[float]] = []
        owners: list[int] = []
        for position, row in enumerate(generated):
            for candidate in row.candidates:
                features.append(_feature_row(prediction_trip, row.observation_index,
                                             candidate, outgoing, incoming))
                owners.append(position)
        probs_by_observation = [np.asarray([], dtype=float) for _ in generated]
        if features:
            candidate_probabilities = model.predict_proba(np.asarray(features, dtype=np.float32))[:, 1]
            chunks: list[list[float]] = [[] for _ in generated]
            for owner, probability in zip(owners, candidate_probabilities):
                chunks[owner].append(float(probability))
            probs_by_observation = [np.asarray(values, dtype=float) for values in chunks]
        decoded = viterbi_decode(prediction_trip, generated, probs_by_observation)
        targets = alignment.arc_ids_by_observation
        aligned_indices = [i for i, target in enumerate(targets) if target is not None]
        covered_indices = [i for i in aligned_indices if any(
            candidate.arc_id == targets[i] for candidate in generated[i].candidates)]
        classifier_correct = sum(decoded.candidate_only_arc_ids[i] == targets[i] for i in aligned_indices)
        decoder_correct = sum(decoded.arc_ids[i] == targets[i] for i in aligned_indices)

        def scored_continuity(sequence: tuple[int | None, ...]) -> float | None:
            pairs = [(a, b) for a, b in zip(sequence, sequence[1:])
                     if a is not None and b is not None]
            if not pairs:
                return None
            return sum(int(prediction_trip.arcs[a].to_node == prediction_trip.arcs[b].from_node or a == b)
                       for a, b in pairs) / len(pairs)

        trip_records.append({
            "trip_id": prediction_trip.trip_id,
            "observations": len(prediction_trip.observations),
            "aligned_observations": len(aligned_indices),
            "candidate_covered_aligned": len(covered_indices),
            "candidate_misses": len(aligned_indices) - len(covered_indices),
            "candidate_miss_rate": (len(aligned_indices) - len(covered_indices)) / max(1, len(aligned_indices)),
            "classifier_top1_covered": classifier_correct / max(1, len(covered_indices)),
            "classifier_top1_all_aligned": classifier_correct / max(1, len(aligned_indices)),
            "viterbi_top1_all_aligned": decoder_correct / max(1, len(aligned_indices)),
            "classifier_continuity": scored_continuity(decoded.candidate_only_arc_ids),
            "viterbi_continuity": scored_continuity(decoded.arc_ids),
            "viterbi_transition_count": decoded.transition_count,
        })
        print(f"decoded student-development trip {prediction_trip.trip_id}", flush=True)

    aligned_total = sum(int(row["aligned_observations"]) for row in trip_records)
    covered_total = sum(int(row["candidate_covered_aligned"]) for row in trip_records)
    classifier_correct_total = sum(float(row["classifier_top1_all_aligned"]) * int(row["aligned_observations"])
                                   for row in trip_records)
    viterbi_correct_total = sum(float(row["viterbi_top1_all_aligned"]) * int(row["aligned_observations"])
                                for row in trip_records)
    summary = {
        "partition": "student_test_development_only",
        "training_partition": "train",
        "model": "RandomForestClassifier n_estimators=120 max_depth=14 min_samples_leaf=2",
        "candidate_settings": {"radius_m": radius_m, "max_candidates": max_candidates},
        "alignment_max_distance_m": 250.0,
        "aligned_observations": aligned_total,
        "candidate_miss_rate": 1 - covered_total / max(1, aligned_total),
        "classifier_top1_all_aligned": classifier_correct_total / max(1, aligned_total),
        "viterbi_top1_all_aligned": viterbi_correct_total / max(1, aligned_total),
        "classifier_mean_trip_continuity": float(np.nanmean([row["classifier_continuity"] for row in trip_records
                                                              if row["classifier_continuity"] is not None])),
        "viterbi_mean_trip_continuity": float(np.nanmean([row["viterbi_continuity"] for row in trip_records
                                                           if row["viterbi_continuity"] is not None])),
        "training_seconds": fit_seconds,
        "trip_results": trip_records,
        "limitations": [
            "Student-test labels are route-sequence-derived weak alignments, not manually verified observation labels.",
            "Viterbi transitions use direct arc adjacency, same-arc progression and a finite discontinuity fallback.",
            "Evaluator-testing trips and labels were not accessed.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--student-test", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/decoder_student_test.json"))
    args = parser.parse_args()
    result = evaluate(args.train, args.student_test, args.output)
    print(json.dumps({key: result[key] for key in (
        "aligned_observations", "candidate_miss_rate", "classifier_top1_all_aligned",
        "viterbi_top1_all_aligned", "classifier_mean_trip_continuity", "viterbi_mean_trip_continuity")}, indent=2))


if __name__ == "__main__":
    main()
