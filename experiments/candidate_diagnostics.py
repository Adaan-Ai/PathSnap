"""Measure trip-grouped candidate coverage and reference-arc miss rate."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from pathsnap.alignment import align_route_to_track
from pathsnap.candidates import CandidateGenerator
from pathsnap.data import load_trip


def diagnose(dataset: Path, output: Path, *, radius_m: float = 50.0,
             max_candidates: int = 8, alignment_distance_m: float = 250.0) -> dict[str, object]:
    generator = CandidateGenerator(radius_m=radius_m, max_candidates=max_candidates)
    trip_results: list[dict[str, object]] = []
    started = time.perf_counter()
    for directory in sorted(path for path in dataset.iterdir() if path.is_dir()):
        trip = load_trip(directory, include_route_labels=True)
        alignment = align_route_to_track(trip, max_alignment_distance_m=alignment_distance_m)
        candidates = generator.generate(trip)
        aligned_pairs = [(target, row) for target, row in zip(alignment.arc_ids_by_observation, candidates)
                         if target is not None]
        misses = sum(target not in {candidate.arc_id for candidate in row.candidates}
                     for target, row in aligned_pairs)
        candidate_observations = sum(bool(row.candidates) for row in candidates)
        trip_results.append({
            "trip_id": trip.trip_id,
            "observations": len(trip.observations),
            "candidate_observation_coverage": candidate_observations / max(1, len(trip.observations)),
            "aligned_observations": len(aligned_pairs),
            "alignment_coverage": len(aligned_pairs) / max(1, len(trip.observations)),
            "aligned_arc_candidate_misses": misses,
            "candidate_miss_rate": misses / max(1, len(aligned_pairs)),
            "candidate_rows": sum(len(row.candidates) for row in candidates),
            "alignment_segments": alignment.alignment_segments,
        })

    observation_count = sum(int(row["observations"]) for row in trip_results)
    aligned_count = sum(int(row["aligned_observations"]) for row in trip_results)
    misses = sum(int(row["aligned_arc_candidate_misses"]) for row in trip_results)
    report = {
        "partition": "training",
        "trip_count": len(trip_results),
        "settings": {"candidate_radius_m": radius_m,
                     "maximum_candidates_per_observation": max_candidates,
                     "weak_alignment_max_distance_m": alignment_distance_m},
        "observations": observation_count,
        "aligned_observations": aligned_count,
        "weak_alignment_coverage": aligned_count / max(1, observation_count),
        "aligned_arc_candidate_misses": misses,
        "candidate_miss_rate": misses / max(1, aligned_count),
        "candidate_observation_coverage": sum(int(row["observations"]) * float(row["candidate_observation_coverage"])
                                               for row in trip_results) / max(1, observation_count),
        "median_trip_candidate_miss_rate": statistics.median(float(row["candidate_miss_rate"])
                                                             for row in trip_results),
        "runtime_seconds": round(time.perf_counter() - started, 2),
        "label_caveat": "Miss rates compare candidates to monotonic route-derived weak labels, not independent per-observation truth.",
        "trips": trip_results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/candidate_diagnostics.json"))
    parser.add_argument("--radius-m", type=float, default=50.0)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--alignment-distance-m", type=float, default=250.0)
    args = parser.parse_args()
    report = diagnose(args.dataset, args.output, radius_m=args.radius_m,
                      max_candidates=args.max_candidates,
                      alignment_distance_m=args.alignment_distance_m)
    print(json.dumps({key: report[key] for key in (
        "trip_count", "observations", "aligned_observations", "candidate_miss_rate",
        "candidate_observation_coverage", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
