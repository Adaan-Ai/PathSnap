from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pathsnap.candidates import CandidateGenerator, candidate_features
from pathsnap.data import load_trip
from pathsnap.eda import haversine_m, read_rows, run
from pathsnap.alignment import align_route_to_track


class EdaTests(unittest.TestCase):
    def test_haversine_returns_metres(self) -> None:
        self.assertAlmostEqual(haversine_m(0, 0, 0, 1), 111_195, delta=150)

    def test_read_rows_counts_malformed_and_non_finite_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "points.track"
            source.write_text("1 2 3\ninvalid row\n4 5 inf\n", encoding="utf-8")
            rows, malformed = read_rows(source, 3)
        self.assertEqual(rows, [[1.0, 2.0, 3.0]])
        self.assertEqual(malformed, 2)

    def test_training_eda_writes_inventory_and_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "train"
            trip = root / "trip-a"
            trip.mkdir(parents=True)
            (trip / "trip-a.track").write_text("8.0 49.0 1\n8.0001 49.0 2\n", encoding="utf-8")
            (trip / "trip-a.nodes").write_text("8.0 49.0\n8.0001 49.0\n", encoding="utf-8")
            (trip / "trip-a.arcs").write_text("0 1\n", encoding="utf-8")
            (trip / "trip-a.route").write_text("0\n", encoding="utf-8")
            output = Path(temp) / "reports"
            report = run(root, output)
            aggregate = json.loads((output / "training_eda_summary.json").read_text(encoding="utf-8"))
            inventory = (output / "training_trip_inventory.csv").read_text(encoding="utf-8")

        self.assertEqual(report["trip_count"], 1)
        self.assertEqual(aggregate["required_file_availability"][".track"], 1)
        self.assertIn("route_directed_discontinuities", inventory)
        self.assertIn("trip-a", inventory)

    def test_route_labels_are_opt_in_and_trip_local(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trip = Path(temp) / "trip-a"
            trip.mkdir()
            (trip / "trip-a.track").write_text("8 49 1\n8.0001 49 2\n", encoding="utf-8")
            (trip / "trip-a.nodes").write_text("8 49\n8.0001 49\n", encoding="utf-8")
            (trip / "trip-a.arcs").write_text("0 1\n", encoding="utf-8")
            (trip / "trip-a.route").write_text("0\n", encoding="utf-8")
            prediction_input = load_trip(trip)
            training_input = load_trip(trip, include_route_labels=True)
        self.assertIsNone(prediction_input.route_arc_ids)
        self.assertEqual(training_input.route_arc_ids, (0,))
        self.assertEqual(training_input.arcs[0].arc_id, 0)

    def test_candidate_search_and_features_preserve_each_observation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trip_dir = Path(temp) / "trip-b"
            trip_dir.mkdir()
            (trip_dir / "trip-b.track").write_text("8 49 1\n8.00005 49 2\n50 20 3\n", encoding="utf-8")
            (trip_dir / "trip-b.nodes").write_text("8 49\n8.0001 49\n", encoding="utf-8")
            (trip_dir / "trip-b.arcs").write_text("0 1\n", encoding="utf-8")
            (trip_dir / "trip-b.route").write_text("0\n", encoding="utf-8")
            trip = load_trip(trip_dir)
        generated = CandidateGenerator(radius_m=30, max_candidates=3).generate(trip)
        features = candidate_features(trip, generated)
        self.assertEqual(len(generated), 3)
        self.assertEqual(generated[0].candidates[0].arc_id, 0)
        self.assertEqual(generated[2].status, "no_candidate_within_radius")
        self.assertEqual(len(features), 2)
        self.assertIn("perpendicular_distance_m", features[0])

    def test_route_alignment_is_monotonic_and_never_forces_missing_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trip_dir = Path(temp) / "trip-c"
            trip_dir.mkdir()
            (trip_dir / "trip-c.track").write_text(
                "8.00001 49 1\n8.00011 49 2\n8.00019 49 3\n50 20 4\n", encoding="utf-8")
            (trip_dir / "trip-c.nodes").write_text(
                "8 49\n8.0001 49\n8.0002 49\n", encoding="utf-8")
            (trip_dir / "trip-c.arcs").write_text("0 1\n1 2\n", encoding="utf-8")
            (trip_dir / "trip-c.route").write_text("0\n1\n", encoding="utf-8")
            trip = load_trip(trip_dir, include_route_labels=True)
        candidates = CandidateGenerator(radius_m=20).generate(trip)
        alignment = align_route_to_track(trip, candidates)
        positions = [position for position in alignment.route_positions_by_observation if position is not None]
        self.assertEqual(alignment.aligned_observations, 3)
        self.assertEqual(alignment.unmatched_observations, (3,))
        self.assertEqual(positions, sorted(positions))


if __name__ == "__main__":
    unittest.main()
