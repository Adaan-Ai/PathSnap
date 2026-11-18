from __future__ import annotations

import unittest
import numpy as np
from pathsnap.data import TrackObservation, TripData, RoadArc
from pathsnap.candidates import ObservationCandidates, RoadCandidate
from pathsnap.decoder import viterbi_decode, DecodeResult


class TestViterbiDecoder(unittest.TestCase):
    def setUp(self):
        self.trip = TripData(
            trip_id="test_viterbi",
            observations=(
                TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=0.0),
                TrackObservation(longitude=0.001, latitude=0.001, timestamp_s=1.0),
            ),
            nodes=((0.0, 0.0), (0.001, 0.001), (0.002, 0.002)),
            arcs=(
                RoadArc(arc_id=0, from_node=0, to_node=1),
                RoadArc(arc_id=1, from_node=1, to_node=2),
            ),
            route_arc_ids=None,
        )

    def test_single_candidate_per_observation(self):
        cand0 = RoadCandidate(
            arc_id=0,
            perpendicular_distance_m=5.0,
            projection_fraction=0.5,
            segment_length_m=100.0,
            road_bearing_deg=45.0,
            movement_bearing_deg=45.0,
            direction_difference_deg=0.0,
        )
        cand1 = RoadCandidate(
            arc_id=1,
            perpendicular_distance_m=5.0,
            projection_fraction=0.5,
            segment_length_m=100.0,
            road_bearing_deg=45.0,
            movement_bearing_deg=45.0,
            direction_difference_deg=0.0,
        )

        generated = [
            ObservationCandidates(observation_index=0, timestamp_s=0.0, candidates=(cand0,), status="ok"),
            ObservationCandidates(observation_index=1, timestamp_s=1.0, candidates=(cand1,), status="ok"),
        ]
        probabilities = [np.array([1.0]), np.array([1.0])]

        result = viterbi_decode(self.trip, generated, probabilities)
        self.assertIsInstance(result, DecodeResult)
        self.assertEqual(result.arc_ids, (0, 1))
        self.assertEqual(result.connected_transition_rate, 1.0)

    def test_empty_candidates_handled(self):
        cand0 = RoadCandidate(
            arc_id=0,
            perpendicular_distance_m=5.0,
            projection_fraction=0.5,
            segment_length_m=100.0,
            road_bearing_deg=45.0,
            movement_bearing_deg=45.0,
            direction_difference_deg=0.0,
        )
        generated = [
            ObservationCandidates(observation_index=0, timestamp_s=0.0, candidates=(cand0,), status="ok"),
            ObservationCandidates(observation_index=1, timestamp_s=1.0, candidates=(), status="empty"),
        ]
        probabilities = [np.array([1.0]), np.array([])]

        result = viterbi_decode(self.trip, generated, probabilities)
        self.assertEqual(result.arc_ids, (0, None))


if __name__ == "__main__":
    unittest.main()
