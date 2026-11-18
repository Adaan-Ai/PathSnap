from __future__ import annotations

import unittest
from pathsnap.data import TrackObservation, TripData, RoadArc
from pathsnap.kinematics import haversine_distance, compute_instantaneous_velocity
from pathsnap.filters import filter_duplicate_timestamps, filter_speed_outliers
from pathsnap.kalman import GPSKalmanFilter2D, smooth_gps_track
from pathsnap.preprocessing import PreprocessConfig, preprocess_observations, preprocess_trip


class TestKinematics(unittest.TestCase):
    def test_haversine_same_point(self):
        dist = haversine_distance(37.7749, -122.4194, 37.7749, -122.4194)
        self.assertAlmostEqual(dist, 0.0, places=4)

    def test_haversine_known_distance(self):
        # Approx 111 km per degree latitude
        dist = haversine_distance(0.0, 0.0, 1.0, 0.0)
        self.assertGreater(dist, 110000)
        self.assertLess(dist, 112000)

    def test_instantaneous_velocity(self):
        dist, speed = compute_instantaneous_velocity(0.0, 0.0, 0.0, 1.0, 0.0, 3600.0)
        self.assertGreater(dist, 110000)
        self.assertAlmostEqual(speed, dist / 3600.0, places=3)


class TestFilters(unittest.TestCase):
    def test_filter_duplicate_timestamps(self):
        obs = [
            TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=1.0),
            TrackObservation(longitude=0.1, latitude=0.1, timestamp_s=1.0),
            TrackObservation(longitude=0.2, latitude=0.2, timestamp_s=2.0),
        ]
        filtered = filter_duplicate_timestamps(obs)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0].timestamp_s, 1.0)
        self.assertEqual(filtered[1].timestamp_s, 2.0)

    def test_filter_speed_outliers(self):
        obs = [
            TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=0.0),
            # Impossible jump in 1 second (~111 km)
            TrackObservation(longitude=0.0, latitude=1.0, timestamp_s=1.0),
            # Realistic step
            TrackObservation(longitude=0.0001, latitude=0.0001, timestamp_s=2.0),
        ]
        filtered = filter_speed_outliers(obs, max_speed_mps=50.0)
        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[1].timestamp_s, 2.0)


class TestKalmanFilter(unittest.TestCase):
    def test_kalman_smoother(self):
        obs = [
            TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=0.0),
            TrackObservation(longitude=0.0001, latitude=0.0001, timestamp_s=1.0),
            TrackObservation(longitude=0.0002, latitude=0.0002, timestamp_s=2.0),
        ]
        smoothed = smooth_gps_track(obs)
        self.assertEqual(len(smoothed), 3)
        self.assertAlmostEqual(smoothed[0].latitude, 0.0, places=4)


class TestPipeline(unittest.TestCase):
    def test_preprocess_trip(self):
        trip = TripData(
            trip_id="test_001",
            observations=(
                TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=0.0),
                TrackObservation(longitude=0.0, latitude=0.0, timestamp_s=0.0),
                TrackObservation(longitude=0.0001, latitude=0.0001, timestamp_s=1.0),
            ),
            nodes=((0.0, 0.0), (0.0001, 0.0001)),
            arcs=(RoadArc(arc_id=1, from_node=0, to_node=1),),
            route_arc_ids=None,
        )
        cleaned = preprocess_trip(trip)
        self.assertEqual(len(cleaned.observations), 2)
        self.assertEqual(cleaned.trip_id, "test_001")


if __name__ == "__main__":
    unittest.main()
