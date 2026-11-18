"""Lightweight 2D kinematic Kalman Filter for smoothing noisy GPS coordinates."""

from __future__ import annotations

import math
from typing import List, Sequence
from pathsnap.data import TrackObservation


class GPSKalmanFilter2D:
    """Constant-velocity 2D Kalman Filter in latitude/longitude degrees space."""

    def __init__(
        self,
        process_noise_std: float = 1e-4,
        measurement_noise_std: float = 5e-5,
    ) -> None:
        self.q = process_noise_std ** 2
        self.r = measurement_noise_std ** 2

        # State vector: [lat, lon, v_lat, v_lon]
        self.x = [0.0, 0.0, 0.0, 0.0]
        # Diagonal covariance approximation
        self.p = [1.0, 1.0, 1.0, 1.0]
        self.last_timestamp: float | None = None

    def initialize(self, lat: float, lon: float, timestamp_s: float) -> None:
        self.x = [lat, lon, 0.0, 0.0]
        self.p = [self.r, self.r, 1.0, 1.0]
        self.last_timestamp = timestamp_s

    def update(self, lat: float, lon: float, timestamp_s: float) -> tuple[float, float]:
        if self.last_timestamp is None:
            self.initialize(lat, lon, timestamp_s)
            return lat, lon

        dt = max(1e-3, timestamp_s - self.last_timestamp)
        self.last_timestamp = timestamp_s

        # Predict step
        self.x[0] += self.x[2] * dt
        self.x[1] += self.x[3] * dt
        self.p[0] += (self.p[2] * (dt ** 2)) + self.q
        self.p[1] += (self.p[3] * (dt ** 2)) + self.q

        # Measurement update (lat)
        k_lat = self.p[0] / (self.p[0] + self.r)
        self.x[0] += k_lat * (lat - self.x[0])
        self.p[0] *= (1.0 - k_lat)

        # Measurement update (lon)
        k_lon = self.p[1] / (self.p[1] + self.r)
        self.x[1] += k_lon * (lon - self.x[1])
        self.p[1] *= (1.0 - k_lon)

        return self.x[0], self.x[1]


def smooth_gps_track(
    observations: Sequence[TrackObservation],
    process_noise_std: float = 1e-4,
    measurement_noise_std: float = 5e-5,
) -> List[TrackObservation]:
    """Apply Kalman smoothing over a sequence of TrackObservation fixes."""
    if len(observations) <= 1:
        return list(observations)

    kf = GPSKalmanFilter2D(
        process_noise_std=process_noise_std,
        measurement_noise_std=measurement_noise_std,
    )
    smoothed: List[TrackObservation] = []
    for obs in observations:
        s_lat, s_lon = kf.update(obs.latitude, obs.longitude, obs.timestamp_s)
        smoothed.append(TrackObservation(longitude=s_lon, latitude=s_lat, timestamp_s=obs.timestamp_s))

    return smoothed
