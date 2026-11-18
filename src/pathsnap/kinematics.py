"""Geodesic and kinematic helper utilities for GPS track analysis."""

from __future__ import annotations

import math
from typing import Tuple

# WGS-84 mean Earth radius in meters
EARTH_RADIUS_M = 6371008.8


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points on the Earth (in meters)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def compute_instantaneous_velocity(
    lat1: float, lon1: float, t1: float,
    lat2: float, lon2: float, t2: float
) -> Tuple[float, float]:
    """Compute (distance_meters, speed_mps) between two sequential timestamped fixes."""
    dt = t2 - t1
    if dt < 0:
        raise ValueError(f"Timestamps must be non-decreasing: t1={t1}, t2={t2}")
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    speed = dist / dt if dt > 0 else 0.0
    return dist, speed
