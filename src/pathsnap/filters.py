"""GPS track filtering and outlier rejection strategies."""

from __future__ import annotations

from typing import List, Sequence
from pathsnap.data import TrackObservation
from pathsnap.kinematics import compute_instantaneous_velocity


def filter_speed_outliers(
    observations: Sequence[TrackObservation],
    max_speed_mps: float = 45.0,  # ~162 km/h, reasonable upper bound for city/highway traffic
) -> List[TrackObservation]:
    """Filter out anomalous GPS jumps where implied velocity exceeds realistic physical limits."""
    if len(observations) <= 1:
        return list(observations)

    cleaned: List[TrackObservation] = [observations[0]]
    for obs in observations[1:]:
        last_valid = cleaned[-1]
        _, speed = compute_instantaneous_velocity(
            last_valid.latitude, last_valid.longitude, last_valid.timestamp_s,
            obs.latitude, obs.longitude, obs.timestamp_s,
        )
        if speed <= max_speed_mps:
            cleaned.append(obs)
    return cleaned


def filter_duplicate_timestamps(
    observations: Sequence[TrackObservation]
) -> List[TrackObservation]:
    """Remove redundant zero-time interval observations keeping first valid occurrence."""
    if not observations:
        return []
    deduped: List[TrackObservation] = [observations[0]]
    for obs in observations[1:]:
        if obs.timestamp_s > deduped[-1].timestamp_s:
            deduped.append(obs)
    return deduped
