"""High-level preprocessing pipeline combining filtering and smoothing for PathSnap trips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple
from pathsnap.data import TrackObservation, TripData
from pathsnap.filters import filter_duplicate_timestamps, filter_speed_outliers
from pathsnap.kalman import smooth_gps_track


@dataclass(frozen=True)
class PreprocessConfig:
    max_speed_mps: float = 45.0
    apply_kalman_smoothing: bool = True
    process_noise_std: float = 1e-4
    measurement_noise_std: float = 5e-5


def preprocess_observations(
    observations: Sequence[TrackObservation],
    config: PreprocessConfig = PreprocessConfig(),
) -> Tuple[TrackObservation, ...]:
    """Clean track observations: deduplicate timestamps, remove velocity outliers, and smooth noise."""
    if not observations:
        return ()

    cleaned = filter_duplicate_timestamps(observations)
    cleaned = filter_speed_outliers(cleaned, max_speed_mps=config.max_speed_mps)

    if config.apply_kalman_smoothing and len(cleaned) > 1:
        cleaned = smooth_gps_track(
            cleaned,
            process_noise_std=config.process_noise_std,
            measurement_noise_std=config.measurement_noise_std,
        )

    return tuple(cleaned)


def preprocess_trip(
    trip: TripData,
    config: PreprocessConfig = PreprocessConfig(),
) -> TripData:
    """Return a new TripData with preprocessed and cleaned observations."""
    clean_obs = preprocess_observations(trip.observations, config=config)
    return TripData(
        trip_id=trip.trip_id,
        observations=clean_obs,
        nodes=trip.nodes,
        arcs=trip.arcs,
        route_arc_ids=trip.route_arc_ids,
    )
