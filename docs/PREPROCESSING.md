# PathSnap GPS Preprocessing & Signal Cleaning Pipeline

## Overview

The `pathsnap.preprocessing` module provides robust filtering, noise rejection, and trajectory smoothing for raw GPS observations prior to candidate arc generation and Viterbi sequence decoding.

Raw GPS observations frequently encounter:
1. **Timestamp Jitter / Redundancy**: Multiple sample fixes logged at the exact same second or out-of-order timestamps.
2. **Kinematic Outliers / Sensor Teleportation**: Multi-kilometer coordinate jumps occurring over 1-second intervals due to satellite loss or multipath reflection.
3. **Measurement Noise**: High-frequency coordinate jitter while traversing straight road segments.

---

## Architecture & Modules

### 1. `pathsnap.kinematics`
- **`haversine_distance(lat1, lon1, lat2, lon2)`**: Spherical great-circle distance (meters) computed using WGS-84 mean radius.
- **`compute_instantaneous_velocity(lat1, lon1, t1, lat2, lon2, t2)`**: Measures incremental physical displacement and derived speed ($m/s$).

### 2. `pathsnap.filters`
- **`filter_duplicate_timestamps(observations)`**: Enforces strict monotonic timestamp sequencing, discarding duplicate time ticks.
- **`filter_speed_outliers(observations, max_speed_mps=45.0)`**: Discards non-physical GPS coordinate leaps exceeding plausible road vehicle thresholds (~160 km/h).

### 3. `pathsnap.kalman`
- **`GPSKalmanFilter2D`**: Constant-velocity state-space Kalman filter tracking position $[\text{lat}, \text{lon}]$ and velocity $[\dot{\text{lat}}, \dot{\text{lon}}]$.
- **`smooth_gps_track(observations)`**: Applies forward recursive filtering to reduce spatial variance.

### 4. `pathsnap.preprocessing`
- **`preprocess_observations(...)`**: End-to-end observation pipeline with configurable `PreprocessConfig`.
- **`preprocess_trip(trip, config=...)`**: Returns a clean `TripData` structure ready for downstream candidate generation.

---

## Usage Example

```python
from pathsnap.data import load_trip
from pathsnap.preprocessing import preprocess_trip, PreprocessConfig

trip = load_trip(trip_directory)

# Configure custom thresholds
config = PreprocessConfig(
    max_speed_mps=35.0,  # ~126 km/h max speed
    apply_kalman_smoothing=True,
    process_noise_std=1e-4,
    measurement_noise_std=5e-5
)

clean_trip = preprocess_trip(trip, config=config)
```
