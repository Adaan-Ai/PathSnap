# Phase 2: preprocessing and candidate generation

## Verified data semantics

The dataset guide defines `.arcs` as two integer node IDs per row and assigns each arc its zero-based row index. `.route` contains a chronological sequence of those arc IDs. This is trip-local: arc ID `0` in one trip is not a shared class label for arc ID `0` in another. The documentation explicitly says there is no direct row correspondence between `.track` and `.route`.

The EDA found a median of 1,529 GPS observations and 280 reference route arcs per trip. The sequences must not be zipped or treated as observation labels. Ten training trips had at least one directed route-arc endpoint discontinuity (13 adjacent pairs total); their route encodings need review during alignment.

## Implemented

- `src/pathsnap/data.py` strictly reads whitespace-delimited trip files, validates coordinates and graph indices, and keeps the reference route out of `TripData` unless `include_route_labels=True` is explicitly requested.
- `src/pathsnap/candidates.py` uses a local equirectangular projection and grid index to find directed arc candidates within a configurable radius. It retains one result record per observation, including explicit statuses for empty candidate sets.
- Candidate features include perpendicular distance, projection fraction, segment length, road and movement bearings, direction difference, GPS step distance, timestamp delta, speed, and local in/out degree. Arc and trip IDs are metadata; they must not be fed as global class identifiers.
- Five unit tests now cover data-label opt-in, malformed rows, distance units, candidate selection, empty candidates, and inventory generation.

## Pilot observation (not a model result)

On training trip `00000046`, the default pilot settings (50 m radius, at most 8 candidates) produced candidates for all 1,401 observations and 6,736 candidate rows in about 0.15 seconds on this machine. This single-trip observation-coverage check does **not** estimate reference-arc candidate miss rate, validate the radius across trips, or establish model quality.

A training-wide diagnostic using those same provisional settings found at least one candidate for 190,188 of 208,417 observations (91.3%). The monotonic alignment routine matched 184,393 observations (88.5%) to route arcs present in those candidate lists, creating 166 route-order segments. These are weak-label coverage diagnostics on the same training partition, not model accuracy or an unbiased candidate-miss estimate.

Trip `00000054` is a clear stress case: only 14.5% of its observations received a 50 m candidate and only 11.0% aligned to a route candidate under the provisional top-8 setting. A distance-to-route diagnostic found 90.5% of its GPS points within 200 m of some route arc, while only 34.1% had a route arc among the nearest 8 network candidates at 200 m. Increasing the list to 32 candidates raised that figure to 79.2%. This suggests candidate truncation and severe GPS noise need further study; it does not justify selecting 200 m/32 as the final settings. For this trip, the GPS trace accumulates about 169 km of point-to-point movement while its reference route arcs total about 31 km, although the route and track endpoints are within roughly 25 m. That is evidence of substantial GPS jitter and supports treating the route-to-track targets as weak labels.

## Next work and limits

Calibrate the initial monotonic route-to-track alignment and candidate settings, inspect the discontinuity cases, and preserve unmatched observations before fitting candidate classifiers. Then run trip-grouped cross-validation on the 80 training trips and, if supplied, development comparisons on the separate student-testing partition. The evaluator partition remains excluded. The invigilator-only evaluator set is not needed for these steps.
