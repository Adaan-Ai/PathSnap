# Phase 2: preprocessing and candidate generation

## Verified data semantics

The dataset guide defines `.arcs` as two integer node IDs per row and assigns each arc its zero-based row index. `.route` contains a chronological sequence of those arc IDs. This is trip-local: arc ID `0` in one trip is not a shared class label for arc ID `0` in another. The documentation explicitly says there is no direct row correspondence between `.track` and `.route`.

The EDA found a median of 1,529 GPS observations and 280 reference route arcs per trip. The sequences must not be zipped or treated as observation labels. Ten training trips had at least one directed route-arc endpoint discontinuity (13 adjacent pairs total); their route encodings need review during alignment.

## Implemented

- `src/pathsnap/data.py` strictly reads whitespace-delimited trip files, validates coordinates and graph indices, and keeps the reference route out of `TripData` unless `include_route_labels=True` is explicitly requested.
- `src/pathsnap/candidates.py` uses a local equirectangular projection and grid index to find directed arc candidates within a configurable radius. It retains one result record per observation, including explicit statuses for empty candidate sets.
- `src/pathsnap/alignment.py` independently measures GPS points against the reference route arcs, then uses a monotonic route-order Viterbi pass. Prediction candidates do not supply alignment states, so candidate misses can be measured separately. Points farther than 250 m from the reference route remain unmatched.
- Candidate features include perpendicular distance, projection fraction, segment length, road and movement bearings, direction difference, GPS step distance, timestamp delta, speed, and local in/out degree. Arc and trip IDs are metadata; they must not be fed as global class identifiers.
- Six unit tests now cover data-label opt-in, malformed rows, distance units, candidate selection, empty candidates, monotonic alignment, and inventory generation.

## Pilot observation (not a model result)

On training trip `00000046`, the default pilot settings (50 m radius, at most 8 candidates) produced candidates for all 1,401 observations and 6,736 candidate rows in about 0.15 seconds on this machine. This single-trip observation-coverage check does **not** estimate reference-arc candidate miss rate, validate the radius across trips, or establish model quality.

A training-wide diagnostic using those same provisional settings found at least one candidate for 190,188 of 208,417 observations (91.3%). Independently aligned route targets were available for 208,067 observations (99.83%); the 50 m/top-8 prediction candidates omitted the aligned route arc for 24,452 of those (11.75%). The median per-trip miss rate was 0.34%, but trip `00000054` was a major outlier. The monotonic alignments used 85 route-order segments. These are weak-label diagnostics on the training partition, not model accuracy.

Trip `00000054` is a clear stress case: only 14.5% of its observations received a 50 m candidate, and the 50 m/top-8 list missed 90.7% of its independently aligned route arcs. For this trip, 90.5% of GPS points were within 200 m of some reference-route arc. With 200 m radius, route-arc presence among candidates increased from 58.7% at top-16 to 79.2% at top-32; the top-32 candidate miss rate was still 21.6%. This suggests candidate truncation and severe GPS noise need further study; it does not justify selecting 200 m/32 as the final settings. The GPS trace accumulates about 169 km of point-to-point movement while its reference route arcs total about 31 km, although route and track endpoints are within roughly 25 m. This indicates substantial GPS jitter and supports treating the route-to-track targets as weak labels.

## Next work and limits

Five candidate scorers have been compared with trip-grouped cross-validation and on the separate 10-trip student-development partition. The initial student-test comparison puts Random Forest and Histogram Gradient Boosting nearly tied at 81.2% and 81.0% weak-label Top-1 accuracy, respectively. Candidate settings and route discontinuity cases still need calibration and review. Model tuning and HMM/Viterbi sequence decoding remain next; the evaluator partition remains excluded.
