# PathSnap technical report

## Objective and data contract

PathSnap maps a time-ordered GPS trace to directed road arcs in a trip-local network. A trip is stored as `.track` (longitude, latitude, timestamp), `.nodes` (longitude, latitude), `.arcs` (from-node and to-node IDs), and `.route` (an ordered sequence of trip-local arc IDs). Route length and GPS sample count differ, so their rows cannot be paired directly. Route IDs are local to each trip and are never global model classes.

The implementation keeps `.route` out of prediction inputs by default. Training and development scoring load reference sequences in separate explicit code paths. The live API loads only `.track`, `.nodes`, and `.arcs`; it does not need a `.route` file.

## Data exploration

EDA covered all 80 training trips (320 required files). The tracks contained 208,417 GPS rows; median track length was 1,529 observations. Median road-network size was 9,608 nodes and 19,526.5 arcs; median route sequence length was 280 arcs. Coordinate and graph references were valid. Thirteen directed route discontinuities appeared across ten trips. A high-noise outlier, trip `00000054`, had substantial GPS jitter, which is why route-derived observation labels are treated as weak targets. The reproducible EDA and findings are described in `reports/phase2_findings.md`; coordinate-level inventory/maps are intentionally kept local.

## Candidate features and classifier experiments

Each GPS observation receives up to eight nearby directed-arc candidates within 50 m. Features describe point-to-segment distance and fraction, segment length, road and movement bearings, direction difference, GPS step/time/speed, and local graph degrees. Trip and arc IDs are excluded from classifier features. A monotonic route-order alignment assigns weak labels only when projection distance is at most 250 m.

Five model families were compared with five-fold GroupKFold by trip, using a seeded per-trip training/validation sample: Logistic Regression, KNN, Random Forest, Histogram Gradient Boosting, and RBF SVM. Random Forest and Histogram Gradient Boosting led the sampled coverage-adjusted Top-1 CV score at about 0.778 and 0.779. On the separate 10-trip student-development partition, Random Forest scored 0.8118 Top-1 accuracy over 19,160 aligned observations and HGB scored 0.8096. These are weak-label results, not independently verified route accuracy. Full comparison details and machine-readable results are in `reports/model_comparison.md` and `experiments/results/`.

A 10-trial Gaussian-process Bayesian search tuned Random Forest tree count, depth, leaf size, and feature fraction using training trips only. Its best Top-1 CV score was 0.7765 (82 trees, depth 21, leaf size 1, feature fraction 0.573), slightly below the default CV score and within fold variation. The default RF remains selected provisionally. Search trials are retained in `experiments/results/bayesian_search_record.json` and `bayesian_search_trials.csv`.

## Sequence decoding

Random Forest candidate probabilities are Viterbi emissions. The transition score rewards remaining on an arc or moving to a directed-connected arc and penalizes disagreement between road distance and GPS displacement. Empty candidate observations remain unmatched and split decoding. A finite fallback allows recovery across noise and disconnected route entries.

On the student-development partition, Viterbi raised Top-1 weak-label accuracy from 0.8118 to 0.8948 and mean per-trip directed continuity from 0.8991 to 0.9949. Candidate generation omitted 91 of 19,160 aligned reference arcs (0.47%). This supports keeping Viterbi in the development implementation. The decoder is not a shortest-path map matcher, and the student-test alignment remains weak reference data. Do not use these numbers as the invigilator score. See `reports/decoder_evaluation.md`.

## API and user interface

`experiments/build_model.py` trains the final default Random Forest on the seeded training sample and writes the `models/pathsnap_rf.joblib` model artifact plus JSON metadata. `src/pathsnap/service.py` serves `/health`, the static UI, and `POST /predict`. The request contains a trip ID; the service reads the corresponding trip inputs from a server-side data directory, predicts candidates, and applies Viterbi. It returns predictions indexed by observation, candidate arc IDs/distances, candidate coverage, and sequence continuity. The UI shows these predictions in a table and displays no reference labels.

The deployment host must store authorized input folders and the model. On that laptop, set `PATHSNAP_DATA_ROOT` and `PATHSNAP_MODEL`, set `PYTHONPATH=src`, then run `python -m pathsnap.service`. It listens on port 8000 and binds to `0.0.0.0` by default. A second laptop on the same network opens `http://<host-laptop-ip>:8000/`. The localhost API/UI flow was smoke-tested; a second physical laptop and network were not available in this workspace, so the live two-device connection must still be checked by the team. For the invigilated run, configure the data root only after authorized evaluator inputs are supplied, and do not use evaluator labels in predictions or the UI.

## Reproducibility and integrity

Python package versions are pinned in `requirements.txt`. Commands for EDA, training, tuning, evaluation, model artifact generation, and serving are in the README. Eight unit tests cover parsing, label isolation, candidates, alignment, decoding, and an HTTP prediction without any `.route` file. Experiment reports distinguish weak-label development metrics from final evaluator results. Training data and precise GPS-derived inventories/maps are not committed. The evaluator-held partition has not been accessed.
