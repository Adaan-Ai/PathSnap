# PathSnap

Machine learning-based GPS map matching and route reconstruction using road networks, an HMM, and Viterbi decoding.

## Current status: development pipeline implemented

The repository contains training-partition EDA, strict trip loading, spatial candidate generation, weak-label model experiments, a Viterbi decoder, a prediction API, and a browser UI. Raw datasets stay outside Git; pass the path to a partition you are authorized to analyze. Do not expose evaluator labels to the student prediction pipeline or UI.

The EDA command reads only the directory passed with `--dataset`. It inspects trip files, GPS validity and timing, geographic extents, network sizes/connectivity, and route-index/continuity diagnostics. It does not assume one GPS sample corresponds to one route arc.

The dataset guide confirms that `.route` contains an ordered sequence of trip-local arc IDs. It does not provide an observation-to-arc alignment. `load_trip()` therefore excludes the route by default; training code must opt in to labels explicitly. Candidate generation ranks nearby directed arcs in a trip-local projected grid and retains empty candidate sets for observations with no nearby arc. Its initial radius/top-k values are pilot settings, not tuned production parameters.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python src/pathsnap/eda.py `
  --dataset "C:\path\to\train" `
  --output reports/eda
$env:PYTHONPATH = "src"
python src/pathsnap/eda_visualizations.py `
  --dataset "C:\path\to\train" `
  --inventory reports/eda/training_trip_inventory.csv `
  --output reports/eda/visualizations
python experiments/generate_development_map.py
python experiments/candidate_diagnostics.py `
  --dataset "C:\path\to\train" `
  --output experiments/results/candidate_diagnostics.json
python experiments/train_compare.py `
  --dataset "C:\path\to\train" `
  --output experiments/results
python experiments/bayesian_tune.py `
  --dataset "C:\path\to\train" `
  --output experiments/results
python experiments/evaluate_student_test.py `
  --train "C:\path\to\train" `
  --student-test "C:\path\to\student_test" `
  --output experiments/results
python experiments/compare_tuned_rf.py `
  --train "C:\path\to\train" `
  --student-test "C:\path\to\student_test" `
  --output experiments/results/tuned_rf_comparison.json
python experiments/evaluate_decoder.py `
  --train "C:\path\to\train" `
  --student-test "C:\path\to\student_test"
python experiments/build_model.py `
  --dataset "C:\path\to\train" `
  --output models/pathsnap_rf.joblib
python experiments/smoke_service.py `
  --dataset "C:\path\to\student_test" `
  --trip-id "00000000"
```

To start the API/UI on the data-host laptop, run `.\scripts\start_server.ps1 -DataRoot "C:\path\to\authorized\trips"`. The service listens on `0.0.0.0:8000`; from the second laptop, check it with `.\scripts\check_server.ps1 -BaseUrl "http://<data-host-ip>:8000"` and open that address in a browser. If Windows Firewall blocks access, allow Python on the private network profile on the host; do not expose the service on a public network. Keep authorized trip files on the host. The service loads only `.track`, `.nodes`, and `.arcs`; it never reads or returns `.route` labels. For invigilated testing, set the data root only after the organiser supplies the permitted evaluator inputs.

The EDA output is derived from the supplied training partition. It includes a JSON aggregate summary and per-trip CSV inventory. Visualizations include training-set size distributions and a Folium map of one real training trip's GPS observations and road network. The model-comparison runner uses GroupKFold by trip and records five candidate scorers' results; observation-level targets are weak labels aligned monotonically from route sequences. The development evaluator loads student-test routes only in its separate scoring path. The initial 10-trip student-test comparison is in `reports/model_comparison.md`; Random Forest and Histogram Gradient Boosting are close, and these weak-label results are provisional.

Per-trip inventories and generated maps stay local: they include precise GPS locations from the provided dataset and are ignored by Git. The map-generation code is included so the team can recreate them from its local data.

`experiments/generate_development_map.py` creates a four-layer Folium map for permitted student-development trip `00000000`, with the road network, raw GPS observations, route-derived weak reference and Random Forest plus Viterbi output. It also writes a static figure for the report. The `.route` file is read only to construct the development visualization; the prediction call itself excludes route labels. Generated geographic files stay local and should not be committed.

## Planned phases

1. Dataset exploration and validation complete for the supplied training partition.
2. Trip loading, candidate generation, and data-integrity checks implemented; see `reports/phase2_findings.md`.
3. Monotonic route-sequence alignment implemented as a weak-label method with documented limitations.
4. Five model families compared with trip-grouped CV and the 10-trip student-development partition; search records retained.
5. Random Forest plus Viterbi decoder selected provisionally based on student-development measurements; see `reports/decoder_evaluation.md`.
6. Local HTTP API and browser UI implemented with a server-side prediction boundary.
7. Local service smoke test and second-device access instructions prepared. A second physical laptop/network is not attached to this workspace, so the two-device connection still needs a live check by the team.
8. Technical report and reproducible run instructions are in `reports/` and this README. The invigilator-held evaluator run remains for the final session.

The Bayesian search optimizes Random Forest settings using only trip-grouped folds from `train`; its full trial record is retained. The evaluator-testing partition is reserved for the final invigilated evaluation. No evaluator data or labels are needed to run these commands.
