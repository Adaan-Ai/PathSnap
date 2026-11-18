# PathSnap

Machine learning-based GPS map matching and route reconstruction using road networks, an HMM, and Viterbi decoding.

## Current phase: preprocessing and candidate generation

The repository contains the training-partition inventory and EDA tools, plus strict trip loading and spatial candidate generation. The raw dataset stays outside Git; pass the path to the partition you are authorized to analyze. Do not place evaluator-testing data in the student prediction pipeline or expose its labels to the UI.

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
```

The EDA output is derived from the supplied training partition. It includes a JSON aggregate summary and per-trip CSV inventory. Visualizations include training-set size distributions and a Folium map of one real training trip's GPS observations and road network. The model-comparison runner uses GroupKFold by trip and records five candidate scorers' results; observation-level targets are weak labels aligned monotonically from route sequences. The development evaluator loads student-test routes only in its separate scoring path. The initial 10-trip student-test comparison is in `reports/model_comparison.md`; Random Forest and Histogram Gradient Boosting are close, and these weak-label results are provisional.

Per-trip inventories and generated maps stay local: they include precise GPS locations from the provided dataset and are ignored by Git. The map-generation code is included so the team can recreate them from its local data.

## Planned phases

1. Dataset exploration and validation.
2. Establish label semantics and trip-level splits; implement preprocessing and candidate generation (in progress; candidate misses require follow-up).
3. Align training route sequences to GPS with a monotonic weak-label method and inspect its limitations.
4. Compare five distinct models using trip-grouped cross-validation; keep optimization trials and measurements (CV complete; see `reports/model_comparison.md`).
5. Tune and compare models on the supplied student-development trips; keep the invigilator partition reserved for final evaluation.
6. Implement and measure HMM/Viterbi decoding against classifier-only predictions.
7. Build the API and evaluator UI with a server-side prediction boundary.
8. Verify two-laptop deployment and prepare a reproducible technical report.

The Bayesian search optimizes Random Forest settings using only trip-grouped folds from `train`; its full trial record is retained. The evaluator-testing partition is reserved for the final invigilated evaluation. No evaluator data or labels are needed to run these commands.
