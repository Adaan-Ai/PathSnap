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
python src/pathsnap/eda.py `
  --dataset "C:\path\to\train" `
  --output reports/eda
$env:PYTHONPATH = "src"
python src/pathsnap/eda_visualizations.py `
  --dataset "C:\path\to\train" `
  --inventory reports/eda/training_trip_inventory.csv `
  --output reports/eda/visualizations
```

The EDA output is derived from the supplied training partition. It includes a JSON aggregate summary and per-trip CSV inventory. Visualizations include training-set size distributions and a Folium map of one real training trip's GPS observations and road network.

Per-trip inventories and generated maps stay local: they include precise GPS locations from the provided dataset and are ignored by Git. The map-generation code is included so the team can recreate them from its local data.

## Planned phases

1. Dataset exploration and validation.
2. Establish label semantics and trip-level splits; implement preprocessing and candidate generation (in progress; observation-level label alignment remains open).
3. Build and inspect leakage-safe features.
4. Compare five distinct models using trip-grouped cross-validation; keep optimization trials and measurements.
5. Tune the selected model on training trips and compare on permitted student-development trips if supplied.
6. Implement and measure HMM/Viterbi decoding against classifier-only predictions.
7. Build the API and evaluator UI with a server-side prediction boundary.
8. Verify two-laptop deployment and prepare a reproducible technical report.

The evaluator-testing partition is reserved for the final invigilated evaluation. No evaluator data or labels are needed to run the training EDA command.
