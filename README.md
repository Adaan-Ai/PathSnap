# PathSnap: Distributed Map-Matching & Route Reconstruction

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

**PathSnap** is a distributed machine learning system that reconstructs realistic vehicle trajectories by matching noisy GPS observations to a road network. Developed for the PES University UE24CS352A Machine Learning Hackathon.

---

## 🏗️ System Architecture

PathSnap is designed around a **Two-Laptop Distributed Deployment**:

* **Laptop 1 (Model & API Server):**
  * Hosts the trained candidate classifier, spatial index, and Hidden Markov Model (HMM) Viterbi decoder.
  * Serves inference requests via **FastAPI** (`0.0.0.0:8000`).
  * Computes server-side accuracy against ground-truth routes while isolating test labels from the client.
  * Outputs live request logs in the terminal for evaluator verification.

* **Laptop 2 (Evaluator UI):**
  * Evaluator-facing dashboard built with **Streamlit** (`port 8501`).
  * Features a dropdown selector for the 10 reserved evaluation trips.
  * Renders interactive **Folium** maps displaying raw GPS points, ground-truth routes, and reconstructed paths with layer controls.
  * Displays real-time evaluation metrics: Top-1 Accuracy, Route Continuity, Sequence F1, and Inference Latency.

---

## 📁 Repository Structure

```
├── api/
│   ├── main.py             # FastAPI server with terminal logging
│   └── inference.py        # Map-matching engine & Viterbi route decoder
├── ui/
│   ├── app.py              # Streamlit evaluator dashboard
│   ├── map_renderer.py     # Folium multi-layer map generator
│   └── eda_view.py         # Phase 1 Preprocessing & Feature Engineering showcase
├── src/
│   ├── spatial_utils.py    # Metric 2D projection, Haversine, distance & heading math
│   └── data_loader.py      # Dataset loader for .track, .nodes, .arcs, and .route
├── artifacts/              # Directory for trained model weights & scalers
├── run_api.bat             # One-click Windows launcher for Laptop 1
├── run_ui.bat              # One-click Windows launcher for Laptop 2
├── requirements.txt        # Pinned dependencies
└── README.md               # Project documentation
```

---

## 🚀 Setup & Execution Instructions

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone <repo-url>
cd hack
pip install -r requirements.txt
```

### 2. Starting Laptop 1 (API Server)
Run the FastAPI backend:
```bash
# Using batch file:
run_api.bat

# Or directly:
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 3. Starting Laptop 2 (Evaluator UI)
Connect Laptop 2 to the same network/hotspot as Laptop 1, then run:
```bash
# Using batch file:
run_ui.bat

# Or directly:
streamlit run ui/app.py
```
In the UI sidebar, enter Laptop 1's IP address (e.g., `http://192.168.1.150:8000`) and click **Test Connection**.

---

## 🔬 Methodology Highlights

1. **Local Metric Coordinate Projection:**
   * Angular coordinates $(lon, lat)$ are projected to a local 2D metric Cartesian plane $(x, y)$ in meters to avoid Euclidean distance distortion across degrees.
2. **Candidate Generation:**
   * Arc midpoints are indexed using a `scipy.spatial.cKDTree`. For each GPS observation, candidates within a 45m radius ($K=10$) are extracted.
3. **Feature Engineering:**
   * Perpendicular cross-track distance ($d_\perp$).
   * Angular heading discrepancy ($|\theta_{\text{GPS}} - \theta_{\text{arc}}|$).
   * Normalized projection factor ($t \in [0, 1]$).
   * Instantaneous speed consistency.
4. **HMM & Viterbi Sequential Decoding:**
   * Emission probabilities model spatial and directional fit.
   * Transition probabilities enforce road network topology from `.arcs` to prevent unrealistic teleportation across buildings.
