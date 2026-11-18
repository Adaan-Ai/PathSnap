import streamlit as st
import requests
import json
import time
import numpy as np
from streamlit_folium import st_folium

from map_renderer import create_trip_folium_map
from eda_view import render_phase1_eda_dashboard
from src.data_loader import load_trip_data, list_trips

# Configure page
st.set_page_config(
    page_title="PathSnap: Map-Matching & Route Reconstruction",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Hackathon Evaluation
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d2/PES_University_logo.png/220px-PES_University_logo.png", width=120)
    st.title("Deployment Config")
    st.caption("UE24CS352A Machine Learning")
    
    st.markdown("### 🖥️ Two-Laptop Deployment")
    api_url = st.text_input(
        "Backend API URL (Laptop 1)", 
        value="http://10.0.9.171:8000",
        help="Enter the IP of Laptop 1, e.g., http://10.0.9.171:8000"
    )
    
    # Connection Check
    col_conn, col_ping = st.columns([2, 1])
    is_online = False
    server_info = None
    with col_conn:
        if st.button("Test Connection"):
            try:
                t0 = time.perf_counter()
                resp = requests.get(f"{api_url}/health", timeout=3.0)
                latency = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Connected! ({latency:.0f}ms)")
                    st.caption(f"Model on Laptop 1: **{data.get('model', 'Active')}**")
                    is_online = True
                else:
                    st.error(f"Status: {resp.status_code}")
            except Exception as e:
                st.error(f"Server unreachable: {e}")
                
    st.divider()
    enable_hmm = st.toggle("Apply HMM + Viterbi Post-Processing", value=True, help="Refines Laptop 1's classifier-only predictions using road network topology.")
    
    st.divider()
    st.markdown("""
    **Architecture Responsibilities:**
    * **Laptop 1 (API Server):** Hosts Random Forest classifier, API endpoints (`/health`, `/trips`, `/predict`).
    * **Laptop 2 (Client UI):** Evaluator dashboard, Folium map rendering, HMM comparison, metrics calculation.
    """)

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Reconstructing realistic vehicle trajectories from noisy GPS tracks over road networks</div>', unsafe_allow_html=True)

main_tab1, main_tab2 = st.tabs([
    "🚀 Evaluator Demo: Route Reconstruction", 
    "📊 Preprocessing & Feature Engineering Showcase"
])

# ----------------- TAB 1: EVALUATOR DEMO -----------------
with main_tab1:
    col_sel, col_action = st.columns([3, 1])
    
    # Fetch available trips from Laptop 1 backend
    trips_list = ["00000001", "00000002", "00000003", "00000004", "00000005"]
    try:
        r = requests.get(f"{api_url}/trips", timeout=2.0)
        if r.status_code == 200:
            fetched = r.json().get("trips", [])
            if fetched:
                trips_list = fetched
    except Exception:
        pass
        
    with col_sel:
        selected_trip = st.selectbox(
            "Select Evaluation Trip for Map Matching:",
            options=trips_list,
            index=0,
            help="Evaluator can select any of the available evaluation trips."
        )
        
    with col_action:
        st.write("") # spacer
        st.write("")
        run_btn = st.button("⚡ Reconstruct Route", type="primary", use_container_width=True)
        
    if run_btn:
        with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 (http://10.0.9.171:8000/predict)..."):
            try:
                t_start = time.perf_counter()
                
                # Load local trip files for geometry rendering
                local_data = load_trip_data(selected_trip)
                nodes = local_data["nodes"]
                arcs = local_data["arcs"]
                track = local_data["track"]
                gt_route = local_data["route"]
                
                # Query Laptop 1
                response = None
                backend_used = False
                try:
                    response = requests.post(
                        f"{api_url}/predict",
                        json={"trip_id": selected_trip},
                        headers={"Content-Type": "application/json"},
                        timeout=15.0
                    )
                except Exception as net_err:
                    st.warning(f"Backend on Laptop 1 timed out or busy ({net_err}). Falling back to local high-speed inference engine.")
                    
                predicted_arcs = []
                model_name = "Random Forest (Laptop 1)"
                decoder_name = "classifier_only"
                
                if response and response.status_code == 200:
                    resp_data = response.json()
                    backend_used = True
                    model_name = resp_data.get("model", "Random Forest")
                    decoder_name = resp_data.get("decoder", "classifier_only")
                    for p in resp_data.get("predictions", []):
                        if "predicted_arc_id" in p:
                            predicted_arcs.append(int(p["predicted_arc_id"]))
                else:
                    # High-speed local fallback
                    from api.inference import inference_engine
                    local_res = inference_engine.process_trip(selected_trip)
                    st.session_state["last_result"] = local_res
                    backend_used = False
                    
                if backend_used:
                    # Process Laptop 1's predictions into geometry
                    arc_endpoints = np.column_stack([arcs["from_node"].values, arcs["to_node"].values])
                    node_lat = nodes["lat"].values
                    node_lon = nodes["lon"].values
                    
                    # Compute unique sequential arcs
                    reconstructed_route_arcs = []
                    for aid in predicted_arcs:
                        if not reconstructed_route_arcs or reconstructed_route_arcs[-1] != aid:
                            reconstructed_route_arcs.append(aid)
                            
                    # Route continuity
                    continuous_steps = 0
                    total_transitions = max(1, len(reconstructed_route_arcs) - 1)
                    for i in range(total_transitions):
                        aid1 = reconstructed_route_arcs[i]
                        aid2 = reconstructed_route_arcs[i+1]
                        if aid1 < len(arc_endpoints) and aid2 < len(arc_endpoints):
                            if arc_endpoints[aid1, 1] == arc_endpoints[aid2, 0]:
                                continuous_steps += 1
                    route_continuity = (continuous_steps / total_transitions) * 100.0 if total_transitions > 0 else 100.0
                    
                    # Top 1 Accuracy against ground truth
                    top1_acc = None
                    seq_f1 = None
                    if gt_route:
                        gt_set = set(gt_route)
                        correct = sum(1 for a in predicted_arcs if a in gt_set)
                        top1_acc = (correct / max(1, len(predicted_arcs))) * 100.0
                        
                        rec_set = set(reconstructed_route_arcs)
                        tp = len(rec_set.intersection(gt_set))
                        p_ = tp / max(1, len(rec_set))
                        r_ = tp / max(1, len(gt_set))
                        seq_f1 = (2 * p_ * r_ / max(1e-6, p_ + r_)) * 100.0
                        
                    reconstructed_lines = []
                    for aid in reconstructed_route_arcs:
                        if aid < len(arc_endpoints):
                            fn, tn = arc_endpoints[aid]
                            reconstructed_lines.append([
                                [float(node_lat[fn]), float(node_lon[fn])],
                                [float(node_lat[tn]), float(node_lon[tn])]
                            ])
                            
                    gt_lines = []
                    if gt_route:
                        for aid in gt_route:
                            if aid < len(arc_endpoints):
                                fn, tn = arc_endpoints[aid]
                                gt_lines.append([
                                    [float(node_lat[fn]), float(node_lon[fn])],
                                    [float(node_lat[tn]), float(node_lon[tn])]
                                ])
                                
                    gps_coords = [[float(lat), float(lon)] for lon, lat in zip(track["lon"].values, track["lat"].values)]
                    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                    
                    st.session_state["last_result"] = {
                        "trip_id": selected_trip,
                        "partition": local_data["partition"],
                        "num_observations": len(gps_coords),
                        "reconstructed_arc_count": len(reconstructed_route_arcs),
                        "top1_accuracy": round(top1_acc, 2) if top1_acc is not None else None,
                        "route_continuity": round(route_continuity, 2),
                        "sequence_f1": round(seq_f1, 2) if seq_f1 is not None else None,
                        "inference_time_ms": round(elapsed_ms, 2),
                        "gps_coordinates": gps_coords,
                        "reconstructed_route_lines": reconstructed_lines,
                        "ground_truth_route_lines": gt_lines,
                        "model_name": model_name,
                        "decoder": decoder_name,
                        "backend_laptop": "Laptop 1 (10.0.9.171)"
                    }
                    st.success(f"Successfully processed Trip {selected_trip} via Laptop 1 Random Forest model!")
                    
            except Exception as e:
                st.error(f"Error processing trip: {e}")
                
    # Display results if available in session state
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        
        st.divider()
        st.markdown(f"#### 📈 Evaluation Metrics for Trip `{res['trip_id']}`")
        if res.get("backend_laptop"):
            st.caption(f"Served by: **{res.get('backend_laptop')}** | Model: **{res.get('model_name', 'Random Forest')}** | Decoder: **{res.get('decoder', 'classifier_only')}**")
            
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Top-1 Observation Acc", f"{res.get('top1_accuracy', 'N/A')}%")
        with m_col2:
            st.metric("Route Continuity", f"{res.get('route_continuity', 'N/A')}%")
        with m_col3:
            st.metric("Sequence F1", f"{res.get('sequence_f1', 'N/A')}%")
        with m_col4:
            st.metric("Total Latency", f"{res.get('inference_time_ms', 0):.0f} ms")
        with m_col5:
            st.metric("Total GPS Observations", f"{res.get('num_observations', 0)}")
            
        st.divider()
        
        # Interactive Folium Map
        st.markdown("#### 🗺️ Interactive Trajectory Visualizer")
        st.caption("Comparing Raw GPS Observations (Blue Dots), Ground Truth (.route, Green), and Reconstructed Trajectory (Red)")
        
        folium_map = create_trip_folium_map(res)
        st_folium(folium_map, width="100%", height=550)
        
    else:
        st.info("👆 Select any evaluation trip above and click **Reconstruct Route** to query Laptop 1 and visualize the route.")

# ----------------- TAB 2: PRE-EVAL SHOWCASE -----------------
with main_tab2:
    render_phase1_eda_dashboard()
