import streamlit as st
import requests
import json
import time
from streamlit_folium import st_folium

from map_renderer import create_trip_folium_map
from eda_view import render_phase1_eda_dashboard

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
        value="http://localhost:8000",
        help="Enter the IP of Laptop 1, e.g., http://192.168.1.150:8000"
    )
    
    # Connection Check
    col_conn, col_ping = st.columns([2, 1])
    is_online = False
    with col_conn:
        if st.button("Test Connection"):
            try:
                t0 = time.perf_counter()
                resp = requests.get(f"{api_url}/health", timeout=2.0)
                latency = (time.perf_counter() - t0) * 1000.0
                if resp.status_code == 200:
                    st.success(f"Connected ({latency:.0f}ms)")
                    is_online = True
                else:
                    st.error(f"Status: {resp.status_code}")
            except Exception as e:
                st.error("Server unreachable")
    
    st.divider()
    st.markdown("""
    **Architecture Responsibilities:**
    * **Laptop 1 (API Server):** Model weights, candidate classifier, HMM/Viterbi engine, terminal logs.
    * **Laptop 2 (Client UI):** Evaluator interface, Folium map rendering, metrics review.
    """)

# ----------------- MAIN VIEW -----------------
st.markdown('<div class="main-title">PathSnap: Distributed Map-Matching & Route Reconstruction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Reconstructing realistic vehicle trajectories from noisy GPS tracks over road networks</div>', unsafe_allow_html=True)

main_tab1, main_tab2 = st.tabs([
    "🚀 Evaluator Demo: Route Reconstruction", 
    "📊 Phase 1 Showcase (11:30 AM Evaluation)"
])

# ----------------- TAB 1: EVALUATOR DEMO -----------------
with main_tab1:
    col_sel, col_action = st.columns([3, 1])
    
    # Try fetching available trips from backend
    trips_list = [
        "00000000", "00000007", "00000009", "00000040", 
        "00000052", "00000056", "00000065", "00000076", 
        "00000077", "00000091"
    ]
    try:
        r = requests.get(f"{api_url}/api/trips", timeout=1.5)
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
            help="Evaluator can select any of the 10 reserved evaluation trips."
        )
        
    with col_action:
        st.write("") # spacer
        st.write("")
        run_btn = st.button("⚡ Reconstruct Route", type="primary", use_container_width=True)
        
    if run_btn:
        with st.spinner(f"Sending Trip {selected_trip} to Laptop 1 API for map matching and Viterbi decoding..."):
            try:
                t_start = time.perf_counter()
                response = requests.post(
                    f"{api_url}/api/predict/{selected_trip}",
                    timeout=30.0
                )
                roundtrip_ms = (time.perf_counter() - t_start) * 1000.0
                
                if response.status_code == 200:
                    data = response.json()
                    st.session_state["last_result"] = data
                    st.session_state["roundtrip_ms"] = roundtrip_ms
                    st.success(f"Trip {selected_trip} successfully processed by Laptop 1!")
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Failed to communicate with API server at {api_url}: {e}")
                
    # Display results if available in session state
    if "last_result" in st.session_state:
        res = st.session_state["last_result"]
        
        st.divider()
        st.markdown(f"#### 📈 Evaluation Metrics for Trip `{res['trip_id']}`")
        
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        with m_col1:
            st.metric("Top-1 Observation Acc", f"{res.get('top1_accuracy', 'N/A')}%")
        with m_col2:
            st.metric("Route Continuity", f"{res.get('route_continuity', 'N/A')}%")
        with m_col3:
            st.metric("Sequence F1", f"{res.get('sequence_f1', 'N/A')}%")
        with m_col4:
            st.metric("Inference Latency", f"{res.get('inference_time_ms', 0):.1f} ms")
        with m_col5:
            st.metric("Total GPS Points", f"{res.get('num_observations', 0)}")
            
        st.divider()
        
        # Interactive Folium Map
        st.markdown("#### 🗺️ Interactive Trajectory Visualizer")
        st.caption("Comparing Raw GPS Observations (Blue Dots), Ground Truth (.route, Green), and Reconstructed Trajectory (Red)")
        
        folium_map = create_trip_folium_map(res)
        st_folium(folium_map, width="100%", height=550)
        
    else:
        st.info("👆 Select any evaluation trip above and click **Reconstruct Route** to run inference on Laptop 1 and visualize the route.")

# ----------------- TAB 2: 11:30 AM PRE-EVAL SHOWCASE -----------------
with main_tab2:
    render_phase1_eda_dashboard()
