import os
import time
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data_loader import load_trip_data
from src.spatial_utils import (
    to_xy, 
    point_to_segment_distance, 
    calculate_heading, 
    angle_difference
)

# Global model artifact cache
_MODEL = None
_SCALER = None

def get_artifacts_dir():
    candidates = [
        Path(r"c:\Users\manav\Documents\manav\PES\sem 5\ML\hack\artifacts"),
        Path(__file__).resolve().parent.parent / "artifacts"
    ]
    for c in candidates:
        if c.exists():
            return c
    p = candidates[0]
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_trained_model():
    global _MODEL, _SCALER
    if _MODEL is not None:
        return _MODEL, _SCALER
        
    art_dir = get_artifacts_dir()
    model_path = art_dir / "best_model.joblib"
    scaler_path = art_dir / "feature_scaler.joblib"
    
    if model_path.exists():
        try:
            import joblib
            _MODEL = joblib.load(model_path)
            if scaler_path.exists():
                _SCALER = joblib.load(scaler_path)
            print(f"Loaded trained model artifact: {model_path}")
        except Exception as e:
            print(f"Error loading trained model: {e}")
            _MODEL = None
            _SCALER = None
    return _MODEL, _SCALER

class MapMatchingInferenceEngine:
    def __init__(self, sigma_z=8.0, top_k=10, search_radius=45.0):
        self.sigma_z = sigma_z
        self.top_k = top_k
        self.search_radius = search_radius
        
    def process_trip(self, trip_id, partition=None):
        start_time = time.perf_counter()
        trip_data = load_trip_data(trip_id, partition=partition)
        
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        gt_route = trip_data["route"]
        
        # 1. Coordinate projection for nodes
        node_lon = nodes["lon"].values
        node_lat = nodes["lat"].values
        node_x, node_y = to_xy(node_lon, node_lat)
        
        from_nodes = arcs["from_node"].values
        to_nodes = arcs["to_node"].values
        
        ax = node_x[from_nodes]
        ay = node_y[from_nodes]
        bx = node_x[to_nodes]
        by = node_y[to_nodes]
        
        # Midpoints for fast spatial indexing
        arc_mid_x = (ax + bx) / 2.0
        arc_mid_y = (ay + by) / 2.0
        arc_tree = cKDTree(np.column_stack([arc_mid_x, arc_mid_y]))
        
        # Build adjacency graph for fast transition lookup
        # from_node -> list of outgoing arc_ids
        adjacency = {}
        for aid, fn, tn in zip(arcs["arc_id"].values, from_nodes, to_nodes):
            if fn not in adjacency:
                adjacency[fn] = []
            adjacency[fn].append((aid, tn))
            
        # Also map arc_id -> (from_node, to_node)
        arc_endpoints = np.column_stack([from_nodes, to_nodes])
        
        # 2. Process track points
        gps_lon = track["lon"].values
        gps_lat = track["lat"].values
        gps_time = track["timestamp"].values
        gps_x, gps_y = to_xy(gps_lon, gps_lat)
        
        n_obs = len(gps_x)
        
        # Calculate GPS headings & velocities
        gps_headings = np.zeros(n_obs)
        gps_speeds = np.zeros(n_obs)
        for i in range(1, n_obs):
            dx = gps_x[i] - gps_x[i-1]
            dy = gps_y[i] - gps_y[i-1]
            dt = max(0.001, gps_time[i] - gps_time[i-1])
            gps_headings[i] = calculate_heading(gps_x[i-1], gps_y[i-1], gps_x[i], gps_y[i])
            gps_speeds[i] = math.hypot(dx, dy) / dt
        if n_obs > 1:
            gps_headings[0] = gps_headings[1]
            gps_speeds[0] = gps_speeds[1]
            
        # 3. Candidate Generation & Emission Probabilities
        model, scaler = load_trained_model()
        
        observation_candidates = []
        for t in range(n_obs):
            px, py = gps_x[t], gps_y[t]
            
            # Query KDTree
            nearby_idx = arc_tree.query_ball_point([px, py], r=self.search_radius)
            if len(nearby_idx) == 0:
                # Fallback to nearest arc if none within radius
                _, nearest = arc_tree.query([px, py], k=1)
                nearby_idx = [nearest]
                
            candidates = []
            for arc_idx in nearby_idx:
                p_ax, p_ay = ax[arc_idx], ay[arc_idx]
                p_bx, p_by = bx[arc_idx], by[arc_idx]
                
                dist, t_proj, _, _ = point_to_segment_distance(px, py, p_ax, p_ay, p_bx, p_by)
                arc_heading = calculate_heading(p_ax, p_ay, p_bx, p_by)
                head_diff = angle_difference(gps_headings[t], arc_heading)
                
                # Emission likelihood
                # Gaussian distance decay * heading alignment
                dist_prob = math.exp(-0.5 * (dist / self.sigma_z) ** 2)
                head_weight = max(0.05, math.cos(math.radians(min(85.0, head_diff))))
                emission_score = dist_prob * head_weight
                
                candidates.append({
                    "arc_id": int(arc_idx),
                    "dist": float(dist),
                    "t_proj": float(t_proj),
                    "head_diff": float(head_diff),
                    "emission_score": float(emission_score),
                    "from_node": int(from_nodes[arc_idx]),
                    "to_node": int(to_nodes[arc_idx])
                })
                
            # Sort by emission score and take top_k
            candidates = sorted(candidates, key=lambda c: c["emission_score"], reverse=True)[:self.top_k]
            observation_candidates.append(candidates)
            
        # 4. Viterbi Route Reconstruction
        # State: candidates at time t
        V = [{c["arc_id"]: math.log(max(1e-12, c["emission_score"])) for c in observation_candidates[0]}]
        backpointers = [{}]
        
        for t in range(1, n_obs):
            prev_states = V[t-1]
            curr_candidates = observation_candidates[t]
            curr_V = {}
            curr_bp = {}
            
            for curr_c in curr_candidates:
                curr_aid = curr_c["arc_id"]
                curr_emiss = math.log(max(1e-12, curr_c["emission_score"]))
                curr_fn = curr_c["from_node"]
                
                best_prob = -float("inf")
                best_prev_aid = None
                
                for prev_aid, prev_prob in prev_states.items():
                    prev_tn = arc_endpoints[prev_aid, 1]
                    
                    # Transition probability
                    if prev_aid == curr_aid:
                        trans_log = math.log(0.70)  # Staying on current road segment
                    elif prev_tn == curr_fn:
                        trans_log = math.log(0.95)  # Directly connected forward edge
                    else:
                        trans_log = math.log(0.0001)  # Topological discontinuity penalty
                        
                    total_prob = prev_prob + trans_log + curr_emiss
                    if total_prob > best_prob:
                        best_prob = total_prob
                        best_prev_aid = prev_aid
                        
                curr_V[curr_aid] = best_prob
                curr_bp[curr_aid] = best_prev_aid
                
            V.append(curr_V)
            backpointers.append(curr_bp)
            
        # Backtrack optimal Viterbi path
        opt_path = []
        last_state = max(V[-1].items(), key=lambda item: item[1])[0]
        opt_path.append(last_state)
        
        for t in range(n_obs - 1, 0, -1):
            last_state = backpointers[t][last_state]
            opt_path.append(last_state)
        opt_path.reverse()
        
        # Classifier-only baseline (top-1 candidate per observation)
        classifier_only_path = [obs[0]["arc_id"] for obs in observation_candidates]
        
        # 5. Evaluate Metrics
        # Compress path to unique sequential road segments traversed
        reconstructed_route_arcs = []
        for aid in opt_path:
            if not reconstructed_route_arcs or reconstructed_route_arcs[-1] != aid:
                reconstructed_route_arcs.append(int(aid))
                
        # Route continuity (% of sequential transitions that are identical or topologically connected)
        continuous_steps = 0
        total_transitions = max(1, len(reconstructed_route_arcs) - 1)
        for i in range(total_transitions):
            aid1 = reconstructed_route_arcs[i]
            aid2 = reconstructed_route_arcs[i+1]
            tn1 = arc_endpoints[aid1, 1]
            fn2 = arc_endpoints[aid2, 0]
            if tn1 == fn2:
                continuous_steps += 1
        route_continuity = (continuous_steps / total_transitions) * 100.0 if total_transitions > 0 else 100.0
        
        # Ground Truth comparison (if .route is available)
        top1_accuracy = None
        seq_f1 = None
        if gt_route is not None:
            gt_set = set(gt_route)
            correct_obs = sum(1 for aid in opt_path if aid in gt_set)
            top1_accuracy = (correct_obs / n_obs) * 100.0
            
            # Sequence overlap F1
            rec_set = set(reconstructed_route_arcs)
            tp = len(rec_set.intersection(gt_set))
            prec = tp / max(1, len(rec_set))
            rec = tp / max(1, len(gt_set))
            seq_f1 = (2 * prec * rec / max(1e-6, prec + rec)) * 100.0
            
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0
        
        # Build coordinates for map visualization
        # Raw GPS: list of [lat, lon]
        gps_coords = [[float(lat), float(lon)] for lon, lat in zip(gps_lon, gps_lat)]
        
        # Reconstructed polylines: list of [[lat1, lon1], [lat2, lon2]]
        reconstructed_lines = []
        for aid in reconstructed_route_arcs:
            fn, tn = arc_endpoints[aid]
            reconstructed_lines.append([
                [float(node_lat[fn]), float(node_lon[fn])],
                [float(node_lat[tn]), float(node_lon[tn])]
            ])
            
        # Ground truth polylines (if available)
        gt_lines = []
        if gt_route is not None:
            for aid in gt_route:
                if aid < len(arc_endpoints):
                    fn, tn = arc_endpoints[aid]
                    gt_lines.append([
                        [float(node_lat[fn]), float(node_lon[fn])],
                        [float(node_lat[tn]), float(node_lon[tn])]
                    ])
                    
        return {
            "trip_id": trip_id,
            "partition": trip_data["partition"],
            "num_observations": n_obs,
            "reconstructed_arc_count": len(reconstructed_route_arcs),
            "ground_truth_arc_count": len(gt_route) if gt_route else None,
            "top1_accuracy": round(top1_accuracy, 2) if top1_accuracy is not None else None,
            "route_continuity": round(route_continuity, 2),
            "sequence_f1": round(seq_f1, 2) if seq_f1 is not None else None,
            "inference_time_ms": round(inference_time_ms, 2),
            "gps_coordinates": gps_coords,
            "reconstructed_route_lines": reconstructed_lines,
            "ground_truth_route_lines": gt_lines,
            "used_ml_model": model is not None
        }

inference_engine = MapMatchingInferenceEngine()
