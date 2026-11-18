import os
from pathlib import Path
import pandas as pd
import numpy as np

def get_dataset_root():
    """
    Locates the base data directory containing 'train' and 'student_test'.
    """
    candidates = [
        Path(r"c:\Users\manav\Documents\manav\PES\sem 5\ML\hack\Student folder\Student folder"),
        Path(r"c:\Users\manav\Documents\manav\PES\sem 5\ML\hack\Student folder"),
        Path(__file__).resolve().parent.parent / "Student folder" / "Student folder",
        Path(__file__).resolve().parent.parent / "Student folder"
    ]
    for c in candidates:
        if (c / "train").exists():
            return c
    raise FileNotFoundError("Could not find dataset root containing 'train' directory.")

def list_trips(partition="student_test"):
    """
    Returns list of trip IDs available in a partition ('train' or 'student_test').
    """
    root = get_dataset_root()
    p_dir = root / partition
    if not p_dir.exists():
        return []
    trips = [d.name for d in p_dir.iterdir() if d.is_dir()]
    return sorted(trips)

def load_trip_data(trip_id, partition=None):
    """
    Loads .track, .nodes, .arcs, and .route (if present) for a given trip.
    Returns:
        dict containing:
            'track': DataFrame(lon, lat, timestamp)
            'nodes': DataFrame(node_id, lon, lat)
            'arcs': DataFrame(arc_id, from_node, to_node)
            'route': list of int arc_ids (or None)
            'partition': str
    """
    root = get_dataset_root()
    
    # Locate partition if not provided
    if partition is None:
        if (root / "student_test" / trip_id).exists():
            partition = "student_test"
        elif (root / "train" / trip_id).exists():
            partition = "train"
        else:
            raise FileNotFoundError(f"Trip {trip_id} not found in train or student_test.")
            
    trip_dir = root / partition / trip_id
    
    # 1. Track: lon, lat, timestamp
    track_path = trip_dir / f"{trip_id}.track"
    df_track = pd.read_csv(
        track_path, 
        sep=r"\s+", 
        header=None, 
        names=["lon", "lat", "timestamp"],
        dtype={"lon": np.float64, "lat": np.float64, "timestamp": np.float64}
    )
    
    # 2. Nodes: lon, lat (0-indexed line number is node_id)
    nodes_path = trip_dir / f"{trip_id}.nodes"
    df_nodes = pd.read_csv(
        nodes_path, 
        sep=r"\s+", 
        header=None, 
        names=["lon", "lat"],
        dtype={"lon": np.float64, "lat": np.float64}
    )
    df_nodes["node_id"] = np.arange(len(df_nodes), dtype=np.int32)
    
    # 3. Arcs: from_node, to_node (0-indexed line number is arc_id)
    arcs_path = trip_dir / f"{trip_id}.arcs"
    df_arcs = pd.read_csv(
        arcs_path, 
        sep=r"\s+", 
        header=None, 
        names=["from_node", "to_node"],
        dtype={"from_node": np.int32, "to_node": np.int32}
    )
    df_arcs["arc_id"] = np.arange(len(df_arcs), dtype=np.int32)
    
    # 4. Route: sequence of arc_ids
    route_path = trip_dir / f"{trip_id}.route"
    route_arcs = None
    if route_path.exists():
        df_route = pd.read_csv(route_path, sep=r"\s+", header=None, names=["arc_id"])
        route_arcs = df_route["arc_id"].tolist()
        
    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": df_track,
        "nodes": df_nodes,
        "arcs": df_arcs,
        "route": route_arcs
    }
