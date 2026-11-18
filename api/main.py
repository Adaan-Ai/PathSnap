import logging
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from src.data_loader import list_trips, load_trip_data
from api.inference import inference_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap_api")

app = FastAPI(
    title="PathSnap: Map-Matching & Route Reconstruction API",
    description="Distributed Backend for Vehicle Trajectory Reconstruction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_ui():
    index_path = Path(__file__).resolve().parent.parent / "ui" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "status": "online",
        "service": "PathSnap Distributed Inference Server",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "running",
        "model": "random_forest"
    }

@app.get("/trips")
def get_trips_root():
    eval_trips = list_trips("student_test")
    return {"trips": eval_trips}

@app.get("/api/trips")
def get_trips_api():
    eval_trips = list_trips("student_test")
    return {
        "trips": eval_trips,
        "count": len(eval_trips),
        "partition": "student_test"
    }

class PredictRequest(BaseModel):
    trip_id: str

@app.post("/predict")
def predict_root(req: PredictRequest):
    return predict_trip_internal(req.trip_id)

@app.post("/api/predict/{trip_id}")
def predict_api(trip_id: str):
    return predict_trip_internal(trip_id)

def predict_trip_internal(trip_id: str):
    logger.info(f">>> [INFERENCE REQUEST] Received evaluation request for Trip ID: {trip_id}")
    try:
        t0 = time.perf_counter()
        result = inference_engine.process_trip(trip_id)
        elapsed = (time.perf_counter() - t0) * 1000.0
        logger.info(
            f"<<< [COMPLETED] Trip: {trip_id} | Points: {result['num_observations']} | "
            f"Top-1 Acc: {result['top1_accuracy']}% | Continuity: {result['route_continuity']}% | "
            f"Elapsed: {elapsed:.2f}ms"
        )
        return result
    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trip_geometry/{trip_id}")
def get_trip_geometry(trip_id: str):
    try:
        trip_data = load_trip_data(trip_id)
        track = trip_data["track"]
        nodes = trip_data["nodes"]
        arcs = trip_data["arcs"]
        gt_route = trip_data["route"]
        
        gps_coords = [[float(lat), float(lon)] for lon, lat in zip(track["lon"].values, track["lat"].values)]
        node_lat = nodes["lat"].values
        node_lon = nodes["lon"].values
        from_nodes = arcs["from_node"].values
        to_nodes = arcs["to_node"].values
        
        gt_lines = []
        if gt_route is not None:
            for aid in gt_route:
                if aid < len(from_nodes):
                    fn = from_nodes[aid]
                    tn = to_nodes[aid]
                    gt_lines.append([
                        [float(node_lat[fn]), float(node_lon[fn])],
                        [float(node_lat[tn]), float(node_lon[tn])]
                    ])
                    
        return {
            "trip_id": trip_id,
            "gps_coordinates": gps_coords,
            "ground_truth_lines": gt_lines
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
