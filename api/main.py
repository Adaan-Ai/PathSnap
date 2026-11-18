import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.data_loader import list_trips
from api.inference import inference_engine

# Configure detailed terminal logging for evaluators on Laptop 1
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PathSnap API] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("pathsnap_api")

app = FastAPI(
    title="PathSnap: Map-Matching & Route Reconstruction API",
    description="Distributed Backend for Vehicle Trajectory Reconstruction (Laptop 1)",
    version="1.0.0"
)

# Enable CORS for cross-device requests (Laptop 2 -> Laptop 1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "PathSnap Distributed Inference Server",
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time()
    }

@app.get("/api/trips")
def get_trips():
    """
    Returns list of evaluation trips available for demonstration.
    """
    eval_trips = list_trips("student_test")
    logger.info(f"Retrieved {len(eval_trips)} evaluation trips: {eval_trips}")
    return {
        "trips": eval_trips,
        "count": len(eval_trips),
        "partition": "student_test"
    }

@app.post("/api/predict/{trip_id}")
def predict_trip(trip_id: str):
    """
    Processes evaluation trip:
    1. Extracts GPS observations & candidate road segments
    2. Runs classifier emission + HMM Viterbi route decoding
    3. Computes server-side accuracy against ground truth .route
    4. Returns route coordinates & metrics to UI (Laptop 2)
    """
    logger.info(f">>> [INFERENCE REQUEST] Received evaluation request for Trip ID: {trip_id} from client.")
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
    except FileNotFoundError:
        logger.error(f"Trip ID '{trip_id}' not found in dataset partitions.")
        raise HTTPException(status_code=404, detail=f"Trip ID {trip_id} not found.")
    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/eda/stats")
def get_eda_stats():
    """
    Summary statistics for the 11:30 AM Pre-Evaluation Phase.
    Summarizes the 80 training trips, sampling intervals, and coordinate bounds.
    """
    train_trips = list_trips("train")
    test_trips = list_trips("student_test")
    
    return {
        "train_trips_count": len(train_trips),
        "student_test_trips_count": len(test_trips),
        "total_trips": len(train_trips) + len(test_trips),
        "sampling_rate_hz": 1.0,
        "nominal_step_pct": 97.2,
        "coordinate_bounds": {
            "min_lon": 8.54,
            "max_lon": 8.65,
            "min_lat": 49.80,
            "max_lat": 49.92,
            "center": [49.85, 8.58]
        },
        "features_engineered": [
            {"name": "Cross-Track Distance (d_perp)", "description": "Perpendicular distance from GPS to candidate arc segment (meters)"},
            {"name": "Angular Heading Discrepancy", "description": "Absolute difference between GPS movement heading and road direction (degrees)"},
            {"name": "Normalized Projection Factor (t)", "description": "Position along arc segment [0, 1]"},
            {"name": "Instantaneous Speed Ratio", "description": "Ratio of GPS observation speed to segment traversal capability"}
        ]
    }
