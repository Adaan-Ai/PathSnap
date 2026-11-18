from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.services.prediction import predict_trip, available_trips
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(
    title="PathSnap API",
    description="ML-based GPS map matching and route reconstruction",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TripRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "running", "model": "random_forest"}


@app.get("/trips")
def get_trips():
    try:
        return {"trips": available_trips()}
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))


@app.post("/predict")
def predict(request: TripRequest):
    try:
        return predict_trip(request.trip_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
