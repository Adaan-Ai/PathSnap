from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="PathSnap API")


class TripRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "running"}


@app.get("/trips")
def get_trips():
    # Return only the permitted evaluator trip IDs.
    return {"trips": ["trip_01", "trip_02"]}


@app.post("/predict")
def predict(request: TripRequest):
    # Replace this with your actual prediction pipeline.
    # It must load GPS observations, extract features,
    # predict candidates and apply HMM/Viterbi.

    try:
        # Example:
        # result = predict_trip(request.trip_id)

        return {
            "trip_id": request.trip_id,
            "predicted_route": [],
            "metrics": {}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
