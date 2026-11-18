
# PathSnap API Contract

## 1. Health Check

GET /health

Response:
{
  "status": "running"
}

## 2. Get Evaluation Trips

GET /trips

Response:
{
  "trips": ["actual_trip_id_1", "actual_trip_id_2"]
}

Only permitted evaluator trip IDs should be returned.

## 3. Predict Route

POST /predict

Request:
{
  "trip_id": "actual_trip_id"
}

Response:
{
  "trip_id": "actual_trip_id",
  "predicted_route": [],
  "metrics": {}
}

The predicted route must use an agreed representation,
such as road-network arc IDs or geographic coordinates.

Evaluation must happen server-side.

Ground-truth labels must never be returned to the UI.
