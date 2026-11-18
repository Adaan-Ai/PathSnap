@echo off
echo Starting PathSnap Evaluator Web Interface on http://localhost:8000/ ...
start http://localhost:8000/
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
pause
