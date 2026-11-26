@echo off
echo ========================================================
echo DeepTruth AI - Fake / AI-Generated Image Detector
echo Subject: Intelligent Model Design Using AI (Sem 5)
echo Bennett University Project Showcase
echo ========================================================
echo.
echo Starting FastAPI Web Server on http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
