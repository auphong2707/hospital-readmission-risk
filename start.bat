@echo off
REM Start the Hospital Readmission Risk Dashboard

echo Starting Hospital Readmission Risk Dashboard...
echo.

REM Change to the data-api directory
cd phase-7-stakeholder-dashboards\data-api

REM Run the FastAPI server
echo Starting FastAPI server on http://localhost:8000
echo.
echo Dashboards available at:
echo   - Data Analyst: http://localhost:8000/dashboards/data-analyst
echo   - Doctor: http://localhost:8000/dashboards/doctor
echo   - API Docs: http://localhost:8000/docs
echo.

python main.py
