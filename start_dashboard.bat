@echo off
start "" python run_dashboard.py
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000