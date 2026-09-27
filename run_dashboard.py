"""
Phenox Model Recovery - Dashboard Runner
Starts the Uvicorn web server hosting the FastAPI backend and MLOps Dashboard UI.
"""

import os
import sys
import webbrowser

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import uvicorn

def main():
    port = 8000
    host = "127.0.0.1"
    url = f"http://{host}:{port}"
    
    print("=" * 65)
    print("  PHENOX MODEL RECOVERY - MLOps DASHBOARD")
    print("=" * 65)
    print(f"-> Starting Dashboard Server at: {url}")
    print(f"-> Swagger API Docs available at: {url}/docs")
    print("-> Press Ctrl+C to stop the server.")
    print("=" * 65)
    
    uvicorn.run("api:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    main()
