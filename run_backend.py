# run_backend.py
import uvicorn
import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

if __name__ == "__main__":
    uvicorn.run(
        "web.backend.main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )