"""
run.py — Single-command launcher for both FastAPI backend and Streamlit frontend.

Usage:
    python run.py          → Starts both servers
    python run.py --api    → Start only the FastAPI backend
    python run.py --ui     → Start only the Streamlit frontend

The backend runs on port 8000, the frontend on port 8501.
"""

import subprocess
import sys
import time
import argparse


def start_api():
    """Start FastAPI backend on port 8000."""
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.server:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=".",
    )


def start_ui():
    """Start Streamlit frontend on port 8501."""
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=".",
    )


def main():
    parser = argparse.ArgumentParser(description="Launch RAG Research Paper Analyzer")
    parser.add_argument("--api", action="store_true", help="Start only the FastAPI backend")
    parser.add_argument("--ui", action="store_true", help="Start only the Streamlit frontend")
    args = parser.parse_args()

    processes = []

    try:
        if args.api:
            print("[*] Starting FastAPI backend on http://localhost:8000")
            print("[*] Swagger docs at http://localhost:8000/docs")
            processes.append(start_api())
        elif args.ui:
            print("[*] Starting Streamlit frontend on http://localhost:8501")
            processes.append(start_ui())
        else:
            # Start both
            print("[*] Starting FastAPI backend on http://localhost:8000")
            print("[*] Swagger docs at http://localhost:8000/docs")
            api_proc = start_api()
            processes.append(api_proc)

            time.sleep(3)  # Give API time to start

            print("[*] Starting Streamlit frontend on http://localhost:8501")
            ui_proc = start_ui()
            processes.append(ui_proc)

            print("\n" + "=" * 50)
            print("  Both servers are running!")
            print("  API:     http://localhost:8000")
            print("  Swagger: http://localhost:8000/docs")
            print("  UI:      http://localhost:8501")
            print("  Press Ctrl+C to stop both.")
            print("=" * 50 + "\n")

        # Wait for processes
        for p in processes:
            p.wait()

    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
        for p in processes:
            p.terminate()
        print("[*] Done.")


if __name__ == "__main__":
    main()
