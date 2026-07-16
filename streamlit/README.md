# Streamlit Frontend (Legacy)

This is the **legacy Streamlit-based frontend** for the AI Research Paper Analyzer. It communicates with the same FastAPI backend as the primary React frontend.

## Quick Start

```bash
# Ensure the FastAPI backend is running first
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** to use the Streamlit interface.

## Configuration

Set the `API_URL` environment variable if the backend is not running on `localhost:8000`:

```bash
API_URL=https://your-backend-url.com streamlit run app.py
```

## Notes

- This frontend is kept for personal/fallback use
- The primary frontend is the React (Vite) app in `frontend/`
- Both frontends call the same FastAPI backend REST API
- No backend logic is duplicated — all processing happens server-side
