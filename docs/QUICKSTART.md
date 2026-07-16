# 🚀 Quick Start Guide: Running ResearchLens

This guide provides step-by-step instructions for running the **ResearchLens** AI platform locally, via Docker, or in production.

---

## 🏗️ Local Development (Manual Setup)

For the best development experience with hot-reloading.

### 1. Prerequisites
- **Python 3.11+** installed.
- **Node.js 18+** installed.
- A **Groq API Key** (Free tier works perfectly).

### 2. Backend Setup (Terminal 1)
```bash
# Navigate to the root directory
cd Rag_final

# Create and activate a virtual environment (CRITICAL)
python -m venv venv
venv\Scripts\activate
source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt

# Start the FastAPI server
python -m uvicorn backend.api.server:app --reload --port 8000
```
*   **Health Check:** Open `http://localhost:8000/health`
*   **API Docs:** Open `http://localhost:8000/docs`

### 3. Frontend Setup (Terminal 2)
```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the Vite development server
npm run dev
```
*   **Access UI:** Open `http://localhost:5173`

---

## 🐳 Docker Setup (One-Click)

The simplest way to run the entire stack without installing dependencies manually.

```bash
cd Rag_final/deployment
docker-compose up --build
```
- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8000

---

## 🔑 API Key Configuration

ResearchLens offers two ways to manage your credentials:

1.  **Environment File (.env):** Create a `.env` file in the root directory (copy from `.env.example`).
2.  **Runtime UI Config:** You can enter your **Groq** and **Pinecone** keys directly on the "Setup" page in the browser. These are stored in memory and are never saved to disk.

---

## 🛠️ Troubleshooting & Common Fixes

| Issue | Cause | Solution |
|:---|:---|:---|
| **ModuleNotFoundError** | Python environment issue | Ensure you activated the `venv` before running `pip install` and `uvicorn`. |
| **API Offline** | Backend not running | Check Terminal 1. Ensure `uvicorn` is running on port 8000. |
| **500 Rate Limit Error** | Groq Free Tier limit | The free tier allows 100K tokens/day. Wait an hour or use a different key. |
| **Pinecone Errors** | Missing/Invalid Key | Ensure the key matches your environment (e.g., `us-east-1-aws`). |
| **Vite Proxy Error** | Backend port mismatch | Ensure backend is on `:8000`. Frontend `vite.config.js` proxies `/api` to `:8000`. |

---

## 🌍 Production Deployment

### Frontend (Vercel)
1. Push your code to GitHub.
2. Link the `frontend/` directory to Vercel.
3. Set `VITE_API_URL` to your hosted backend URL.

### Backend (Railway / Render / AWS)
1. Use the provided `Dockerfile` in `backend/`.
2. Ensure you set `GROQ_API_KEY` as an environment variable in your hosting provider.