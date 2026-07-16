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
venv\Scripts\activate          # Windows
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

## 🐳 Docker Guide for Beginners

Docker allows you to package an application and all its dependencies into a "container." This guarantees that the application will run exactly the same way on any computer, without the headache of installing Python, Node.js, or configuring environments manually.

### 1. Fix "Cannot find the file specified" Error
The error you encountered (`The system cannot find the file specified / dockerDesktopLinuxEngine`) means **Docker Desktop is either not installed or not currently running** on your machine.

**To fix this:**
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2. Install it (leave default settings, ensure WSL 2 backend is selected if prompted).
3. **Open the "Docker Desktop" app** from your Windows Start Menu. 
4. Wait for the icon in your system tray (bottom right) to turn green or say "Engine running." **Docker must be open and running in the background for commands to work.**

### 2. Running the Project with Docker
Once Docker Desktop is running, you can launch the entire project (backend + frontend) with a single command:

```bash
# Navigate to the deployment folder
cd Rag_final/deployment

# Build and start the containers
docker-compose up --build
```
*(Note: The `--build` flag ensures it builds the latest version of your code. You can drop it on subsequent runs to start faster: `docker-compose up`)*

- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8000

To **stop** the servers, press `Ctrl+C` in the terminal, or run:
```bash
docker-compose down
```

### 3. How to Send / Share this App using Docker
There are two ways to share a Dockerized application:

#### Option A: Sharing the Source Code (Easiest for teams/collaborators)
Because you have a `Dockerfile` and `docker-compose.yml`, anyone can run your app if they have the code.
1. Zip the entire `Rag_final` folder (excluding `venv/`, `node_modules/`, `__pycache__` to keep it small).
2. Send the zip file to your colleague.
3. Tell them to:
   - Unzip it
   - Install and open Docker Desktop
   - Open a terminal, `cd` into the unzipped `Rag_final/deployment` folder, and run `docker-compose up --build`.
   - The app will magically work for them, no Python/Node setup required!

#### Option B: Sharing pre-built Images (For end-users)
If you don't want to share source code, you can build the images and push them to [Docker Hub](https://hub.docker.com) (like GitHub, but for Docker containers).
1. Create a free Docker Hub account.
2. Build and tag your images:
   ```bash
   docker build -t yourusername/researchlens-backend ./backend
   docker build -t yourusername/researchlens-frontend ./frontend
   ```
3. Push them to the cloud:
   ```bash
   docker push yourusername/researchlens-backend
   docker push yourusername/researchlens-frontend
   ```
4. Update `docker-compose.yml` to pull `image: yourusername/researchlens-backend` instead of using `build: ...`
5. Now, you only need to send someone the `docker-compose.yml` file. When they run `docker-compose up`, it will download the pre-built application directly from Docker Hub!
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