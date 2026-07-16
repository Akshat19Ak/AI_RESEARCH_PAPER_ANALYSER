# Deployment Configurations

This directory contains deployment configurations for various platforms.

## Files

| File | Platform | Description |
|------|----------|-------------|
| `docker-compose.yml` | Docker | Full stack (backend + frontend) |
| `render.yaml` | Render | Backend deployment |
| `vercel.json` | Vercel | Frontend deployment |

## Docker

```bash
cd deployment
docker-compose up --build
```
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000

## Render (Backend)

1. Connect your GitHub repo to Render
2. Render auto-detects `render.yaml`
3. Set `GROQ_API_KEY` in Render dashboard

## Vercel (Frontend)

1. Link the `frontend/` directory to Vercel
2. Set `VITE_API_URL` to your Render backend URL
3. Vercel auto-detects `vercel.json` in `frontend/`
