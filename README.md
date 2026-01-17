# .env
# __pycache__/
# *.pyc
# faiss_index/
# .streamlit/
# SELF_README.md

# --- PROJECT SPECIFIC ---
# Your virtual environment
venv/

# Your API keys and secrets (CRITICAL)
.env

# RAG specific: FAISS indices and vector data
faiss_index/
*.index
*.pkl

# --- PYTHON JUNK ---
# Only catches what Python actually creates while running
**/__pycache__/
*.py[cod]

# --- IDE & OS ---
.vscode/
.DS_Store
Thumbs.db

SELF_README.md