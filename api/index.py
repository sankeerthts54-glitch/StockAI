"""
index.py
--------
Vercel serverless entry point for FastAPI backend.
"""

import os
import sys

# Ensure project root and backend are on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
backend_dir = os.path.join(root_dir, "backend")

for path in [root_dir, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.main import app
