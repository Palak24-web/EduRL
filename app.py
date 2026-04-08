# Copyright (c) 2026 - Personalized Learning Path OpenEnv Environment
"""
Root app.py — required by Hugging Face Spaces.

This file re-exports the FastAPI `app` object from server/app.py so that
HF Spaces can locate it, while the OpenEnv checker finds server/app.py.
"""

import os
import sys

# Ensure the project root is on sys.path so `server.*` imports resolve
sys.path.insert(0, os.path.dirname(__file__))

from server.app import app, main  # noqa: F401  re-export for HF Spaces

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 7860)))
    args = parser.parse_args()
    main(port=args.port)

# Keep a bare main() call so openenv validate recognises the entry point
if __name__ == "__main__":
    main()
