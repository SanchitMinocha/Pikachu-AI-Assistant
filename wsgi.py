"""
WSGI entry point for Apache mod_wsgi deployment.

Apache config should point WSGIScriptAlias to this file.
See apache/sanchit-ai.conf for the full Apache VirtualHost configuration.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env if present
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())

from app import app as application  # noqa: F401 — mod_wsgi looks for `application`
