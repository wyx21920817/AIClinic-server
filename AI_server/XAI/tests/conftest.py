import os
import sys

# Ensure the project root (XAI directory) is on sys.path so imports like
# `import ecg_utils` work when running `pytest` from the XAI folder.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


