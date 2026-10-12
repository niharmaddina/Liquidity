# Make `backend/` importable so the tests use the same flat imports as the scripts.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
