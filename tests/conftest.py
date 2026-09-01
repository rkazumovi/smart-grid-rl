"""Puts src/ on sys.path so tests import "forecasting", "api", etc. the same way every
script in this project already does (see e.g. lstm_model.py's own sys.path.insert) --
tests should exercise the real import story, not a special one that only works for
pytest."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))