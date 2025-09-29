# -*- coding: utf-8 -*-
"""Pytest configuration and fixtures for the API tests."""
import os
import sys
import tempfile
import joblib
import pytest

# Ensure project root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app import create_app

@pytest.fixture
def client():
    """Flask test client with MODEL_PATH pointing to model."""
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()
