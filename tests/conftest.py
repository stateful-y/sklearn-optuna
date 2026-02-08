"""Test configuration and fixtures for Sklearn-Optuna."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {"key": "value", "number": 42}
