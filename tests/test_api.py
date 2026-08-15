"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from sophia.api.app import app


def test_api_health_and_models():
    client = TestClient(app)
    
    # 1. Health
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    # 2. Models
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "pool" in data
    assert "rotating_models" in data
    assert len(data["rotating_models"]) > 0
