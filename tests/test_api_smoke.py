"""Import-and-wire-up smoke test for the FastAPI app: no trained checkpoint is required
(the /health endpoint never touches one), but this test DOES exercise the exact same
import mechanics main.py documents and depends on (src/api added to sys.path via
--app-dir, `api` imported as a top-level package). This is precisely the class of bug
that broke this project once already -- a module name collision between
grid_intelligence/agents.py and the real agents/ package caused an ImportError that only
showed up when running the script directly, not from casual code reading -- so a test
that actually imports the app (instead of only checking the source parses) is the
regression guard for that failure mode, not a redundant one.
"""
import os
import sys

from fastapi.testclient import TestClient

_API_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "api")
sys.path.insert(0, _API_DIR)

import main  # noqa: E402  (must follow the sys.path.insert above, same as --app-dir would)

client = TestClient(main.app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": main.API_VERSION}


def test_metrics_endpoint_is_exposed():
    client.get("/health")  # make sure at least one request has been counted first
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"http_requests_total" in response.content


def test_forecast_rejects_unknown_target():
    response = client.get("/forecast/not_a_real_target")
    assert response.status_code == 422


def test_policy_rejects_unknown_policy_name():
    response = client.get("/policy/NOT_A_POLICY")
    assert response.status_code == 422


def test_docs_are_served():
    response = client.get("/docs")
    assert response.status_code == 200