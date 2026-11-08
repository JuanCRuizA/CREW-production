"""HTML views (service/views.py) against fake `repository`, `services` and `crew.scoring` modules.

The real modules belong to the API layer; here they are replaced in sys.modules so the views are tested alone.
"""
import sys
import types
from pathlib import Path

import pytest
from flask import Flask

SERVICE_DIR = Path(__file__).resolve().parents[1] / "service"
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import views  # noqa: E402  (needs the service directory on the path)

SCORE = {
    "model_version": "crew-v1",
    "bundle_sha256": "ab" * 32,
    "sk_id_curr": 100002,
    "pd_raw": 0.8123,
    "pd_calibrated": 0.3011,
    "thresholds": {"scale": "raw", "statistical_optimal": 0.79, "business_optimal": 0.55},
    "flags": {"above_statistical": True, "above_business": True},
    "explanation": {
        "basis": "margin_log_odds",
        "shap_bias": -0.12,
        "margin": 1.46,
        "increasing_risk": [{"feature": "EXT_SOURCE_3", "value": 0.08, "shap": 0.51}],
        "decreasing_risk": [{"feature": "DAYS_BIRTH", "value": -18000, "shap": -0.22}],
    },
    "scored_at": "2026-11-08T10:00:00+00:00",
}


class DatabaseNotConfigured(RuntimeError):
    pass


class ClientNotFound(LookupError):
    pass


class RecordError(ValueError):
    def __init__(self, problems):
        self.problems = problems
        super().__init__(f"{len(problems)} problem(s) in the record")


@pytest.fixture
def fakes(monkeypatch):
    state = {"clients": {100002: {"sk_id_curr": 100002, "target": 1, "loaded_at": "2026-11-08"}},
             "score": SCORE, "no_db": False, "score_calls": [], "score_error": None}

    def guard():
        if state["no_db"]:
            raise DatabaseNotConfigured("DATABASE_URL is not set")

    repository = types.ModuleType("repository")
    repository.DatabaseNotConfigured = DatabaseNotConfigured
    repository.count_clients = lambda: (guard(), len(state["clients"]))[1]
    repository.list_clients = lambda limit, offset: (guard(), [
        {"sk_id_curr": 100002, "target": 1, "pd_raw": 0.8123, "pd_calibrated": 0.3011, "above_statistical": True,
         "above_business": True, "model_version": "crew-v1", "scored_at": "2026-11-08T10:00:00+00:00"},
        {"sk_id_curr": 100003, "target": 0, "pd_raw": None, "pd_calibrated": None, "above_statistical": None,
         "above_business": None, "model_version": None, "scored_at": None},
    ][offset:offset + limit])[1]
    repository.get_client = lambda sk: (guard(), state["clients"].get(sk))[1]
    repository.latest_score = lambda sk: (guard(), state["score"])[1]

    services = types.ModuleType("services")
    services.ClientNotFound = ClientNotFound

    def score_client(bundle, sk_id_curr, source):
        guard()
        state["score_calls"].append((bundle, sk_id_curr, source))
        if state["score_error"] is not None:
            raise state["score_error"]
        if sk_id_curr not in state["clients"]:
            raise ClientNotFound(sk_id_curr)
        return SCORE

    services.score_client = score_client
    crew = types.ModuleType("crew")
    scoring = types.ModuleType("crew.scoring")
    scoring.RecordError = RecordError
    crew.scoring = scoring
    for name, module in {"repository": repository, "services": services, "crew": crew, "crew.scoring": scoring}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return state


@pytest.fixture
def app(fakes):
    app = Flask(__name__)
    app.extensions["crew_bundle"] = types.SimpleNamespace(model_version="crew-v1")
    app.extensions["crew_bundle_error"] = None
    app.register_blueprint(views.bp)
    return app


@pytest.fixture
def client(app):
    return app.test_client()


def test_list_renders_both_pds_and_banner(client):
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 200
    assert "0.8123" in html and "0.3011" in html
    assert "not scored" in html
    assert "Not for credit decisions" in html
    assert "crew-v1" in html


@pytest.mark.parametrize("page", ["0", "-1", "abc", "1.5", ""])
def test_list_rejects_invalid_page(client, page):
    assert client.get(f"/?page={page}").status_code == 400


def test_list_page_beyond_data_is_empty(client):
    resp = client.get("/?page=3")
    assert resp.status_code == 200
    assert "No clients loaded" in resp.get_data(as_text=True)


def test_detail_shows_score_thresholds_and_drivers(client):
    html = client.get("/clients/100002").get_data(as_text=True)
    assert "EXT_SOURCE_3" in html and "DAYS_BIRTH" in html
    assert "raw scale" in html
    assert "log-odds" in html


def test_detail_unscored_client(client, fakes):
    fakes["score"] = None
    html = client.get("/clients/100002").get_data(as_text=True)
    assert "not been scored" in html


def test_detail_missing_client_is_404(client):
    assert client.get("/clients/999").status_code == 404


def test_rescore_redirects_to_detail(client, fakes, app):
    resp = client.post("/clients/100002/rescore")
    assert resp.status_code == 303
    assert resp.headers["Location"].endswith("/clients/100002")
    assert fakes["score_calls"] == [(app.extensions["crew_bundle"], 100002, "ui")]


def test_rescore_missing_client_is_404(client):
    assert client.post("/clients/999/rescore").status_code == 404


def test_rescore_record_error_lists_problems(client, fakes):
    fakes["score_error"] = RecordError([{"code": "missing_feature", "field": "AMT_CREDIT", "message": "AMT_CREDIT is missing"}])
    resp = client.post("/clients/100002/rescore")
    html = resp.get_data(as_text=True)
    assert resp.status_code == 422
    assert "AMT_CREDIT is missing" in html and "missing_feature" in html


def test_rescore_with_bundle_error_is_500(client, app, fakes):
    app.extensions["crew_bundle_error"] = "booster.json is corrupt"
    resp = client.post("/clients/100002/rescore")
    assert resp.status_code == 500
    assert fakes["score_calls"] == []


@pytest.mark.parametrize("method, path", [("get", "/"), ("get", "/clients/100002"), ("post", "/clients/100002/rescore")])
def test_no_database_is_503(client, fakes, method, path):
    fakes["no_db"] = True
    resp = getattr(client, method)(path)
    assert resp.status_code == 503
    assert "DATABASE_URL is not set" in resp.get_data(as_text=True)
