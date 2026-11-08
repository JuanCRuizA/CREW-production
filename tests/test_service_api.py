"""HTTP contract of the Flask service (service/api.py), through Flask's test client.

The /api/score tests port every assertion of the former Vercel function test. The valid record is built from
the bundle itself, so no legacy files are needed. A broken bundle is injected through create_app, starting from
a really tampered copy of the bundle. Database endpoints run against a monkeypatched repository: no real database.
"""
import json
import pathlib
import random
import shutil

import api
import pytest
import repository
from app import create_app
from crew.bundle import BundleError, load_bundle
from crew.scoring import score_record

SOURCE = pathlib.Path(__file__).resolve().parents[1] / "models" / "crew-v1"


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(SOURCE)


@pytest.fixture()
def features(bundle):
    return {name: (next(iter(bundle.categorical_codes[name])) if name in bundle.categorical_codes else 0.0) for name in bundle.feature_names}


def call(client, method, body=None, content_type="application/json", path="/api/score", **kwargs):
    """Return (status, response, parsed JSON). `body` may be bytes (sent as is) or a JSON-able object."""
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode("utf-8")
    sent = {"content_type": content_type} if content_type is not None and body is not None else {}
    response = client.open(path, method=method, data=body, **sent, **kwargs)
    return response.status_code, response, json.loads(response.get_data(as_text=True))


def error_of(result):
    status, _, payload = result
    return status, payload["error"]["code"], [(p["code"], p["field"]) for p in payload["error"].get("problems", [])]


# ------------------------------------------------------------------ 200
def test_get_describes_the_model(client, bundle):
    status, response, payload = call(client, "GET")
    assert status == 200 and response.headers["Content-Type"] == "application/json"
    assert payload["model_version"] == "crew-v1" and payload["bundle_sha256"] == bundle.sha256
    assert payload["n_features"] == 216 and payload["feature_names"] == list(bundle.feature_names)
    assert set(payload["fields"]) == {"pd_raw", "pd_calibrated", "flags", "explanation"}


def test_post_returns_the_score_and_echoes_the_identifier(client, bundle, features):
    status, response, payload = call(client, "POST", {"sk_id_curr": 100002, "features": features})
    assert status == 200 and response.headers["Cache-Control"] == "no-store"
    assert payload == json.loads(json.dumps(score_record(bundle, features, 100002)))
    assert payload["sk_id_curr"] == 100002 and "pd" not in payload  # two explicit fields, no headline `pd`
    assert call(client, "POST", {"sk_id_curr": "abc-1", "features": features})[2]["sk_id_curr"] == "abc-1"
    assert call(client, "POST", {"features": features})[2]["sk_id_curr"] is None


def test_key_order_does_not_change_the_response_body(client, features):
    items = list(features.items())
    random.Random(3).shuffle(items)
    assert call(client, "POST", {"features": features})[2] == call(client, "POST", {"features": dict(items)})[2]


def test_infinity_text_and_null_follow_the_bundle_rules(client, features):
    sent = {**features, "DEBT_TO_INCOME": "Infinity", "INCOME_TO_CREDIT": "-Infinity", "EXT_SOURCE_MEAN": None}
    cleaned = {**features, "DEBT_TO_INCOME": 10, "INCOME_TO_CREDIT": -10, "EXT_SOURCE_MEAN": 0}
    first, second = call(client, "POST", {"features": sent})[2], call(client, "POST", {"features": cleaned})[2]
    assert (first["pd_raw"], first["pd_calibrated"]) == (second["pd_raw"], second["pd_calibrated"])


# ------------------------------------------------------------------ 400, 405, 413, 415
def test_malformed_json_is_400(client):
    assert error_of(call(client, "POST", b"{"))[:2] == (400, "invalid_json")
    assert error_of(call(client, "POST", b"\xff\xfe"))[:2] == (400, "invalid_json")


def test_non_standard_tokens_and_duplicate_keys_are_400(client):
    status, code, _ = error_of(call(client, "POST", b'{"features": {"AMT_CREDIT": NaN}}'))
    assert (status, code) == (400, "invalid_json")
    status, code, _ = error_of(call(client, "POST", b'{"features": {"AMT_CREDIT": Infinity}}'))
    assert (status, code) == (400, "invalid_json")
    result = call(client, "POST", b'{"features": {"AMT_CREDIT": 1, "AMT_CREDIT": 2}}')
    assert error_of(result)[:2] == (400, "invalid_json") and "duplicate key 'AMT_CREDIT'" in result[2]["error"]["message"]


@pytest.mark.parametrize("body", [
    [], {}, {"features": []}, {"features": {}, "extra": 1}, {"features": {}, "sk_id_curr": True},
    {"features": {}, "sk_id_curr": 1.5}, {"features": {}, "sk_id_curr": ""}, {"features": {}, "sk_id_curr": "x" * 65},
    {"features": {}, "sk_id_curr": [1]},
])
def test_malformed_structure_is_400(client, body):
    assert error_of(call(client, "POST", body))[:2] == (400, "invalid_request")


@pytest.mark.parametrize("length", ["abc", "-5"])
def test_bad_content_length_is_400(client, length):
    result = call(client, "POST", b"{}", environ_overrides={"CONTENT_LENGTH": length})
    assert error_of(result)[:2] == (400, "invalid_request")


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH", "OPTIONS"])
def test_other_methods_are_405(client, method):
    status, response, payload = call(client, method, b"{}")
    assert status == 405 and response.headers["Allow"] == "GET, POST" and payload["error"]["code"] == "method_not_allowed"


def test_body_over_the_limit_is_413(client, features):
    padded = json.dumps({"features": features, "sk_id_curr": "x"}).encode() + b" " * (api.MAX_BODY_BYTES + 10)
    assert error_of(call(client, "POST", padded))[:2] == (413, "payload_too_large")


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", None])
def test_wrong_content_type_is_415(client, features, content_type):
    assert error_of(call(client, "POST", {"features": features}, content_type=content_type))[:2] == (415, "unsupported_media_type")


def test_content_type_parameters_are_accepted(client, features):
    assert call(client, "POST", {"features": features}, content_type="application/json; charset=utf-8")[0] == 200


# ------------------------------------------------------------------ 422: only the client's data
def test_client_data_problems_are_422_with_every_problem_listed(client, features):
    record = {k: v for k, v in features.items() if k != "AMT_CREDIT"}
    record.update(OCCUPATION_TYPE="Astronaut", NAME_INCOME_TYPE=None, EXT_SOURCE_MEAN="0.5", SK_ID_CURR=1)
    status, code, problems = error_of(call(client, "POST", {"features": record}))
    assert (status, code) == (422, "invalid_record")
    assert sorted(problems) == sorted([
        ("missing_feature", "AMT_CREDIT"), ("unknown_category", "OCCUPATION_TYPE"), ("null_not_allowed", "NAME_INCOME_TYPE"),
        ("invalid_value", "EXT_SOURCE_MEAN"), ("unknown_feature", "SK_ID_CURR"),
    ])


# ------------------------------------------------------------------ 500: the bundle's fault, never 422
@pytest.fixture()
def broken_client(tmp_path):
    tampered = tmp_path / "crew-v1"
    shutil.copytree(SOURCE, tampered)
    path = tampered / "preprocessing.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cleaning"]["missing_feature_policy"] = "fill"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        load_bundle(tampered)
    return create_app({"CREW_BUNDLE": None, "CREW_BUNDLE_ERROR": caught.value}).test_client()


def test_unusable_bundle_is_500_with_the_cause_for_every_request(broken_client, features):
    expected = "cleaning.missing_feature_policy must be 'error', got 'fill'"
    for result in (
        call(broken_client, "GET"),
        call(broken_client, "POST", {"features": features}),
        call(broken_client, "POST", {"features": {}}),  # an invalid record is still the bundle's problem first
        call(broken_client, "POST", b"{"),
        call(broken_client, "GET", path="/health"),
        call(broken_client, "POST", path="/api/clients/100002/score"),
    ):
        status, payload = result[0], result[2]
        assert status == 500 and payload["error"]["code"] == "bundle_error" and expected in payload["error"]["message"]


# ------------------------------------------------------------------ /health
@pytest.mark.parametrize("url, configured", [(None, False), ("postgresql://user@host/db", True)])
def test_health_reports_the_bundle_and_whether_a_database_is_configured(client, bundle, monkeypatch, url, configured):
    if url is None:
        monkeypatch.delenv(repository.DATABASE_URL_ENV, raising=False)
    else:
        monkeypatch.setenv(repository.DATABASE_URL_ENV, url)
    status, _, payload = call(client, "GET", path="/health")
    assert status == 200
    assert payload == {"status": "ok", "model_version": "crew-v1", "bundle_sha256": bundle.sha256, "database_configured": configured}


# ------------------------------------------------------------------ POST /api/clients/<id>/score
@pytest.fixture()
def stored(monkeypatch, features):
    """Fake repository: client 100002 has `features`; save_score records its calls."""
    saved = []
    monkeypatch.setattr(repository, "get_features", lambda sk_id: dict(features) if sk_id == 100002 else None)

    def save_score(sk_id_curr, result, source):
        saved.append((sk_id_curr, result, source))
        return {"score_id": 7, "scored_at": "2026-11-08T10:00:00+00:00"}

    monkeypatch.setattr(repository, "save_score", save_score)
    return saved


def test_stored_client_is_scored_saved_and_returned(client, bundle, features, stored):
    status, response, payload = call(client, "POST", path="/api/clients/100002/score")
    assert status == 200 and response.headers["Cache-Control"] == "no-store"
    expected = json.loads(json.dumps(score_record(bundle, features, 100002)))
    assert payload == {**expected, "score_id": 7, "scored_at": "2026-11-08T10:00:00+00:00"}
    assert len(stored) == 1 and stored[0][0] == 100002 and stored[0][2] == "api"
    assert json.loads(json.dumps(stored[0][1])) == expected  # what is saved is exactly what /api/score would return


def test_unknown_client_is_404_and_nothing_is_saved(client, stored):
    assert error_of(call(client, "POST", path="/api/clients/999/score"))[:2] == (404, "client_not_found")
    assert stored == []


def test_invalid_stored_features_are_422_and_nothing_is_saved(client, features, stored, monkeypatch):
    broken = {k: v for k, v in features.items() if k != "AMT_CREDIT"}
    monkeypatch.setattr(repository, "get_features", lambda sk_id: broken)
    status, code, problems = error_of(call(client, "POST", path="/api/clients/100002/score"))
    assert (status, code, problems) == (422, "invalid_record", [("missing_feature", "AMT_CREDIT")])
    assert stored == []


def test_without_database_url_the_database_endpoint_is_503(client, monkeypatch):
    monkeypatch.delenv(repository.DATABASE_URL_ENV, raising=False)
    monkeypatch.setattr(repository, "_pool", None)
    assert error_of(call(client, "POST", path="/api/clients/100002/score"))[:2] == (503, "database_not_configured")


def test_non_integer_client_id_is_not_routed(client):
    assert client.post("/api/clients/abc/score").status_code == 404


# ------------------------------------------------------------------ repository checks that need no database
@pytest.mark.parametrize("limit, offset", [(0, 0), (501, 0), (10, -1), (True, 0), ("10", 0), (10, 1.0)])
def test_list_clients_rejects_a_bad_page_before_connecting(limit, offset, monkeypatch):
    monkeypatch.delenv(repository.DATABASE_URL_ENV, raising=False)
    with pytest.raises(ValueError):
        repository.list_clients(limit, offset)


def test_save_score_rejects_an_unknown_source_before_connecting(monkeypatch):
    monkeypatch.delenv(repository.DATABASE_URL_ENV, raising=False)
    with pytest.raises(ValueError, match="source must be one of"):
        repository.save_score(1, {}, "batch")


def test_nan_is_never_serialised_into_jsonb():
    wrapped = repository._jsonb({"pd_raw": float("nan")})
    with pytest.raises(ValueError):
        wrapped.dumps(wrapped.obj)  # the serialiser psycopg calls when the value is sent
