"""HTTP contract of web/api/score.py, tested against a real server on 127.0.0.1 (ephemeral port).

The valid record is built from the bundle itself, so no legacy files are needed. A broken bundle is injected by
subclassing `handler` (the same literal class Vercel imports), starting from a really tampered copy of the bundle.
"""
import http.client
import json
import pathlib
import random
import shutil
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest
import score
from _crew.bundle import BundleError, load_bundle
from _crew.scoring import score_record

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "models" / "crew-v1"


@contextmanager
def serve(handler_class):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def call(port, method, body=None, content_type="application/json", headers=None):
    """Return (status, headers, parsed JSON). `body` may be bytes (sent as is) or a JSON-able object."""
    if body is not None and not isinstance(body, bytes):
        body = json.dumps(body).encode("utf-8")
    sent = dict(headers or {})
    if content_type is not None and body is not None:
        sent["Content-Type"] = content_type
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, "/api/score", body=body, headers=sent)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, response, payload


@pytest.fixture(scope="module")
def port():
    with serve(score.handler) as running_port:
        yield running_port


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(SOURCE)


@pytest.fixture()
def features(bundle):
    return {name: (next(iter(bundle.categorical_codes[name])) if name in bundle.categorical_codes else 0.0) for name in bundle.feature_names}


def error_of(result):
    status, _, payload = result
    return status, payload["error"]["code"], [(p["code"], p["field"]) for p in payload["error"].get("problems", [])]


# ------------------------------------------------------------------ 200
def test_get_describes_the_model(port, bundle):
    status, response, payload = call(port, "GET")
    assert status == 200 and response.getheader("Content-Type") == "application/json"
    assert payload["model_version"] == "crew-v1" and payload["bundle_sha256"] == bundle.sha256
    assert payload["n_features"] == 216 and payload["feature_names"] == list(bundle.feature_names)
    assert set(payload["fields"]) == {"pd_raw", "pd_calibrated", "flags", "explanation"}


def test_post_returns_the_score_and_echoes_the_identifier(port, bundle, features):
    status, response, payload = call(port, "POST", {"sk_id_curr": 100002, "features": features})
    assert status == 200 and response.getheader("Cache-Control") == "no-store"
    assert payload == json.loads(json.dumps(score_record(bundle, features, 100002)))
    assert payload["sk_id_curr"] == 100002 and "pd" not in payload  # two explicit fields, no headline `pd`
    assert call(port, "POST", {"sk_id_curr": "abc-1", "features": features})[2]["sk_id_curr"] == "abc-1"
    assert call(port, "POST", {"features": features})[2]["sk_id_curr"] is None


def test_key_order_does_not_change_the_response_body(port, features):
    items = list(features.items())
    random.Random(3).shuffle(items)
    assert call(port, "POST", {"features": features})[2] == call(port, "POST", {"features": dict(items)})[2]


def test_infinity_text_and_null_follow_the_bundle_rules(port, features):
    sent = {**features, "DEBT_TO_INCOME": "Infinity", "INCOME_TO_CREDIT": "-Infinity", "EXT_SOURCE_MEAN": None}
    cleaned = {**features, "DEBT_TO_INCOME": 10, "INCOME_TO_CREDIT": -10, "EXT_SOURCE_MEAN": 0}
    first, second = call(port, "POST", {"features": sent})[2], call(port, "POST", {"features": cleaned})[2]
    assert (first["pd_raw"], first["pd_calibrated"]) == (second["pd_raw"], second["pd_calibrated"])


# ------------------------------------------------------------------ 400, 405, 413, 415
def test_malformed_json_is_400(port):
    assert error_of(call(port, "POST", b"{"))[:2] == (400, "invalid_json")
    assert error_of(call(port, "POST", b"\xff\xfe"))[:2] == (400, "invalid_json")


def test_non_standard_tokens_and_duplicate_keys_are_400(port):
    status, code, _ = error_of(call(port, "POST", b'{"features": {"AMT_CREDIT": NaN}}'))
    assert (status, code) == (400, "invalid_json")
    status, code, _ = error_of(call(port, "POST", b'{"features": {"AMT_CREDIT": Infinity}}'))
    assert (status, code) == (400, "invalid_json")
    result = call(port, "POST", b'{"features": {"AMT_CREDIT": 1, "AMT_CREDIT": 2}}')
    assert error_of(result)[:2] == (400, "invalid_json") and "duplicate key 'AMT_CREDIT'" in result[2]["error"]["message"]


@pytest.mark.parametrize("body", [
    [], {}, {"features": []}, {"features": {}, "extra": 1}, {"features": {}, "sk_id_curr": True},
    {"features": {}, "sk_id_curr": 1.5}, {"features": {}, "sk_id_curr": ""}, {"features": {}, "sk_id_curr": "x" * 65},
    {"features": {}, "sk_id_curr": [1]},
])
def test_malformed_structure_is_400(port, body):
    assert error_of(call(port, "POST", body))[:2] == (400, "invalid_request")


@pytest.mark.parametrize("length", ["abc", "-5"])
def test_bad_content_length_is_400(port, length):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.putrequest("POST", "/api/score")
    connection.putheader("Content-Type", "application/json")
    connection.putheader("Content-Length", length)
    connection.endheaders()
    response = connection.getresponse()
    assert response.status == 400 and json.loads(response.read())["error"]["code"] == "invalid_request"


@pytest.mark.parametrize("method", ["PUT", "DELETE", "PATCH"])
def test_other_methods_are_405(port, method):
    status, response, payload = call(port, method, b"{}")
    assert status == 405 and response.getheader("Allow") == "GET, POST" and payload["error"]["code"] == "method_not_allowed"


def test_body_over_the_limit_is_413(port, features):
    padded = json.dumps({"features": features, "sk_id_curr": "x"}).encode() + b" " * (score.MAX_BODY_BYTES + 10)
    assert error_of(call(port, "POST", padded))[:2] == (413, "payload_too_large")


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", None])
def test_wrong_content_type_is_415(port, features, content_type):
    assert error_of(call(port, "POST", {"features": features}, content_type=content_type))[:2] == (415, "unsupported_media_type")


def test_content_type_parameters_are_accepted(port, features):
    assert call(port, "POST", {"features": features}, content_type="application/json; charset=utf-8")[0] == 200


# ------------------------------------------------------------------ 422: only the client's data
def test_client_data_problems_are_422_with_every_problem_listed(port, features):
    record = {k: v for k, v in features.items() if k != "AMT_CREDIT"}
    record.update(OCCUPATION_TYPE="Astronaut", NAME_INCOME_TYPE=None, EXT_SOURCE_MEAN="0.5", SK_ID_CURR=1)
    status, code, problems = error_of(call(port, "POST", {"features": record}))
    assert (status, code) == (422, "invalid_record")
    assert sorted(problems) == sorted([
        ("missing_feature", "AMT_CREDIT"), ("unknown_category", "OCCUPATION_TYPE"), ("null_not_allowed", "NAME_INCOME_TYPE"),
        ("invalid_value", "EXT_SOURCE_MEAN"), ("unknown_feature", "SK_ID_CURR"),
    ])


# ------------------------------------------------------------------ 500: the bundle's fault, never 422
@pytest.fixture()
def broken_port(tmp_path):
    tampered = tmp_path / "crew-v1"
    shutil.copytree(SOURCE, tampered)
    path = tampered / "preprocessing.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["cleaning"]["missing_feature_policy"] = "fill"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(BundleError) as caught:
        load_bundle(tampered)

    class BrokenHandler(score.handler):
        bundle = None
        bundle_error = caught.value

    with serve(BrokenHandler) as running_port:
        yield running_port


def test_unusable_bundle_is_500_with_the_cause_for_every_request(broken_port, features):
    expected = "cleaning.missing_feature_policy must be 'error', got 'fill'"
    for result in (
        call(broken_port, "GET"),
        call(broken_port, "POST", {"features": features}),
        call(broken_port, "POST", {"features": {}}),  # an invalid record is still the bundle's problem first
        call(broken_port, "POST", b"{"),
    ):
        status, payload = result[0], result[2]
        assert status == 500 and payload["error"]["code"] == "bundle_error" and expected in payload["error"]["message"]
