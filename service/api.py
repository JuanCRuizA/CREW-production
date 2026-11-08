"""JSON API: GET /health, GET|POST /api/score (stateless) and POST /api/clients/<id>/score (database).

/api/score keeps the contract of the former Vercel function: strict JSON, 400/405/413/415 for a malformed
request, 422 only for problems in the client's data, 500 `bundle_error` when the model bundle is unusable.
Unexpected errors are NOT caught here: Flask answers 500 and the traceback reaches the logs.
Features are never logged (data minimisation).
"""
import json

from flask import Blueprint, Response, current_app, request

import repository
import services
from crew.scoring import RecordError, score_record

MAX_BODY_BYTES = 64 * 1024  # 216 features are about 5 KB
DRAIN_LIMIT_BYTES = 1024 * 1024  # bodies up to this size are consumed before an early error, so the client sees the reply
MAX_SK_ID_LENGTH = 64
TOP_LEVEL_KEYS = {"features", "sk_id_curr"}
ALLOWED = "GET, POST"
ROUTED_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"]  # the extra ones answer a JSON 405
FIELD_DOCS = {
    "pd_raw": "Booster output. Inflated by scale_pos_weight: NOT a probability of default. Thresholds and SHAP use this scale.",
    "pd_calibrated": "Isotonic calibration of pd_raw fitted on training data. Under the crew-v1 NaN policy it under-estimates.",
    "flags": "above_statistical and above_business: pd_raw >= the raw-scale threshold, compared in float64.",
    "explanation": "SHAP contributions to the raw margin (log-odds), not to pd_calibrated.",
}

bp = Blueprint("api", __name__)


def _reject_constant(token: str):
    raise ValueError(f"non-standard JSON constant {token}: send infinity as the text 'Infinity' or '-Infinity', and NaN as null")


def _reject_duplicates(pairs: list) -> dict:
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen.add(key)
    return dict(pairs)


def _parse_body(body: bytes):
    """Strict JSON: no NaN/Infinity tokens and no duplicate keys. Raises ValueError with the reason."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"body is not UTF-8: {exc}") from exc
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates)


def _check_request(payload):
    """Return (features, sk_id_curr, None), or (None, None, message) when the request structure is malformed."""
    if not isinstance(payload, dict):
        return None, None, "the body must be a JSON object"
    unexpected = sorted(set(payload) - TOP_LEVEL_KEYS)
    if unexpected:
        return None, None, f"unexpected top-level keys {unexpected}; allowed: {sorted(TOP_LEVEL_KEYS)}"
    if not isinstance(payload.get("features"), dict):
        return None, None, "'features' is required and must be an object"
    sk_id = payload.get("sk_id_curr")
    valid_id = sk_id is None or (isinstance(sk_id, int) and not isinstance(sk_id, bool)) or (isinstance(sk_id, str) and 0 < len(sk_id) <= MAX_SK_ID_LENGTH)
    if not valid_id:
        return None, None, f"'sk_id_curr' must be an integer or a text of 1 to {MAX_SK_ID_LENGTH} characters"
    return payload["features"], sk_id, None


def _send(status: int, payload: dict, extra_headers: dict = None) -> Response:
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    headers = {"Cache-Control": "no-store", **(extra_headers or {})}  # a score must never be served from a cache
    return Response(body, status=status, content_type="application/json", headers=headers)


def _fail(status: int, code: str, message: str, problems: list = None, extra_headers: dict = None) -> Response:
    error = {"code": code, "message": message}
    if problems is not None:
        error["problems"] = problems
    return _send(status, {"error": error}, extra_headers)


def _bundle():
    """(bundle, None), or (None, 500 response): a defective bundle is the server's fault, whatever the client sent."""
    error = current_app.extensions["crew_bundle_error"]
    if error is not None:
        return None, _fail(500, "bundle_error", f"the model bundle is not usable: {error}")
    return current_app.extensions["crew_bundle"], None


@bp.get("/health")
def health():
    bundle, failure = _bundle()
    if failure:
        return failure
    return _send(200, {
        "status": "ok",
        "model_version": bundle.model_version,
        "bundle_sha256": bundle.sha256,
        "database_configured": repository.is_configured(),  # only checks the variable: /health never opens a connection
    })


@bp.route("/api/score", methods=ROUTED_METHODS, provide_automatic_options=False)
def score():
    if request.method == "GET":
        return _describe()
    if request.method == "POST":
        return _score_posted()
    return _fail(405, "method_not_allowed", "use GET or POST", extra_headers={"Allow": ALLOWED})


def _describe() -> Response:
    bundle, failure = _bundle()
    if failure:
        return failure
    return _send(200, {
        "model_version": bundle.model_version,
        "bundle_sha256": bundle.sha256,
        "n_features": len(bundle.feature_names),
        "feature_names": list(bundle.feature_names),
        "fields": FIELD_DOCS,
    })


def _score_posted() -> Response:
    try:
        length = int(request.headers.get("Content-Length", ""))
        if length < 0:
            raise ValueError
    except ValueError:
        return _fail(400, "invalid_request", "a valid Content-Length header is required")
    body = request.stream.read(min(length, DRAIN_LIMIT_BYTES))  # consume first so early errors never reset the connection

    bundle, failure = _bundle()
    if failure:
        return failure
    if request.mimetype != "application/json":
        return _fail(415, "unsupported_media_type", "Content-Type must be application/json")
    if length > MAX_BODY_BYTES:
        return _fail(413, "payload_too_large", f"the body is {length} bytes; the limit is {MAX_BODY_BYTES}")
    try:
        payload = _parse_body(body)
    except ValueError as exc:  # JSONDecodeError is a ValueError
        return _fail(400, "invalid_json", f"the body is not valid standard JSON: {exc}")
    features, sk_id_curr, problem = _check_request(payload)
    if problem:
        return _fail(400, "invalid_request", problem)
    try:
        result = score_record(bundle, features, sk_id_curr)
    except RecordError as exc:
        return _fail(422, "invalid_record", str(exc), problems=exc.problems)
    return _send(200, result)


@bp.post("/api/clients/<int:sk_id_curr>/score")
def score_stored_client(sk_id_curr: int):
    """Score a client from its stored features and save the score (crew.scores + crew.score_audit_log)."""
    bundle, failure = _bundle()
    if failure:
        return failure
    try:
        result = services.score_client(bundle, sk_id_curr, source="api")
    except repository.DatabaseNotConfigured as exc:
        return _fail(503, "database_not_configured", str(exc))
    except services.ClientNotFound as exc:
        return _fail(404, "client_not_found", str(exc))
    except RecordError as exc:
        return _fail(422, "invalid_record", f"the stored features of client {sk_id_curr} are not valid: {exc}", problems=exc.problems)
    return _send(200, result)
