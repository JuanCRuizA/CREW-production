"""POST /api/score: score one client record with the CREW model bundle. GET /api/score: describe the model.

This file only does HTTP. Scoring lives in _crew (no sklearn, pandas or shap; no database).
Status codes: 400/405/413/415 for a malformed request, 422 only for problems in the client's data,
500 `bundle_error` when the model bundle itself is unusable. Unexpected errors are NOT caught here:
they propagate, Vercel answers its own 500 and the traceback reaches the logs.
Features are never logged (data minimisation).
"""
import json
import sys
from http.server import BaseHTTPRequestHandler

from _crew.bundle import BundleError, load_bundle
from _crew.scoring import RecordError, score_record

MAX_BODY_BYTES = 64 * 1024  # 216 features are about 5 KB; Vercel itself cuts at 4.5 MB
DRAIN_LIMIT_BYTES = 1024 * 1024  # bodies up to this size are consumed before an early error, so the client sees the reply
MAX_SK_ID_LENGTH = 64
TOP_LEVEL_KEYS = {"features", "sk_id_curr"}
FIELD_DOCS = {
    "pd_raw": "Booster output. Inflated by scale_pos_weight: NOT a probability of default. Thresholds and SHAP use this scale.",
    "pd_calibrated": "Isotonic calibration of pd_raw fitted on training data. Under the crew-v1 NaN policy it under-estimates.",
    "flags": "above_statistical and above_business: pd_raw >= the raw-scale threshold, compared in float64.",
    "explanation": "SHAP contributions to the raw margin (log-odds), not to pd_calibrated.",
}

# The model bundle is loaded once, when the function starts. A defect is kept and reported on every request.
try:
    BUNDLE, BUNDLE_ERROR = load_bundle(), None
except BundleError as exc:
    BUNDLE, BUNDLE_ERROR = None, exc
    print(f"BUNDLE ERROR: {exc}", file=sys.stderr)


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


class handler(BaseHTTPRequestHandler):
    bundle = BUNDLE  # tests subclass this class to inject a different bundle or a bundle error
    bundle_error = BUNDLE_ERROR

    def log_message(self, format, *args):  # Vercel already logs requests; never log anything from the body
        pass

    def _send(self, status: int, payload: dict, extra_headers: dict = None) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")  # a score must never be served from a cache
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _fail(self, status: int, code: str, message: str, problems: list = None, extra_headers: dict = None) -> None:
        error = {"code": code, "message": message}
        if problems is not None:
            error["problems"] = problems
        self._send(status, {"error": error}, extra_headers)

    def _bundle_unusable(self) -> bool:
        """A defective bundle is the server's fault: always 500, whatever the client sent."""
        if self.bundle_error is None:
            return False
        self._fail(500, "bundle_error", f"the model bundle is not usable: {self.bundle_error}")
        return True

    def do_GET(self) -> None:
        if self._bundle_unusable():
            return
        self._send(200, {
            "model_version": self.bundle.model_version,
            "bundle_sha256": self.bundle.sha256,
            "n_features": len(self.bundle.feature_names),
            "feature_names": list(self.bundle.feature_names),
            "fields": FIELD_DOCS,
        })

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", ""))
            if length < 0:
                raise ValueError
        except ValueError:
            return self._fail(400, "invalid_request", "a valid Content-Length header is required")
        body = self.rfile.read(min(length, DRAIN_LIMIT_BYTES))  # consume first so early errors never reset the connection

        if self._bundle_unusable():
            return
        if self.headers.get_content_type() != "application/json":
            return self._fail(415, "unsupported_media_type", "Content-Type must be application/json")
        if length > MAX_BODY_BYTES:
            return self._fail(413, "payload_too_large", f"the body is {length} bytes; the limit is {MAX_BODY_BYTES}")
        try:
            payload = _parse_body(body)
        except ValueError as exc:  # JSONDecodeError is a ValueError
            return self._fail(400, "invalid_json", f"the body is not valid standard JSON: {exc}")
        features, sk_id_curr, problem = _check_request(payload)
        if problem:
            return self._fail(400, "invalid_request", problem)
        try:
            result = score_record(self.bundle, features, sk_id_curr)
        except RecordError as exc:
            return self._fail(422, "invalid_record", str(exc), problems=exc.problems)
        self._send(200, result)

    def _method_not_allowed(self) -> None:
        self._fail(405, "method_not_allowed", "use GET or POST", extra_headers={"Allow": "GET, POST"})

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _method_not_allowed
