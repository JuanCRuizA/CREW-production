"""Score one client record against a loaded Bundle.

Everything wrong with the record is a RecordError (HTTP 422). Bundle problems never get here:
load_bundle() already rejected them, so this module trusts the Bundle it receives.
Needs only numpy and xgboost.
"""
import math

import numpy as np
import xgboost as xgb

from .bundle import Bundle

N_INCREASING = 3  # same counts as the Streamlit calculator (nlargest(3) / nsmallest(2))
N_DECREASING = 2
MAX_PROBLEMS = 50
INFINITY_TEXT = {"Infinity": math.inf, "-Infinity": -math.inf}


class RecordError(ValueError):
    """The client's record cannot be scored. `problems` lists every defect found, not only the first."""

    def __init__(self, problems: list):
        self.problems = problems
        shown = "; ".join(p["message"] for p in problems[:3])
        super().__init__(f"{len(problems)} problem(s) in the record: {shown}")


def _problem(code: str, field: str, message: str) -> dict:
    return {"code": code, "field": field, "message": message}


def _kind(value) -> str:
    return "null" if value is None else {bool: "boolean", int: "number", float: "number", str: "text"}.get(type(value), type(value).__name__)


def _encode_categorical(bundle: Bundle, name: str, value, problems: list):
    codes = bundle.categorical_codes[name]
    if value is None:
        label = bundle.cleaning["categorical_null_label"]
        if label not in codes:
            problems.append(_problem("null_not_allowed", name, f"{name}: null is not allowed, the encoder has no {label!r} class"))
            return None
        return float(codes[label])
    if not isinstance(value, str):
        problems.append(_problem("invalid_value", name, f"{name}: expected text or null, got {_kind(value)}"))
        return None
    if value not in codes:
        problems.append(_problem("unknown_category", name, f"{name}: {value!r} is not one of the {len(codes)} known classes"))
        return None
    return float(codes[value])


def _clean_numeric(bundle: Bundle, name: str, value, problems: list):
    cleaning = bundle.cleaning
    if value is None:
        return float(cleaning["numeric_nan_value"])
    if isinstance(value, str) and value in INFINITY_TEXT:
        return float(cleaning["posinf_value"] if INFINITY_TEXT[value] > 0 else cleaning["neginf_value"])
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(_problem("invalid_value", name, f"{name}: expected a number, null, 'Infinity' or '-Infinity', got {_kind(value)}"))
        return None
    try:
        number = float(value)
    except OverflowError:
        problems.append(_problem("invalid_value", name, f"{name}: the number is too large to represent"))
        return None
    if math.isnan(number):
        return float(cleaning["numeric_nan_value"])
    if math.isinf(number):
        return float(cleaning["posinf_value"] if number > 0 else cleaning["neginf_value"])
    return number


def encode_record(bundle: Bundle, features: dict) -> list:
    """Build the feature vector in the bundle's feature order, or raise RecordError with every defect."""
    known = set(bundle.feature_names)
    problems = [
        _problem("unknown_feature", str(key), f"{key}: not a model feature")
        for key in sorted(map(str, features)) if key not in known
    ]
    row = []
    for name in bundle.feature_names:
        if name not in features:
            problems.append(_problem("missing_feature", name, f"{name}: missing from features"))
            continue
        if name in bundle.categorical_codes:
            row.append(_encode_categorical(bundle, name, features[name], problems))
        else:
            row.append(_clean_numeric(bundle, name, features[name], problems))
    if problems:
        raise RecordError(problems[:MAX_PROBLEMS])
    return row


def _echo(value):
    """The value as received, made JSON-safe (non-finite floats come back as the text the API accepts)."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _drivers(bundle: Bundle, features: dict, shap: np.ndarray, indices) -> list:
    return [
        {"feature": bundle.feature_names[i], "value": _echo(features[bundle.feature_names[i]]), "shap": float(shap[i])}
        for i in indices
    ]


def predict_row(bundle: Bundle, row: list) -> tuple:
    """Raw PD, margin and SHAP contributions (last element = bias) for one already encoded row."""
    dmatrix = xgb.DMatrix(np.array([row], dtype=np.float64), feature_names=list(bundle.feature_names))
    with bundle.predict_lock:  # concurrent requests may share one Booster; do not assume predict is reentrant
        prob = bundle.booster.predict(dmatrix)
        margin = bundle.booster.predict(dmatrix, output_margin=True)
        contribs = bundle.booster.predict(dmatrix, pred_contribs=True)
    return float(prob[0]), float(margin[0]), contribs[0]  # float32 PD is exact in float64: np.interp must see it


def score_record(bundle: Bundle, features: dict, sk_id_curr=None) -> dict:
    """Score one record. Raises RecordError for bad client data; any other exception is a bug."""
    if not isinstance(features, dict):
        raise RecordError([_problem("invalid_value", "features", f"features must be an object, got {_kind(features)}")])
    pd_raw, margin, contribs = predict_row(bundle, encode_record(bundle, features))

    pd_calibrated = float(np.interp(pd_raw, bundle.calib_x, bundle.calib_y))
    shap = contribs[:-1]
    top_up = [i for i in np.argsort(-shap, kind="stable")[:N_INCREASING] if shap[i] > 0]
    top_down = [i for i in np.argsort(shap, kind="stable")[:N_DECREASING] if shap[i] < 0]

    statistical = bundle.thresholds["statistical_optimal"]
    business = bundle.thresholds["business_optimal"]
    return {
        "model_version": bundle.model_version,
        "bundle_sha256": bundle.sha256,
        "sk_id_curr": sk_id_curr,
        "pd_raw": pd_raw,
        "pd_calibrated": pd_calibrated,
        "thresholds": {"scale": "raw", "statistical_optimal": statistical, "business_optimal": business},
        "flags": {"above_statistical": pd_raw >= statistical, "above_business": pd_raw >= business},
        "explanation": {
            "basis": "margin_log_odds",  # SHAP explains the raw margin, not pd_calibrated
            "shap_bias": float(contribs[-1]),
            "margin": margin,
            "increasing_risk": _drivers(bundle, features, shap, top_up),
            "decreasing_risk": _drivers(bundle, features, shap, top_down),
        },
    }
