"""Load and validate a CREW model bundle (models/crew-v1 layout, schema_version 1).

Everything wrong with the bundle is found here, once, when the module loads it: a
BundleError means the deployment is broken (HTTP 500), never that the client sent bad data.
Needs only json, numpy and xgboost (no sklearn, pandas or shap).
"""
import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb

SCHEMA_VERSION = 1
# model_card.json is documentation: scoring never reads it, so it is neither required nor hashed.
SCORING_FILES = ("booster.json", "calibration.json", "preprocessing.json", "thresholds.json")
DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent / "bundle"
REQUIRED_POLICIES = {
    "missing_feature_policy": "error",
    "unknown_category_policy": "error",
    "categorical_null_policy": "label_if_in_classes_else_error",
}
THRESHOLD_KEYS = ("statistical_optimal", "business_optimal", "default")


class BundleError(Exception):
    """The bundle is missing, corrupt or uses a rule this code does not implement."""


@dataclass(frozen=True)
class Bundle:
    model_version: str
    sha256: str
    feature_names: tuple
    categorical_codes: dict  # column -> {class label: code}; the code is the position in classes_
    cleaning: dict  # posinf_value, neginf_value, numeric_nan_value, categorical_null_label
    calib_x: np.ndarray
    calib_y: np.ndarray
    thresholds: dict  # raw-scale cut points
    booster: xgb.Booster
    predict_lock: threading.Lock = field(default_factory=threading.Lock, compare=False, repr=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _reject_constant(token: str):
    raise ValueError(f"non-standard JSON constant {token}")


def _read_json(path: Path):
    if not path.is_file():
        raise BundleError(f"missing bundle file: {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError and the NaN/Infinity rejection
        raise BundleError(f"{path.name} is not valid JSON: {exc}") from exc


def _schema(name: str, data) -> None:
    _require(isinstance(data, dict), f"{name} must be a JSON object")
    _require(
        data.get("schema_version") == SCHEMA_VERSION,
        f"{name}: schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}",
    )


def _check_preprocessing(pre) -> None:
    _schema("preprocessing.json", pre)
    _require(isinstance(pre.get("model_version"), str), "preprocessing.json: model_version must be a string")
    names = pre.get("feature_names")
    _require(
        isinstance(names, list) and names and all(isinstance(n, str) for n in names),
        "preprocessing.json: feature_names must be a non-empty list of strings",
    )
    _require(len(set(names)) == len(names), "preprocessing.json: feature_names has duplicates")
    _require(
        pre.get("n_features") == len(names),
        f"preprocessing.json: n_features ({pre.get('n_features')!r}) differs from len(feature_names) ({len(names)})",
    )

    cat, enc = pre.get("categorical_features"), pre.get("categorical_encoders")
    _require(
        isinstance(cat, list) and isinstance(enc, dict) and len(set(cat)) == len(cat) and set(cat) == set(enc),
        "preprocessing.json: categorical_features must list exactly the keys of categorical_encoders",
    )
    _require(set(cat) <= set(names), f"preprocessing.json: categorical columns not in feature_names: {sorted(set(cat) - set(names))}")
    for col, encoder in enc.items():
        classes = encoder.get("classes") if isinstance(encoder, dict) else None
        _require(
            isinstance(classes, list) and classes and all(isinstance(c, str) for c in classes),
            f"preprocessing.json: {col} needs a non-empty list of string classes",
        )
        _require(len(set(classes)) == len(classes), f"preprocessing.json: {col} has duplicated classes")

    cleaning = pre.get("cleaning")
    _require(isinstance(cleaning, dict), "preprocessing.json: cleaning must be an object")
    for key, required in REQUIRED_POLICIES.items():
        _require(
            cleaning.get(key) == required,
            f"preprocessing.json cleaning.{key} must be {required!r}, got {cleaning.get(key)!r}",
        )
    for key in ("posinf_value", "neginf_value", "numeric_nan_value"):
        _require(
            _is_number(cleaning.get(key)),
            f"preprocessing.json cleaning.{key} must be a finite number (null is not supported yet), got {cleaning.get(key)!r}",
        )
    _require(isinstance(cleaning.get("categorical_null_label"), str), "preprocessing.json cleaning.categorical_null_label must be a string")


def _check_calibration(cal) -> None:
    _schema("calibration.json", cal)
    _require(cal.get("method") == "isotonic", f"calibration.json: method must be 'isotonic', got {cal.get('method')!r}")
    _require(cal.get("out_of_bounds") == "clip", f"calibration.json: out_of_bounds must be 'clip', got {cal.get('out_of_bounds')!r}")
    x, y = cal.get("x_thresholds"), cal.get("y_thresholds")
    _require(
        isinstance(x, list) and isinstance(y, list) and len(x) == len(y) >= 2 and all(_is_number(v) for v in x + y),
        "calibration.json: x_thresholds and y_thresholds must be lists of finite numbers, same length, at least 2",
    )
    _require(cal.get("n_points") == len(x), f"calibration.json: n_points ({cal.get('n_points')!r}) differs from len(x_thresholds) ({len(x)})")
    _require(all(b > a for a, b in zip(x, x[1:])), "calibration.json: x_thresholds must be strictly increasing")
    _require(
        all(0.0 <= v <= 1.0 for v in y) and all(b >= a for a, b in zip(y, y[1:])),
        "calibration.json: y_thresholds must be non-decreasing and within [0, 1]",
    )


def _check_thresholds(thr) -> None:
    _schema("thresholds.json", thr)
    _require(thr.get("scale") == "raw", f"thresholds.json: scale must be 'raw', got {thr.get('scale')!r}")
    values = thr.get("thresholds")
    _require(
        isinstance(values, dict) and set(values) == set(THRESHOLD_KEYS) and all(_is_number(v) for v in values.values()),
        f"thresholds.json: thresholds must hold exactly {list(THRESHOLD_KEYS)} as finite numbers",
    )


def _sha256(bundle_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in SCORING_FILES:
        digest.update(name.encode() + b"\0" + (bundle_dir / name).read_bytes() + b"\0")
    return digest.hexdigest()


def load_bundle(bundle_dir: Path = DEFAULT_BUNDLE_DIR) -> Bundle:
    """Read, validate and assemble a bundle. Raises BundleError with the cause on any defect."""
    bundle_dir = Path(bundle_dir)
    pre = _read_json(bundle_dir / "preprocessing.json")
    cal = _read_json(bundle_dir / "calibration.json")
    thr = _read_json(bundle_dir / "thresholds.json")
    _check_preprocessing(pre)
    _check_calibration(cal)
    _check_thresholds(thr)

    booster_path = bundle_dir / "booster.json"
    _require(booster_path.is_file(), "missing bundle file: booster.json")
    booster = xgb.Booster()
    try:
        booster.load_model(str(booster_path))
    except xgb.core.XGBoostError as exc:
        raise BundleError(f"booster.json could not be loaded by xgboost {xgb.__version__}: {exc}") from exc
    _require(
        booster.feature_names == pre["feature_names"],
        "booster.json feature_names differ from preprocessing.json feature_names (the order must be identical)",
    )

    return Bundle(
        model_version=pre["model_version"],
        sha256=_sha256(bundle_dir),
        feature_names=tuple(pre["feature_names"]),
        categorical_codes={
            col: {label: code for code, label in enumerate(enc["classes"])}
            for col, enc in pre["categorical_encoders"].items()
        },
        cleaning={k: pre["cleaning"][k] for k in ("posinf_value", "neginf_value", "numeric_nan_value", "categorical_null_label")},
        calib_x=np.array(cal["x_thresholds"], dtype=np.float64),
        calib_y=np.array(cal["y_thresholds"], dtype=np.float64),
        thresholds=dict(thr["thresholds"]),
        booster=booster,
    )
