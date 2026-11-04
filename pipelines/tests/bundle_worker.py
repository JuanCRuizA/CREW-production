"""Score records with a CREW bundle using only json, numpy and xgboost.

Runs in a subprocess so the parity test can prove the bundle path needs no sklearn,
pandas or shap: those modules are blocked before anything else is imported.

usage: python bundle_worker.py <bundle_dir> <records.json> <results.json>
"""
import json
import sys

BLOCKED = ("sklearn", "pandas", "shap")
for _name in BLOCKED:
    sys.modules[_name] = None  # any `import sklearn` now raises ImportError

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402


REQUIRED_POLICIES = {
    "missing_feature_policy": "error",
    "unknown_category_policy": "error",
    "categorical_null_policy": "label_if_in_classes_else_error",
}


def is_null(value) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def clean_record(rec: dict, pre: dict, codes: dict) -> list[float]:
    """Apply the bundle's cleaning rules to one raw record, in feature_names order."""
    cfg = pre["cleaning"]
    row = []
    for name in pre["feature_names"]:
        if name not in rec:
            raise ValueError(f"missing feature: {name}")
        value = rec[name]
        if name in codes:
            null = is_null(value)
            label = cfg["categorical_null_label"] if null else value
            code = codes[name].get(label)
            if code is None:
                raise ValueError(f"{name}: {'null' if null else repr(value)} is not in the encoder classes")
            row.append(float(code))
        elif is_null(value):
            row.append(np.nan if cfg["numeric_nan_value"] is None else float(cfg["numeric_nan_value"]))
        elif value == float("inf"):
            row.append(float(cfg["posinf_value"]))
        elif value == float("-inf"):
            row.append(float(cfg["neginf_value"]))
        else:
            row.append(float(value))
    return row


def main(bundle_dir: str, records_path: str, results_path: str) -> None:
    pre = json.load(open(f"{bundle_dir}/preprocessing.json", encoding="utf-8"))
    for key, required in REQUIRED_POLICIES.items():
        actual = pre["cleaning"].get(key)
        if actual != required:
            raise ValueError(f"preprocessing.json cleaning.{key} must be {required!r}, got {actual!r}")
    cal = json.load(open(f"{bundle_dir}/calibration.json", encoding="utf-8"))
    booster = xgb.Booster()
    booster.load_model(f"{bundle_dir}/booster.json")
    codes = {c: {k: i for i, k in enumerate(e["classes"])} for c, e in pre["categorical_encoders"].items()}
    records = json.load(open(records_path, encoding="utf-8"))

    results: list[dict | None] = [None] * len(records)
    rows, index = [], []
    for i, rec in enumerate(records):
        try:
            rows.append(clean_record(rec, pre, codes))
            index.append(i)
        except ValueError as exc:
            results[i] = {"error": str(exc)}

    if rows:
        dmat = xgb.DMatrix(np.array(rows, dtype=np.float64), feature_names=pre["feature_names"])
        prob = booster.predict(dmat)
        margin = booster.predict(dmat, output_margin=True)
        contribs = booster.predict(dmat, pred_contribs=True)  # (n, n_features + 1); last column = bias
        pd_cal = np.interp(prob.astype(np.float64), cal["x_thresholds"], cal["y_thresholds"])
        for k, i in enumerate(index):
            results[i] = {
                "pd_raw": float(prob[k]),
                "pd_cal": float(pd_cal[k]),
                "margin": float(margin[k]),
                "contribs": [float(v) for v in contribs[k]],
            }

    meta = {
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "forbidden_loaded": [m for m in BLOCKED if sys.modules.get(m) is not None],
        "scipy_loaded": "scipy" in sys.modules,
    }
    json.dump({"meta": meta, "results": results}, open(results_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
