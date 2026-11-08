"""Parity test: JSON bundle (xgboost + numpy only) vs the legacy .pkl pipeline.

Reference path (sklearn, pandas, shap): label_encoders_v2.pkl, reindex fill 0, inf -> +/-10,
NaN -> 0, XGBClassifier.predict_proba, np.interp over the calibrator points and
shap.TreeExplainer. Same rules as scripts/score_local.py (variant "fillna_MISSING"), except
that an unseen label raises (LabelEncoder.transform) instead of becoming 0.

Bundle path: the deployed package web/api/_crew with its mirrored bundle, run by tests/bundle_worker.py in a
subprocess that blocks sklearn, pandas and shap. The worker holds no scoring logic of its own.

Clients come from the legacy SQLite snapshot opened with mode=ro (loan_predictions is not used).
Run with: python -m pytest pipelines/tests/test_bundle_parity.py -s
"""
import json
import pathlib
import sqlite3
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import pytest
import shap

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "legacy"
BUNDLE_DIR = ROOT / "models" / "crew-v1"
DEPLOYED_BUNDLE = ROOT / "web" / "api" / "_crew" / "bundle"  # byte-identical mirror, see test_bundle_mirror.py
DB_PATH = ROOT / "data" / "legacy" / "portfolio_surveillance.db"
WORKER = pathlib.Path(__file__).with_name("bundle_worker.py")

TOL = 1e-6
SUM_TOL = 1e-5  # sum(contribs) vs margin: both float32, 217 terms accumulate rounding
SEED = 42
N_PER_GROUP = 250  # the requirement is >= 200
MIN_PER_GROUP = 200
POOL_WITH_NULLS = 4000

pytestmark = pytest.mark.skipif(
    not (BUNDLE_DIR / "booster.json").exists() or not (DEPLOYED_BUNDLE / "booster.json").exists() or not DB_PATH.exists(),
    reason="needs models/crew-v1, its mirror (pipelines/export_model_bundle.py) and data/legacy/portfolio_surveillance.db",
)


# ---------------------------------------------------------------- reference (.pkl) path
@pytest.fixture(scope="module")
def legacy():
    model = joblib.load(MODEL_DIR / "xgb_credit_model.pkl")
    return {
        "model": model,
        "names": list(joblib.load(MODEL_DIR / "feature_names.pkl")),
        "encoders": joblib.load(MODEL_DIR / "label_encoders_v2.pkl"),
        "modes": joblib.load(MODEL_DIR / "categorical_modes.pkl"),
        "calibrator": joblib.load(MODEL_DIR / "calibrator.pkl"),
        "thresholds": joblib.load(MODEL_DIR / "thresholds.pkl"),
        "explainer": shap.TreeExplainer(model),
    }


def reference_matrix(df: pd.DataFrame, legacy: dict) -> pd.DataFrame:
    """Legacy preprocessing. Raises ValueError for a label the encoder has never seen."""
    X = df.copy()
    for col, le in legacy["encoders"].items():
        X[col] = le.transform(X[col].astype(object).where(X[col].notna(), "MISSING"))
    X = X.reindex(columns=legacy["names"], fill_value=0)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X.replace([np.inf, -np.inf], [10, -10]).fillna(0)


def reference_pd_raw(X: pd.DataFrame, legacy: dict) -> np.ndarray:
    return legacy["model"].predict_proba(X)[:, 1]


def reference_score(df: pd.DataFrame, legacy: dict) -> dict:
    X = reference_matrix(df, legacy)
    pd_raw = reference_pd_raw(X, legacy)
    cal = legacy["calibrator"]
    shap_values = legacy["explainer"].shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    return {
        "pd_raw": pd_raw,
        "pd_cal": np.interp(pd_raw.astype(np.float64), cal.X_thresholds_, cal.y_thresholds_),
        "pd_cal_sklearn": cal.predict(pd_raw),
        "shap": np.asarray(shap_values),
        "bias": float(np.ravel(legacy["explainer"].expected_value)[0]),
        "names": legacy["names"],
    }


def reference_drivers(shap_row: np.ndarray, names: list) -> tuple:
    """Top 3 risk-increasing and top 2 risk-decreasing features, same rule as the API (sign filter, stable ties)."""
    up = [i for i in np.argsort(-shap_row, kind="stable")[:3] if shap_row[i] > 0]
    down = [i for i in np.argsort(shap_row, kind="stable")[:2] if shap_row[i] < 0]
    return [(names[i], float(shap_row[i])) for i in up], [(names[i], float(shap_row[i])) for i in down]


# ---------------------------------------------------------------- bundle path
def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serialisable: {type(value)}")


def run_bundle(records: list[dict]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "records.json").write_text(json.dumps(records, default=_json_default), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "-I", str(WORKER), str(DEPLOYED_BUNDLE), str(tmp / "records.json"), str(tmp / "results.json")],
            capture_output=True,
            text=True,
        )
        if proc.stderr:
            sys.stderr.write(proc.stderr)  # never swallow xgboost / numpy warnings
        assert proc.returncode == 0, f"bundle worker failed:\n{proc.stderr}"
        return json.loads((tmp / "results.json").read_text(encoding="utf-8"))


def to_records(df: pd.DataFrame) -> list[dict]:
    return df.astype(object).where(df.notna(), None).to_dict("records")


def compare(out: dict, ref: dict) -> dict:
    results = out["results"]
    errors = [r["error"] for r in results if "error" in r]
    assert not errors, f"bundle rejected valid records, first error: {errors[0]}"
    contribs = np.array([r["contribs"] for r in results])
    pd_cal = np.array([r["pd_calibrated"] for r in results])
    margins = np.array([r["explanation"]["margin"] for r in results])
    drivers = 0.0
    for k, result in enumerate(results):
        expected = reference_drivers(ref["shap"][k], ref["names"])
        for key, wanted in zip(("increasing_risk", "decreasing_risk"), expected):
            got = result["explanation"][key]
            assert [d["feature"] for d in got] == [name for name, _ in wanted], f"record {k}: {key} features or order differ"
            drivers = max([drivers] + [abs(d["shap"] - value) for d, (_, value) in zip(got, wanted)])
    return {
        "n": len(results),
        "pd_raw": np.abs(np.array([r["pd_raw"] for r in results]) - ref["pd_raw"]).max(),
        "pd_cal": np.abs(pd_cal - ref["pd_cal"]).max(),
        "pd_cal_vs_sklearn": np.abs(pd_cal - ref["pd_cal_sklearn"]).max(),
        "shap": np.abs(contribs[:, :-1] - ref["shap"]).max(),
        "bias": np.abs(contribs[:, -1] - ref["bias"]).max(),
        "drivers": drivers,
        "margin_sum": np.abs(contribs.sum(axis=1) - margins).max(),
    }


# ---------------------------------------------------------------- data (SQLite, read-only)
def _connect():
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)


def _pick_stratified(df: pd.DataFrame, pd_raw: np.ndarray, n: int) -> pd.DataFrame:
    """n rows evenly spread over the sorted PD range, so low, middle and high risk are all covered."""
    order = np.argsort(pd_raw, kind="stable")
    return df.iloc[order[np.linspace(0, len(order) - 1, n).round().astype(int)]]


@pytest.fixture(scope="module")
def clients(legacy):
    cols = ", ".join(f'"{c}"' for c in legacy["names"])
    with _connect() as con:
        no_nulls = pd.read_sql_query(
            f"SELECT {cols} FROM engineered_features WHERE " + " AND ".join(f'"{c}" IS NOT NULL' for c in legacy["names"]),
            con,
        )
        ids = [r[0] for r in con.execute("SELECT SK_ID_CURR FROM engineered_features")]
        sample = np.random.default_rng(SEED).choice(ids, size=POOL_WITH_NULLS, replace=False)
        pool = pd.read_sql_query(
            f"SELECT {cols} FROM engineered_features WHERE SK_ID_CURR IN ({', '.join(str(int(i)) for i in sample)})", con
        )
    with_nulls = pool[pool.isna().any(axis=1)].reset_index(drop=True)
    picked = {}
    for label, frame in (("sin nulos", no_nulls), ("con nulos", with_nulls)):
        pd_raw = reference_pd_raw(reference_matrix(frame, legacy), legacy)
        picked[label] = _pick_stratified(frame, pd_raw, N_PER_GROUP).reset_index(drop=True)
    return picked


@pytest.fixture(scope="module")
def group_results(clients, legacy):
    results = {}
    for label, df in clients.items():
        out = run_bundle(to_records(df))
        results[label] = {"diffs": compare(out, reference_score(df, legacy)), "meta": out["meta"], "df": df, "out": out}
    return results


# ---------------------------------------------------------------- edge cases
INF = float("inf")
EDGE_OK = {
    "+inf DEBT_TO_INCOME": {"DEBT_TO_INCOME": INF},
    "-inf INCOME_TO_CREDIT": {"INCOME_TO_CREDIT": -INF},
    "NaN EXT_SOURCE_MEAN": {"EXT_SOURCE_MEAN": None},
    "null OCCUPATION_TYPE (has MISSING)": {"OCCUPATION_TYPE": None},
    "inf + NaN + null combined": {"DEBT_TO_INCOME": INF, "INCOME_TO_CREDIT": -INF, "EXT_SOURCE_MEAN": None, "HOUSETYPE_MODE": None},
}
EDGE_REJECTED = {  # case: (patch, problem code the deployed package must report)
    "unknown category OCCUPATION_TYPE": ({"OCCUPATION_TYPE": "Astronaut"}, "unknown_category"),
    "unknown category CODE_GENDER": ({"CODE_GENDER": "Z"}, "unknown_category"),
    "null NAME_INCOME_TYPE (no MISSING class)": ({"NAME_INCOME_TYPE": None}, "null_not_allowed"),
}


@pytest.fixture(scope="module")
def edge_base(clients, legacy):
    base = to_records(clients["sin nulos"].iloc[[len(clients["sin nulos"]) // 2]])[0]
    touched = {k for case in (*EDGE_OK.values(), *(patch for patch, _ in EDGE_REJECTED.values())) for k in case}
    assert touched <= set(legacy["names"]), f"edge cases use unknown features: {touched - set(legacy['names'])}"
    return base


# ---------------------------------------------------------------- tests
def test_bundle_scoring_imports_no_sklearn_pandas_shap(group_results):
    for label, res in group_results.items():
        assert res["meta"]["forbidden_loaded"] == [], f"{label}: {res['meta']['forbidden_loaded']}"


@pytest.mark.parametrize("label", ["sin nulos", "con nulos"])
def test_group_parity(group_results, label):
    d = group_results[label]["diffs"]
    assert d["n"] >= MIN_PER_GROUP
    for metric in ("pd_raw", "pd_cal", "pd_cal_vs_sklearn", "shap", "bias", "drivers"):
        assert d[metric] <= TOL, f"{label}: {metric} max abs_diff {d[metric]:.3e} > {TOL}"
    assert d["margin_sum"] <= SUM_TOL, f"{label}: sum(contribs) vs margin {d['margin_sum']:.3e}"


def test_group_with_nulls_covers_both_kinds_of_null(group_results, legacy):
    df = group_results["con nulos"]["df"]
    cat = list(legacy["encoders"])
    num = [c for c in legacy["names"] if c not in cat]
    assert df[cat].isna().any(axis=1).sum() >= 50, "too few clients with a null categorical"
    assert df[num].isna().any(axis=1).sum() >= 50, "too few clients with a null numeric"


def test_edge_cases_parity(edge_base, legacy):
    records = [{**edge_base, **patch} for patch in EDGE_OK.values()]
    out = run_bundle(records)
    ref = reference_score(pd.DataFrame.from_records(records, columns=legacy["names"]), legacy)
    d = compare(out, ref)
    for metric in ("pd_raw", "pd_cal", "pd_cal_vs_sklearn", "shap", "bias", "drivers"):
        assert d[metric] <= TOL, f"edge cases: {metric} max abs_diff {d[metric]:.3e} > {TOL}"


@pytest.mark.parametrize("case", list(EDGE_REJECTED))
def test_unknown_or_unmappable_category_is_an_error_in_both_paths(edge_base, legacy, case):
    patch, code = EDGE_REJECTED[case]
    record = {**edge_base, **patch}
    with pytest.raises(ValueError):
        reference_matrix(pd.DataFrame.from_records([record], columns=legacy["names"]), legacy)
    result = run_bundle([record])["results"][0]
    assert "error" in result, f"bundle scored a record it should reject: {case}"
    assert [(p["code"], p["field"]) for p in result["problems"]] == [(code, next(iter(patch)))]


def test_missing_feature_key_is_an_error(edge_base):
    record = {k: v for k, v in edge_base.items() if k != "EXT_SOURCE_MEAN"}
    result = run_bundle([record])["results"][0]
    assert [(p["code"], p["field"]) for p in result["problems"]] == [("missing_feature", "EXT_SOURCE_MEAN")]


def test_calibrator_json_matches_sklearn_and_clips(legacy):
    cal = json.loads((BUNDLE_DIR / "calibration.json").read_text(encoding="utf-8"))
    # sklearn casts its input to float32 and the calibrator slope reaches ~4e4, so a float64 grid would differ
    # by up to ~2e-5 from rounding alone. booster.predict returns float32, so real PDs are float32-exact.
    probs = np.concatenate([[0.0, cal["x_min"] / 2, cal["x_max"] + 0.01, 1.0], np.linspace(0, 1, 1001)])
    probs = probs.astype(np.float32).astype(np.float64)
    got = np.interp(probs, cal["x_thresholds"], cal["y_thresholds"])
    assert np.abs(got - legacy["calibrator"].predict(probs)).max() <= TOL
    assert got[0] == cal["y_thresholds"][0] and got[3] == cal["y_thresholds"][-1]  # clip at both ends


def test_bundle_files_carry_the_legacy_contract(legacy):
    pre = json.loads((BUNDLE_DIR / "preprocessing.json").read_text(encoding="utf-8"))
    cal = json.loads((BUNDLE_DIR / "calibration.json").read_text(encoding="utf-8"))
    thr = json.loads((BUNDLE_DIR / "thresholds.json").read_text(encoding="utf-8"))
    assert pre["feature_names"] == legacy["names"] and pre["n_features"] == 216
    assert pre["categorical_features"] == list(legacy["encoders"])
    for col, enc in legacy["encoders"].items():
        assert pre["categorical_encoders"][col]["classes"] == [str(c) for c in enc.classes_]
    assert pre["categorical_modes"]["values"] == legacy["modes"]
    assert cal["x_thresholds"] == [float(v) for v in legacy["calibrator"].X_thresholds_] and cal["n_points"] == 236
    assert thr["scale"] == "raw" and thr["thresholds"] == {k: float(v) for k, v in legacy["thresholds"].items()}
    # the exporter only normalises CRLF to LF when copying the legacy card; nothing else may differ
    assert (BUNDLE_DIR / "model_card.json").read_bytes() == (MODEL_DIR / "model_card.json").read_bytes().replace(b"\r\n", b"\n")


def test_report(group_results, capsys):
    sizes = {p.name: p.stat().st_size / 1e6 for p in sorted(BUNDLE_DIR.iterdir())}
    with capsys.disabled():
        print("\n\nmax abs_diff, bundle vs .pkl pipeline (tolerance 1e-6)")
        print(f"{'grupo':<11}{'n':>5}{'pd_raw':>11}{'pd_cal':>11}{'cal_sklearn':>13}{'shap':>11}{'bias':>11}{'drivers':>11}{'sum~margin':>12}")
        for label, res in group_results.items():
            d = res["diffs"]
            print(
                f"{label:<11}{d['n']:>5}{d['pd_raw']:>11.2e}{d['pd_cal']:>11.2e}{d['pd_cal_vs_sklearn']:>13.2e}"
                f"{d['shap']:>11.2e}{d['bias']:>11.2e}{d['drivers']:>11.2e}{d['margin_sum']:>12.2e}"
            )
        meta = next(iter(group_results.values()))["meta"]
        print(f"worker: xgboost {meta['xgboost']}, numpy {meta['numpy']}, forbidden modules loaded: {meta['forbidden_loaded']}, "
              f"scipy loaded: {meta['scipy_loaded']}")
        print("bundle size: " + ", ".join(f"{k} {v:.3f}" for k, v in sizes.items()) + f"  |  TOTAL {sum(sizes.values()):.3f} MB")
