"""Unit tests for _crew.scoring: client-data problems are RecordError (422), results do not depend on key order.

The valid record is built from the bundle itself (first class of each categorical, 0.0 elsewhere), so these
tests need neither the legacy pickles nor the SQLite snapshot. Parity against the .pkl pipeline lives in
test_bundle_parity.py.
"""
import math
import pathlib
import random

import pytest
from _crew.bundle import BundleError, load_bundle
from _crew.scoring import MAX_PROBLEMS, RecordError, score_record

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "models" / "crew-v1"


@pytest.fixture(scope="module")
def bundle():
    return load_bundle(SOURCE)


@pytest.fixture()
def features(bundle):
    row = {}
    for name in bundle.feature_names:
        codes = bundle.categorical_codes.get(name)
        row[name] = next(iter(codes)) if codes else 0.0
    return row


def problems_of(bundle, record):
    with pytest.raises(RecordError) as caught:
        score_record(bundle, record)
    return caught.value.problems


def test_valid_record_scores_and_explains(bundle, features):
    result = score_record(bundle, features, sk_id_curr=100002)
    assert result["model_version"] == "crew-v1" and result["bundle_sha256"] == bundle.sha256
    assert result["sk_id_curr"] == 100002
    assert 0.0 < result["pd_raw"] < 1.0 and 0.0 <= result["pd_calibrated"] <= 1.0
    assert set(result["flags"]) == {"above_statistical", "above_business"}
    explanation = result["explanation"]
    assert abs(1 / (1 + math.exp(-explanation["margin"])) - result["pd_raw"]) < 1e-6
    assert len(explanation["increasing_risk"]) <= 3 and len(explanation["decreasing_risk"]) <= 2
    assert all(d["shap"] > 0 for d in explanation["increasing_risk"])
    assert all(d["shap"] < 0 for d in explanation["decreasing_risk"])
    shaps = [d["shap"] for d in explanation["increasing_risk"]]
    assert shaps == sorted(shaps, reverse=True)


def test_flags_use_greater_or_equal_on_the_raw_scale(bundle, features):
    result = score_record(bundle, features)
    assert result["flags"]["above_statistical"] == (result["pd_raw"] >= result["thresholds"]["statistical_optimal"])
    assert result["flags"]["above_business"] == (result["pd_raw"] >= result["thresholds"]["business_optimal"])
    assert result["thresholds"]["scale"] == "raw"


def test_key_order_does_not_change_the_result(bundle, features):
    reference = score_record(bundle, features, sk_id_curr="a")
    shuffled = list(features.items())
    random.Random(7).shuffle(shuffled)
    assert score_record(bundle, dict(shuffled), sk_id_curr="a") == reference
    assert score_record(bundle, dict(reversed(list(features.items()))), sk_id_curr="a") == reference


def without_driver_values(result):
    """Drivers echo the value as received ("Infinity", not the cleaned 10.0), so compare everything else."""
    explanation = result["explanation"]
    stripped = {k: [{x: y for x, y in d.items() if x != "value"} for d in v] if isinstance(v, list) else v for k, v in explanation.items()}
    return {**result, "explanation": stripped}


def test_cleaning_rules_come_from_the_bundle(bundle, features):
    baseline = score_record(bundle, {**features, "DEBT_TO_INCOME": 10.0, "INCOME_TO_CREDIT": -10.0, "EXT_SOURCE_MEAN": 0.0})
    cleaned = {**features, "DEBT_TO_INCOME": "Infinity", "INCOME_TO_CREDIT": "-Infinity", "EXT_SOURCE_MEAN": None}
    assert without_driver_values(score_record(bundle, cleaned)) == without_driver_values(baseline)
    in_process = {**features, "DEBT_TO_INCOME": math.inf, "INCOME_TO_CREDIT": -math.inf, "EXT_SOURCE_MEAN": math.nan}
    assert without_driver_values(score_record(bundle, in_process)) == without_driver_values(baseline)


def test_drivers_echo_the_value_as_received(bundle, features):
    explanation = score_record(bundle, features)["explanation"]
    drivers = explanation["increasing_risk"] + explanation["decreasing_risk"]
    assert drivers, "the all-default record should have at least one driver"
    assert all(d["value"] == features[d["feature"]] for d in drivers)


def test_non_finite_values_are_echoed_as_json_safe_text():
    from _crew.scoring import _echo

    assert _echo(math.inf) == "Infinity" and _echo(-math.inf) == "-Infinity"
    assert _echo(math.nan) is None and _echo("Infinity") == "Infinity" and _echo(0.25) == 0.25


def test_null_categorical_uses_missing_only_where_the_class_exists(bundle, features):
    assert score_record(bundle, {**features, "OCCUPATION_TYPE": None}) == score_record(bundle, {**features, "OCCUPATION_TYPE": "MISSING"})
    problems = problems_of(bundle, {**features, "NAME_INCOME_TYPE": None})
    assert [(p["code"], p["field"]) for p in problems] == [("null_not_allowed", "NAME_INCOME_TYPE")]


def test_unknown_category(bundle, features):
    problems = problems_of(bundle, {**features, "OCCUPATION_TYPE": "Astronaut"})
    assert [(p["code"], p["field"]) for p in problems] == [("unknown_category", "OCCUPATION_TYPE")]
    assert "'Astronaut'" in problems[0]["message"] and "19 known classes" in problems[0]["message"]


def test_missing_and_unknown_features_are_listed_together(bundle, features):
    record = {k: v for k, v in features.items() if k not in ("AMT_CREDIT", "EXT_SOURCE_MEAN")}
    record.update(SK_ID_CURR=100002, TARGET=1)
    codes = sorted((p["code"], p["field"]) for p in problems_of(bundle, record))
    assert codes == [
        ("missing_feature", "AMT_CREDIT"), ("missing_feature", "EXT_SOURCE_MEAN"),
        ("unknown_feature", "SK_ID_CURR"), ("unknown_feature", "TARGET"),
    ]


@pytest.mark.parametrize("field, value", [
    ("AMT_CREDIT", "12"), ("AMT_CREDIT", True), ("AMT_CREDIT", [1]), ("AMT_CREDIT", {}), ("AMT_CREDIT", "infinity"),
    ("AMT_CREDIT", 10 ** 400), ("CODE_GENDER", 5), ("CODE_GENDER", True), ("CODE_GENDER", ["F"]),
])
def test_wrong_types_are_invalid_value(bundle, features, field, value):
    problems = problems_of(bundle, {**features, field: value})
    assert [(p["code"], p["field"]) for p in problems] == [("invalid_value", field)]


def test_every_problem_is_reported_and_capped(bundle, features):
    codes = {p["code"] for p in problems_of(bundle, {**features, "AMT_CREDIT": "x", "CODE_GENDER": "Z", "NAME_INCOME_TYPE": None, "BOGUS": 1})}
    assert codes == {"invalid_value", "unknown_category", "null_not_allowed", "unknown_feature"}
    assert len(problems_of(bundle, {})) == MAX_PROBLEMS


def test_features_must_be_an_object(bundle):
    assert problems_of(bundle, [1, 2])[0]["field"] == "features"


def test_record_error_is_a_client_error_and_not_a_bundle_error():
    assert issubclass(RecordError, ValueError) and not issubclass(RecordError, BundleError)
