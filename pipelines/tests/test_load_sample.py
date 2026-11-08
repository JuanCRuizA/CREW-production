"""Unit tests for pipelines/load_sample.py: row conversion and stratified sampling. No database, no legacy SQLite."""
import json
import math
import sqlite3

import pytest
from load_sample import (
    SampleError,
    allocate,
    build_features,
    build_sample,
    load_feature_spec,
    stratified_ids,
    to_api_value,
)

FEATURE_SET, FEATURE_NAMES = load_feature_spec()


def test_feature_spec_is_the_bundle():
    assert FEATURE_SET == "crew-v1"
    assert len(FEATURE_NAMES) == 216 and len(set(FEATURE_NAMES)) == 216


@pytest.mark.parametrize("value, expected", [
    (None, None), ("Cash loans", "Cash loans"), (3, 3), (0.25, 0.25),
    (math.nan, None), (math.inf, "Infinity"), (-math.inf, "-Infinity"),
])
def test_to_api_value(value, expected):
    assert to_api_value(value) == expected


@pytest.mark.parametrize("value", [True, b"bytes", [1]])
def test_to_api_value_rejects_unexpected_types(value):
    with pytest.raises(SampleError):
        to_api_value(value)


def test_build_features_keeps_bundle_order_and_is_strict_json():
    row = {name: 1.5 for name in reversed(FEATURE_NAMES)}
    row[FEATURE_NAMES[0]] = math.nan
    row[FEATURE_NAMES[1]] = math.inf
    features = build_features(row, FEATURE_NAMES)
    assert list(features) == FEATURE_NAMES
    assert features[FEATURE_NAMES[0]] is None and features[FEATURE_NAMES[1]] == "Infinity"
    json.dumps(features, allow_nan=False)  # raises if a NaN/inf float slipped through


def test_build_features_fails_on_a_missing_feature():
    row = {name: 0 for name in FEATURE_NAMES[1:]}
    with pytest.raises(SampleError, match="missing 1 features"):
        build_features(row, FEATURE_NAMES)


def test_allocate_is_proportional_and_exact():
    assert allocate({0: 282686, 1: 24825}, 2000) == {0: 1839, 1: 161}
    assert sum(allocate({0: 7, 1: 3}, 5).values()) == 5


def test_allocate_rejects_impossible_sizes():
    with pytest.raises(SampleError):
        allocate({0: 5, 1: 5}, 11)
    with pytest.raises(SampleError):
        allocate({0: 5, 1: 5}, 0)


POPULATION = [(100000 + i, 1 if i % 10 == 0 else 0) for i in range(5000)]


def test_stratified_ids_keeps_class_proportions():
    picked = stratified_ids(POPULATION, 500, seed=42)
    assert len(picked) == 500 and len({sk for sk, _ in picked}) == 500
    assert sum(t for _, t in picked) == 50


def test_stratified_ids_is_deterministic_for_a_seed():
    assert stratified_ids(POPULATION, 300, seed=42) == stratified_ids(list(reversed(POPULATION)), 300, seed=42)
    assert stratified_ids(POPULATION, 300, seed=42) != stratified_ids(POPULATION, 300, seed=7)


def test_stratified_ids_rejects_bad_targets():
    with pytest.raises(SampleError, match="TARGET"):
        stratified_ids([(1, 0), (2, None)], 1, seed=42)


def _toy_source(columns):
    conn = sqlite3.connect(":memory:")
    conn.execute("create table engineered_features (" + ", ".join(f'"{c}"' for c in columns) + ")")
    return conn


def test_build_sample_fails_when_the_source_lacks_a_feature():
    conn = _toy_source(["SK_ID_CURR", "TARGET", *FEATURE_NAMES[:-1]])
    with pytest.raises(SampleError, match=FEATURE_NAMES[-1]):
        build_sample(conn, FEATURE_NAMES, 1, seed=42)


def test_build_sample_end_to_end_on_a_toy_source():
    names = FEATURE_NAMES[:3]
    conn = _toy_source(["SK_ID_CURR", "TARGET", *names])
    rows = [(i, i % 2, float(i), None, "x") for i in range(1, 21)]
    conn.executemany("insert into engineered_features values (?, ?, ?, ?, ?)", rows)
    sample = build_sample(conn, names, 10, seed=42)
    assert len(sample) == 10 and sum(t for _, t, _ in sample) == 5
    sk_id, target, features = sample[0]
    assert features == {names[0]: float(sk_id), names[1]: None, names[2]: "x"}
