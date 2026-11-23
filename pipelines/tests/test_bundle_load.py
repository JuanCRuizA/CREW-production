"""BundleError tests: any defect in the bundle stops the load with its cause (HTTP 500).

None of these may look like bad client data, so BundleError must not be a ValueError.
Every case works on a copy of models/crew-v1 in a temporary directory.
"""
import json
import pathlib
import shutil

import pytest
from _crew.bundle import SCORING_FILES, BundleError, load_bundle

SOURCE = pathlib.Path(__file__).resolve().parents[2] / "models" / "crew-v1"


@pytest.fixture()
def bundle_copy(tmp_path):
    destination = tmp_path / "crew-v1"
    shutil.copytree(SOURCE, destination)
    return destination


def edit(bundle_dir, name, change):
    path = bundle_dir / name
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def _swap_first_two_features(d):
    d["feature_names"][0], d["feature_names"][1] = d["feature_names"][1], d["feature_names"][0]


def _swap_first_two_x(d):
    d["x_thresholds"][0], d["x_thresholds"][1] = d["x_thresholds"][1], d["x_thresholds"][0]


def test_valid_bundle_loads():
    bundle = load_bundle(SOURCE)
    assert bundle.model_version == "crew-v1"
    assert len(bundle.feature_names) == 216 and len(bundle.categorical_codes) == 16
    assert len(bundle.sha256) == 64
    assert bundle.thresholds["business_optimal"] == pytest.approx(0.79)


def test_bundle_error_is_not_a_client_data_error():
    assert not issubclass(BundleError, ValueError)


def test_hash_covers_scoring_files_but_not_the_model_card(bundle_copy):
    reference = load_bundle(bundle_copy).sha256
    (bundle_copy / "model_card.json").write_text("{}", encoding="utf-8")
    assert load_bundle(bundle_copy).sha256 == reference
    edit(bundle_copy, "thresholds.json", lambda d: d["thresholds"].update(default=0.51))
    assert load_bundle(bundle_copy).sha256 != reference


@pytest.mark.parametrize("name", SCORING_FILES)
def test_missing_file(bundle_copy, name):
    (bundle_copy / name).unlink()
    with pytest.raises(BundleError, match=f"missing bundle file: {name}"):
        load_bundle(bundle_copy)


@pytest.mark.parametrize("name", ["preprocessing.json", "calibration.json", "thresholds.json"])
@pytest.mark.parametrize("content", ["{not json", '{"schema_version": NaN}'])
def test_corrupt_json(bundle_copy, name, content):
    (bundle_copy / name).write_text(content, encoding="utf-8")
    with pytest.raises(BundleError, match=f"{name} is not valid JSON"):
        load_bundle(bundle_copy)


def test_corrupt_booster(bundle_copy):
    (bundle_copy / "booster.json").write_text("garbage", encoding="utf-8")
    with pytest.raises(BundleError, match="booster.json could not be loaded"):
        load_bundle(bundle_copy)


CONTENT_DEFECTS = [
    ("fill missing features", "preprocessing.json", lambda d: d["cleaning"].update(missing_feature_policy="fill"),
     "cleaning.missing_feature_policy must be 'error', got 'fill'"),
    ("unknown category as zero", "preprocessing.json", lambda d: d["cleaning"].update(unknown_category_policy="zero"),
     "cleaning.unknown_category_policy must be 'error', got 'zero'"),
    ("categorical null policy", "preprocessing.json", lambda d: d["cleaning"].update(categorical_null_policy="always_label"),
     "cleaning.categorical_null_policy must be 'label_if_in_classes_else_error'"),
    ("numeric nan as null (future)", "preprocessing.json", lambda d: d["cleaning"].update(numeric_nan_value=None),
     "numeric_nan_value must be a finite number"),
    ("posinf as text", "preprocessing.json", lambda d: d["cleaning"].update(posinf_value="inf"),
     "posinf_value must be a finite number"),
    ("schema 2 preprocessing", "preprocessing.json", lambda d: d.update(schema_version=2), "schema_version must be 1, got 2"),
    ("schema 2 calibration", "calibration.json", lambda d: d.update(schema_version=2), "schema_version must be 1, got 2"),
    ("schema 2 thresholds", "thresholds.json", lambda d: d.update(schema_version=2), "schema_version must be 1, got 2"),
    ("duplicated classes", "preprocessing.json",
     lambda d: d["categorical_encoders"]["CODE_GENDER"]["classes"].append("F"), "CODE_GENDER has duplicated classes"),
    ("n_features mismatch", "preprocessing.json", lambda d: d.update(n_features=215), "n_features (215)"),
    ("categorical list without a column", "preprocessing.json",
     lambda d: d["categorical_features"].remove("CODE_GENDER"), "categorical_features must list exactly"),
    ("feature order differs from booster", "preprocessing.json", _swap_first_two_features,
     "booster.json feature_names differ"),
    ("calibration oob not clip", "calibration.json", lambda d: d.update(out_of_bounds="nan"), "out_of_bounds must be 'clip'"),
    ("calibration x not increasing", "calibration.json", _swap_first_two_x, "strictly increasing"),
    ("calibration y out of range", "calibration.json", lambda d: d["y_thresholds"].__setitem__(-1, 1.5), "within [0, 1]"),
    ("calibration n_points", "calibration.json", lambda d: d.update(n_points=10), "n_points (10)"),
    ("thresholds scale", "thresholds.json", lambda d: d.update(scale="calibrated"), "scale must be 'raw'"),
    ("thresholds missing key", "thresholds.json", lambda d: d["thresholds"].pop("business_optimal"), "thresholds must hold exactly"),
]


@pytest.mark.parametrize("label, name, change, message", CONTENT_DEFECTS, ids=[c[0] for c in CONTENT_DEFECTS])
def test_content_defect(bundle_copy, label, name, change, message):
    edit(bundle_copy, name, change)
    with pytest.raises(BundleError) as caught:
        load_bundle(bundle_copy)
    assert message in str(caught.value)
