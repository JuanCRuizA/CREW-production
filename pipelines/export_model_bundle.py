"""
Export the legacy .pkl artifacts to a portable JSON bundle (models/crew-v1/).

The bundle is loaded in production with xgboost + numpy only (no sklearn, pandas or shap).
Everything the scorer needs is stored explicitly: feature order, category -> code maps,
cleaning rules, calibrator points and thresholds. Nothing is refitted or recomputed here.
Requires sklearn at export time only, to unpickle the encoders and the calibrator.
"""
import argparse
import json
import pathlib
import shutil

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models" / "legacy"
BUNDLE_VERSION = "crew-v1"
BUNDLE_DIR = BASE_DIR / "models" / BUNDLE_VERSION
MIRROR_DIR = BASE_DIR / "web" / "api" / "_crew" / "bundle"  # byte-identical copy that ships with the Vercel function

N_FEATURES = 216
N_CATEGORICAL = 16
THRESHOLD_KEYS = {"statistical_optimal", "business_optimal", "default"}
CALCULATOR_ONLY = "what-if risk calculator only; never used by scoring"
CLEANING_NOTES = (
    "numeric_nan_value=0.0 is legacy behaviour (Streamlit fillna(0)), kept for parity. "
    "XGBoost handles NaN natively (learned default direction), so a future version may set it to null "
    "to pass NaN through."
)


def load(file):
    return joblib.load(MODEL_DIR / file)


def load_legacy() -> dict:
    """Load the 6 legacy pickles plus the CSV/JSON that ship with them."""
    medians = pd.read_csv(MODEL_DIR / "population_medians.csv", index_col=0).iloc[:, 0]
    return {
        "model": load("xgb_credit_model.pkl"),
        "feature_names": load("feature_names.pkl"),
        "encoders": load("label_encoders_v2.pkl"),
        "modes": load("categorical_modes.pkl"),
        "calibrator": load("calibrator.pkl"),
        "thresholds": load("thresholds.pkl"),
        "medians": medians,
        "model_card_path": MODEL_DIR / "model_card.json",
    }


def validate(art: dict) -> None:
    """Fail loudly if the legacy artifacts do not match the contract the bundle promises."""
    names = list(art["feature_names"])
    if len(names) != N_FEATURES or len(set(names)) != N_FEATURES:
        raise ValueError(f"feature_names must hold {N_FEATURES} unique names, got {len(names)}/{len(set(names))}")
    if art["model"].get_booster().feature_names != names:
        raise ValueError("feature_names.pkl order differs from the booster's own feature_names")

    encoders, modes = art["encoders"], art["modes"]
    if len(encoders) != N_CATEGORICAL:
        raise ValueError(f"expected {N_CATEGORICAL} label encoders, got {len(encoders)}")
    outside = [c for c in encoders if c not in names]
    if outside:
        raise ValueError(f"categorical columns not in feature_names: {outside}")
    for col, enc in encoders.items():
        classes = list(enc.classes_)
        if len(set(classes)) != len(classes):
            raise ValueError(f"{col}: duplicated classes_")
    if set(modes) != set(encoders):
        raise ValueError(f"categorical_modes keys differ from encoders: {set(modes) ^ set(encoders)}")
    bad_modes = {c: m for c, m in modes.items() if m not in list(encoders[c].classes_)}
    if bad_modes:
        raise ValueError(f"modes not present in classes_: {bad_modes}")

    cal = art["calibrator"]
    if not isinstance(cal, IsotonicRegression) or cal.out_of_bounds != "clip":
        raise ValueError("calibrator must be IsotonicRegression(out_of_bounds='clip')")
    if len(cal.X_thresholds_) != len(cal.y_thresholds_) or not np.all(np.diff(cal.X_thresholds_) > 0):
        raise ValueError("calibrator X_thresholds_ must be strictly increasing and match y_thresholds_ in length")

    if set(art["thresholds"]) != THRESHOLD_KEYS:
        raise ValueError(f"thresholds keys must be {sorted(THRESHOLD_KEYS)}, got {sorted(art['thresholds'])}")

    medians = art["medians"]
    if medians.isna().any() or not set(medians.index) <= set(names):
        raise ValueError("population_medians must have no NaN and only known feature names")


def write_json(path: pathlib.Path, obj: dict) -> None:
    """allow_nan=False: a stray NaN/inf must fail the export, not leak into the bundle."""
    text = json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8", newline="\n") as handle:  # newline="\n": Windows must not write CRLF into the bundle
        handle.write(text)


def build_preprocessing(art: dict) -> dict:
    names = list(art["feature_names"])
    encoders = art["encoders"]
    medians = art["medians"]
    return {
        "schema_version": 1,
        "model_version": BUNDLE_VERSION,
        "n_features": len(names),
        "feature_names": names,
        "categorical_features": list(encoders),
        "categorical_encoders": {col: {"classes": [str(c) for c in enc.classes_]} for col, enc in encoders.items()},
        "categorical_modes": {
            "usage": CALCULATOR_ONLY + " (not a null-imputation rule)",
            "source_file": "categorical_modes.pkl",
            "values": {col: str(mode) for col, mode in art["modes"].items()},
        },
        "cleaning": {
            "posinf_value": 10.0,
            "neginf_value": -10.0,
            "numeric_nan_value": 0.0,
            "categorical_null_label": "MISSING",
            "categorical_null_policy": "label_if_in_classes_else_error",
            "unknown_category_policy": "error",
            "missing_feature_policy": "error",
            "notes": CLEANING_NOTES,
        },
        "whatif_base_medians": {
            "usage": CALCULATOR_ONLY,
            "source_file": "population_medians.csv",
            "values": {name: float(value) for name, value in medians.items()},
            "not_covered_features": [n for n in names if n not in medians.index],
        },
    }


def build_calibration(art: dict) -> dict:
    cal = art["calibrator"]
    x = [float(v) for v in cal.X_thresholds_]  # float32 -> float64 is exact, no extra rounding
    y = [float(v) for v in cal.y_thresholds_]
    return {
        "schema_version": 1,
        "method": "isotonic",
        "apply": "np.interp(pd_raw, x_thresholds, y_thresholds)",
        "out_of_bounds": cal.out_of_bounds,
        "increasing": bool(cal.increasing_),
        "n_points": len(x),
        "x_thresholds": x,
        "y_thresholds": y,
        "x_min": x[0],
        "x_max": x[-1],
    }


def build_thresholds(art: dict) -> dict:
    return {
        "schema_version": 1,
        "scale": "raw",  # booster probability, before the isotonic calibrator
        "thresholds": {key: float(art["thresholds"][key]) for key in sorted(THRESHOLD_KEYS)},
    }


def export_bundle(art: dict, out_dir: pathlib.Path = BUNDLE_DIR, force: bool = False) -> dict[str, float]:
    """Write the 5 bundle files and return {file name: size in MB}."""
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        raise FileExistsError(f"{out_dir} already has files; pass --force to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)

    art["model"].get_booster().save_model(out_dir / "booster.json")
    write_json(out_dir / "preprocessing.json", build_preprocessing(art))
    write_json(out_dir / "calibration.json", build_calibration(art))
    write_json(out_dir / "thresholds.json", build_thresholds(art))
    # The legacy model card may have CRLF (written on Windows). Only the line endings are normalised to LF, so the
    # bundle has LF in every file (see .gitattributes); the content is otherwise unedited.
    card = pathlib.Path(art["model_card_path"]).read_bytes().replace(b"\r\n", b"\n")
    (out_dir / "model_card.json").write_bytes(card)

    return {f.name: f.stat().st_size / 1e6 for f in sorted(out_dir.iterdir())}


def mirror_bundle(src: pathlib.Path = BUNDLE_DIR, dst: pathlib.Path = MIRROR_DIR, force: bool = False) -> list[str]:
    """Copy the bundle files byte for byte to dst (the copy the Vercel function reads)."""
    if dst.exists() and any(dst.iterdir()):
        if not force:
            raise FileExistsError(f"{dst} already has files; pass --force to overwrite")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    names = sorted(f.name for f in src.iterdir() if f.is_file())
    for name in names:
        shutil.copyfile(src / name, dst / name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite an existing bundle and its mirror")
    parser.add_argument("--mirror-only", action="store_true", help="skip the export and only copy the existing bundle to the mirror")
    parser.add_argument("--mirror-dir", type=pathlib.Path, default=MIRROR_DIR, help="where the byte-identical copy goes")
    args = parser.parse_args()

    if not args.mirror_only:
        art = load_legacy()
        validate(art)
        sizes = export_bundle(art, force=args.force)
        for name, mb in sizes.items():
            print(f"{name:<20}{mb:>8.3f} MB")
        print(f"{'TOTAL':<20}{sum(sizes.values()):>8.3f} MB  ->  {BUNDLE_DIR}")
    names = mirror_bundle(BUNDLE_DIR, args.mirror_dir, force=args.force)
    print(f"mirror: {len(names)} files  ->  {args.mirror_dir}")


if __name__ == "__main__":
    main()
