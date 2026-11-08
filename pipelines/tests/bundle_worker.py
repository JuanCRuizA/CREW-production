"""Run the deployed scoring package (web/api/_crew) with sklearn, pandas and shap blocked.

The parity test calls this in a subprocess: it proves the code that ships needs none of those
libraries, and it holds no scoring logic of its own, so what is tested is what is deployed.

usage: python bundle_worker.py <bundle_dir> <records.json> <results.json>
"""
import json
import sys
from pathlib import Path

BLOCKED = ("sklearn", "pandas", "shap")
for _name in BLOCKED:
    sys.modules[_name] = None  # any `import sklearn` now raises ImportError
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web" / "api"))

import numpy as np  # noqa: E402
import xgboost as xgb  # noqa: E402
from _crew.bundle import load_bundle  # noqa: E402
from _crew.scoring import RecordError, encode_record, predict_row, score_record  # noqa: E402


def main(bundle_dir: str, records_path: str, results_path: str) -> None:
    bundle = load_bundle(Path(bundle_dir))  # a BundleError stops the worker with its traceback
    results = []
    for record in json.load(open(records_path, encoding="utf-8")):
        try:
            result = score_record(bundle, record)
            result["contribs"] = [float(v) for v in predict_row(bundle, encode_record(bundle, record))[2]]
        except RecordError as exc:
            result = {"error": str(exc), "problems": exc.problems}
        results.append(result)
    meta = {
        "xgboost": xgb.__version__,
        "numpy": np.__version__,
        "forbidden_loaded": [m for m in BLOCKED if sys.modules.get(m) is not None],
        "scipy_loaded": "scipy" in sys.modules,
    }
    json.dump({"meta": meta, "results": results}, open(results_path, "w", encoding="utf-8"))


if __name__ == "__main__":
    main(*sys.argv[1:4])
