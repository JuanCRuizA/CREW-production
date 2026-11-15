"""Print a valid scoring request built from the bundle (same rule as the API tests). Usage: python deploy/make_sample.py > sample.json"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web", "api"))
from _crew.bundle import load_bundle  # noqa: E402

bundle = load_bundle()
features = {n: (next(iter(bundle.categorical_codes[n])) if n in bundle.categorical_codes else 0.0) for n in bundle.feature_names}
print(json.dumps({"sk_id_curr": 100002, "features": features}))
