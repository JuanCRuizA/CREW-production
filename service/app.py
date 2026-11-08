"""Flask application factory for the CREW service (Render runs: gunicorn --chdir service "app:create_app()").

service/ is the import root: modules are app, api, repository, services and the crew scoring package.
The model bundle is loaded once, when this module is imported. A defect is kept and reported on every request
(HTTP 500 bundle_error) instead of stopping the process, so /health can say why the service is broken.
"""
import os
import sys

from flask import Flask

import api
from crew.bundle import BundleError, load_bundle

try:
    BUNDLE, BUNDLE_ERROR = load_bundle(), None
except BundleError as exc:
    BUNDLE, BUNDLE_ERROR = None, exc
    print(f"BUNDLE ERROR: {exc}", file=sys.stderr)


def create_app(config: dict | None = None) -> Flask:
    """Build the app. Tests inject a bundle by passing CREW_BUNDLE and CREW_BUNDLE_ERROR together in `config`."""
    config = dict(config or {})
    if "CREW_BUNDLE" in config or "CREW_BUNDLE_ERROR" in config:
        bundle, bundle_error = config.pop("CREW_BUNDLE"), config.pop("CREW_BUNDLE_ERROR")
    else:
        bundle, bundle_error = BUNDLE, BUNDLE_ERROR

    secret = os.environ.get("FLASK_SECRET_KEY")
    if "RENDER" in os.environ and not secret:  # Render sets RENDER on every service; the UI's flash messages need it
        raise RuntimeError("FLASK_SECRET_KEY is not set: define it in the Render environment variables")

    app = Flask(__name__)
    if secret:
        app.config["SECRET_KEY"] = secret
    app.config.update(config)
    app.extensions["crew_bundle"] = bundle
    app.extensions["crew_bundle_error"] = bundle_error

    app.register_blueprint(api.bp)
    # --- UI blueprint (owned by the ui agent): register it here, right after the API ---

    return app
