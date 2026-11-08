"""Server-side HTML pages: client list, client detail and re-score (Jinja, no JavaScript).

Data access goes through `repository` and scoring through `services`; both are imported as modules at call
time so tests can replace them. No headline "PD": pd_raw and pd_calibrated are always shown side by side
(crew-v1 decision). Features are never logged or rendered beyond the SHAP drivers.
"""
from flask import Blueprint, current_app, redirect, render_template, request, url_for

PAGE_SIZE = 50

bp = Blueprint("views", __name__, template_folder="templates", static_folder="static", static_url_path="/static/views")


def _error(status: int, title: str, message: str, problems: list = None):
    return render_template("error.html", status=status, title=title, message=message, problems=problems), status


def _no_database(exc):
    return _error(503, "Database not configured", f"The client database is not available: {exc}")


@bp.app_context_processor
def _model_context() -> dict:
    """Footer data: the model version of the loaded bundle, or the reason it is unusable."""
    bundle = current_app.extensions.get("crew_bundle")
    return {"footer_model_version": bundle.model_version if bundle is not None else "bundle not loaded"}


@bp.get("/")
def clients():
    import repository

    raw_page = request.args.get("page", "1")
    if not raw_page.isdigit() or int(raw_page) < 1:
        return _error(400, "Invalid page", f"'page' must be a positive integer, got {raw_page!r}")
    page = int(raw_page)
    try:
        total = repository.count_clients()
        rows = repository.list_clients(PAGE_SIZE, (page - 1) * PAGE_SIZE)
    except repository.DatabaseNotConfigured as exc:
        return _no_database(exc)
    n_pages = max(1, -(-total // PAGE_SIZE))  # ceiling division; an empty table still has one (empty) page
    return render_template("clients.html", rows=rows, page=page, n_pages=n_pages, total=total)


@bp.get("/clients/<int:sk_id_curr>")
def client(sk_id_curr: int):
    import repository

    try:
        info = repository.get_client(sk_id_curr)
        if info is None:
            return _error(404, "Client not found", f"No client with SK_ID_CURR {sk_id_curr} in the sample.")
        score = repository.latest_score(sk_id_curr)
    except repository.DatabaseNotConfigured as exc:
        return _no_database(exc)
    return render_template("client.html", client=info, score=score)


@bp.post("/clients/<int:sk_id_curr>/rescore")
def rescore(sk_id_curr: int):
    import repository
    import services
    from crew.scoring import RecordError

    bundle_error = current_app.extensions.get("crew_bundle_error")
    if bundle_error is not None:
        return _error(500, "Model bundle unusable", f"The model bundle is not usable: {bundle_error}")
    try:
        services.score_client(current_app.extensions["crew_bundle"], sk_id_curr, source="ui")
    except services.ClientNotFound:
        return _error(404, "Client not found", f"No client with SK_ID_CURR {sk_id_curr} in the sample.")
    except RecordError as exc:
        return _error(422, "Client data cannot be scored", str(exc), problems=exc.problems)
    except repository.DatabaseNotConfigured as exc:
        return _no_database(exc)
    return redirect(url_for("views.client", sk_id_curr=sk_id_curr), code=303)  # PRG: a reload never re-scores
