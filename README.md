# CREW - Credit Risk Early Warning system (production)

Production deployment of CREW, the XGBoost credit-risk model built on the Kaggle *Home Credit Default Risk* dataset (307,511 applications, 216 features; AUC 0.7778, Gini 0.5556, Brier 0.0668 on the hold-out set).

**Live scoring API:** https://129-38-95-215.sslip.io/api/score

## Architecture

- **Host**: one Linux VM on Oracle Cloud Infrastructure (Ubuntu 24.04, Ampere ARM, Bogota region).
- **Scoring API**: Python, XGBoost with TreeSHAP explanations, loaded from a portable JSON model bundle (`models/crew-v1`, no pickles in production). Runs in a Docker container that only exposes port 8000 to the internal network.
- **Edge**: Caddy as reverse proxy, with HTTPS certificates from Let's Encrypt issued and renewed automatically.
- **CI/CD**: GitHub Actions. Every push to `dev` runs the test suite on an ARM runner (same architecture as production). Only if it passes, the pipeline deploys over SSH with a dedicated key and smoke-tests the live API.

```
client --HTTPS--> Caddy :443 --HTTP--> api :8000 (XGBoost + SHAP)
```

## Using the API

`GET /api/score` describes the model: version, bundle SHA-256 and the 216 expected features.

`POST /api/score` scores one applicant:

```bash
curl -X POST https://129-38-95-215.sslip.io/api/score \
  -H "Content-Type: application/json" \
  --data @sample.json
```

The response contains `pd_raw` (booster output, inflated by `scale_pos_weight`), `pd_calibrated` (isotonic calibration), flags against the statistical (0.5157) and business (0.79) thresholds, and the SHAP drivers that increase and decrease risk. Features are never logged.
Latency: p50 42 ms, p95 43 ms (100 sequential `POST /api/score` calls made from the VM itself, through Caddy and HTTPS, 23 Nov 2026). It does not include internet travel time from the client.

`deploy/make_sample.py` prints a valid synthetic request built from the bundle.

## Run it yourself

```bash
echo "DOMAIN=your-domain-or-ip.sslip.io" > .env
docker compose up -d --build
```

Tests: `python -m pytest -q --ignore=pipelines/tests/test_bundle_parity.py` (the parity test against the legacy `.pkl` model needs files that are not in git).

## Status

- Done: portable model bundle with parity tests against the legacy model, scoring API in production with HTTPS, CI/CD with tests before every deployment.
- Known limitation: under the crew-v1 NaN policy the calibrated PD under-estimates (see [docs/HALLAZGO_NAN_CALIBRACION.md](docs/HALLAZGO_NAN_CALIBRACION.md)).
- Next: PSI drift monitoring, Postgres on the same VM, and a web view.

The original plan targeted Vercel and Supabase; in November 2026 the deployment moved to a single Oracle Cloud VM. The [migration plan (in Spanish)](docs/Plan_Migracion_v3.md) reflects this.

The research project (EDA, feature engineering, model training, SHAP, agent notebooks, and the original Streamlit dashboard) lives in [JuanCRuizA/credit-risk-early-warning-system](https://github.com/JuanCRuizA/credit-risk-early-warning-system).
