# CREW - Credit Risk Early Warning system (production)

Production deployment of CREW, the XGBoost credit-risk model built on the Kaggle *Home Credit Default Risk* dataset.

- **Frontend**: Next.js on Vercel. A public read-only view, plus an invite-only advanced view.
- **Scoring**: Vercel Python functions. XGBoost with TreeSHAP explanations from a portable JSON model bundle.
- **Data**: Supabase Postgres, medallion architecture (`bronze` → `silver` → `gold`) over the original Kaggle relational model.
- **Jobs**: GitHub Actions for the surveillance agent, manual retraining with a champion/challenger registry, and PSI monitoring (planned).

The research project (EDA, feature engineering, model training, SHAP, agent notebooks, and the original Streamlit dashboard) lives in [bank4growth/credit-risk-early-warning](https://github.com/bank4growth/credit-risk-early-warning).

> Status: in construction. See the [migration plan (in Spanish)](docs/PLAN_MIGRACION.md) for phases and progress.
