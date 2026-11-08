# CREW simple deploy: Supabase + Render + Flask runbook

This runbook takes CREW from the `simple-deploy` branch to a live Flask service on Render that reads and writes a Supabase Postgres database. Every step says **what** it does, **why**, and **what breaks if you skip it**.

| Item | Value |
|---|---|
| Supabase project | `crew-production`, ref `ssgsljmzbxdxzwwzpimr`, region `us-west-2` (Oregon), Free plan, Nano compute |
| Render service | `crew-simple` (Free plan, region `oregon`), defined in [`render.yaml`](../render.yaml) |
| Deploy branch | `simple-deploy` on `JuanCRuizA/CREW-production` |
| Rollback tag | `pre-fase-simple-20261108` |

Two database identities, never mixed:

| Variable | Role | Where it lives | Used by |
|---|---|---|---|
| `DATABASE_URL_ADMIN` | `postgres` (owner) | your local `.env` only | `pipelines/load_sample.py` |
| `DATABASE_URL` | `crew_app` (least privilege) | Render env var (and local `.env` for smoke tests) | the Flask service |

---

## A. Supabase

### A1. Get (or reset) the database password
- **What:** Supabase dashboard -> Project Settings -> Database -> *Reset database password* if you did not save it at project creation. Store it in your password manager.
- **Why:** the CLI and the admin connection string need it. Supabase never shows it again.
- **If skipped:** `supabase link` and the loader cannot authenticate.

### A2. Install the Supabase CLI and log in
- **What (Windows, pick one):**
  - Scoop: `scoop bucket add supabase https://github.com/supabase/scoop-bucket.git` then `scoop install supabase`
  - No install: prefix every command with `npx supabase` (needs Node.js).
  - Then `supabase login` (opens the browser and stores an access token on your machine, not in the repo).
- **Why:** the CLI applies the versioned SQL in `supabase/migrations/` and records which migrations ran. CLAUDE.md rule: never edit the schema by hand in the dashboard.
- **If skipped:** you can still use the SQL Editor alternative in A4, but you lose the migration history (`supabase_migrations.schema_migrations`).

### A3. Link the repo to the project
- **What:** from the repo root: `supabase link --project-ref ssgsljmzbxdxzwwzpimr` (it asks for the password from A1).
- **Why:** tells the CLI which remote database `db push` targets. It writes local state to `supabase/.temp/` (git-ignored).
- **If skipped:** `supabase db push` fails with "Cannot find project ref".

### A4. Apply the schema migration
- **What:** `supabase db push --dry-run` (lists the migrations it would run: expect `0001_crew_schema.sql`), then `supabase db push`.
  - **Alternative:** Dashboard -> SQL Editor -> paste the full content of `supabase/migrations/0001_crew_schema.sql` -> Run. Run it once only.
- **Why:** creates schema `crew` with `clients`, `client_features`, `scores`, `score_audit_log`, enables RLS on all four with policies only for `crew_app` (none for `anon` or `authenticated`), and creates the `crew_app` role with minimal grants.
- **If skipped:** the loader and the service fail with `schema "crew" does not exist`.
- **Note:** if `supabase` asks to initialise a `supabase/config.toml`, answer yes only for local files; it does not change the remote project.

### A5. Set the `crew_app` password
- **What:** SQL Editor, one line (generate a strong password, no reserved URL characters makes life easier):
  ```sql
  alter role crew_app with password 'paste-a-long-random-password-here';
  ```
  Do not save this query in the SQL Editor snippets; store the password in your password manager.
- **Why:** the migration creates the role without a password so no secret ever enters git.
- **If skipped:** Render cannot connect (`password authentication failed for user "crew_app..."`).

### A6. Keep the `crew` schema out of the Data API
- **What:** Project Settings -> Data API (API settings) -> *Exposed schemas*: make sure `crew` is **not** listed (leave the defaults, `public` and `graphql_public`).
- **Why:** this path talks to Postgres directly from Flask; nothing should be reachable through PostgREST with the public anon key. RLS has no policies for `anon` or `authenticated`, so it already denies them access; this is a second layer (defence in depth).
- **If skipped:** no data leak while RLS stays on, but one mistaken policy later would expose client data through the public REST endpoint.

### A7. Build the two session pooler connection strings
- **What:** click **Connect** (top bar) -> *Connection string* -> **Session pooler**. Copy the URI. It looks like:
  ```
  postgresql://postgres.ssgsljmzbxdxzwwzpimr:[YOUR-PASSWORD]@aws-0-us-west-2.pooler.supabase.com:5432/postgres
  ```
  Use the exact host the dialog shows (the `aws-N` prefix varies). Then build:
  - `DATABASE_URL_ADMIN` = that URI with the A1 password, plus `?sslmode=require`.
  - `DATABASE_URL` = same host and port, user changed to `crew_app.ssgsljmzbxdxzwwzpimr`, password from A5, plus `?sslmode=require`.
  - Percent-encode reserved characters in passwords (`@` -> `%40`, `#` -> `%23`, `&` -> `%26`, `?` -> `%3F`, `/` -> `%2F`, space -> `%20`).
  - Put both in your local `.env` (git-ignored). Only `DATABASE_URL` will go to Render.
- **Why session pooler:** the *direct* connection (`db.ssgsljmzbxdxzwwzpimr.supabase.co`) is IPv6-only without the paid IPv4 add-on, and Render cannot reach IPv6. The session pooler (Supavisor, port 5432) is IPv4 and keeps a real session, so psycopg prepared statements work. Custom roles connect through the shared pooler as `role.projectref`.
- **If skipped / wrong:** see Troubleshooting (IPv6, tenant not found, prepared statements).

### A8. Load the sample
- **What:** from the repo root with your Python environment active:
  ```
  python pipelines/load_sample.py --dry-run
  python pipelines/load_sample.py
  ```
  The dry run reads the local legacy snapshot and reports the sample (about 2,000 clients, stratified by TARGET, seed 42) without writing. The real run inserts into `crew.clients` and `crew.client_features` through `DATABASE_URL_ADMIN`.
- **Why:** the service scores clients whose features are stored in the database. The loader refuses records with missing features (no silent defaults).
- **If skipped:** the UI client list is empty and `POST /api/clients/<id>/score` returns 404.

### A9. Verify the load
- **What:** SQL Editor:
  ```sql
  select count(*) as clients, sum(target) as defaulters from crew.clients;
  select count(*) as feature_rows from crew.client_features;
  select count(*) from crew.scores;          -- 0 before the first scoring
  select rolname, rolcanlogin, rolbypassrls from pg_roles where rolname = 'crew_app';
  select relname, relrowsecurity from pg_class
   where relnamespace = 'crew'::regnamespace and relkind = 'r';
  ```
- **Expected:** `clients` = `feature_rows` (about 2,000); the defaulter share close to the population rate (about 8%); `crew_app` can log in and does **not** bypass RLS; every table has `relrowsecurity = true`.
- **If skipped:** a partial load surfaces later as confusing 404s in production.

---

## B. Render

### B1. Push the branch (you do it)
- **What:** `git push -u origin simple-deploy` (after reviewing the integrated branch). Narrated step: this publishes the branch to GitHub; Render can only build what exists on GitHub.
- **Why:** the Blueprint points at `branch: simple-deploy`.
- **If skipped:** Render reports that the branch does not exist.

### B2. Create the Blueprint
- **What:** Render dashboard -> **New** -> **Blueprint** -> connect your GitHub account (grant access to `JuanCRuizA/CREW-production` only, not all repositories) -> select the repo -> branch `simple-deploy` -> Render reads `render.yaml` and shows one service, `crew-simple`.
- **Why:** the Blueprint is infrastructure as code: plan, region, build and start commands, health check and env var names are versioned in git instead of clicked by hand.
- **If skipped (manual Web Service instead):** works, but copy every setting from `render.yaml` by hand and expect drift.

### B3. Fill the secret
- **What:** Render asks for `DATABASE_URL` (declared `sync: false`). Paste the **crew_app** URI from A7. `FLASK_SECRET_KEY` is generated by Render; `PYTHON_VERSION` comes from the file.
- **Why:** `sync: false` values are requested only at Blueprint creation and are never stored in git. Later changes go to Service -> Environment.
- **Never:** paste `DATABASE_URL_ADMIN`, the `postgres` password, or the Supabase `service_role` key into Render.
- **If skipped:** the service starts and fails loudly on the first DB access (by design, there is no default).

### B4. Deploy and read the logs
- **What:** click **Apply**. Watch Service -> Logs: `pip install -r service/requirements.txt`, then `gunicorn ... "app:create_app()"`, then the health check on `/health`.
- **Why:** the first build downloads xgboost-cpu and numpy (a few minutes). The deploy is marked live only after `/health` answers 200.
- **If skipped:** n/a; if it fails, see Troubleshooting.

### B5. Smoke test
Replace `<host>` with the `onrender.com` URL shown in the dashboard.
```
curl -i https://<host>/health
curl -s https://<host>/api/score
curl -s -X POST https://<host>/api/clients/<sk_id_curr>/score
```
- **Expected:** `/health` 200; `GET /api/score` returns `model_version` `crew-v1` and a `bundle_sha256` identical to the local one (`python -c "import sys; sys.path.insert(0, 'service'); from crew.bundle import load_bundle; print(load_bundle().sha256)"`); the POST returns `pd_raw`, `pd_calibrated`, flags and drivers, and adds one row to `crew.scores` and one to `crew.score_audit_log` (check with A9 queries). Take a real `sk_id_curr` from `select sk_id_curr from crew.clients limit 1;`.
- Then open `https://<host>/` in the browser for the UI.

### B6. Know the Free plan behaviour
- The service sleeps after 15 minutes without traffic; the next request wakes it in about one minute (Render shows a loading page). Do not ping it to keep it awake: it burns the monthly free instance hours.
- `autoDeploy: true`: every push to `simple-deploy` rebuilds and redeploys.
- Supabase Free pauses a project after about a week of inactivity; resume it from the dashboard.

---

## C. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection to server ... Network is unreachable` or an IPv6 address in the error | Using the direct host `db.<ref>.supabase.co` (IPv6-only) | Use the session pooler URI from A7 |
| `Tenant or user not found` | Pooler user without the project suffix | User must be `crew_app.ssgsljmzbxdxzwwzpimr` (or `postgres.<ref>`) |
| `prepared statement "_pg3_0" already exists` / does not exist | Port 6543 (transaction pooler) | Use port 5432 (session pooler) |
| `password authentication failed` | Wrong password or unencoded reserved character | Re-run A5, percent-encode the password |
| `permission denied for schema crew` / `for table ...` | Connected as a role without grants, or migration not applied | Check A4 ran and the user is `crew_app` |
| `new row violates row-level security policy` | Migration expects RLS plus explicit grants; a policy or grant is missing | Fix it in a new migration file, never in the dashboard |
| 500 with `"code": "bundle_error"` | Bundle files missing or changed (CRLF, partial copy) | `pytest` locally; check `.gitattributes` `-text` covers the bundle path; redeploy |
| Build fails or times out installing xgboost | Version without a wheel for Python 3.12 / pip resolving from source | Keep `xgboost-cpu==3.1.2` and `PYTHON_VERSION=3.12.x`; clear build cache and redeploy |
| `ModuleNotFoundError: crew` / `app` | Start command without `--chdir service` | Use the start command in `render.yaml` |
| First request takes about a minute | Free plan cold start | Expected (B6) |

---

## D. Security checklist

- [ ] No real secret in git: `.env` is ignored; only `.env.example` with placeholders is committed. Before each push, read `git diff origin/simple-deploy..HEAD` and look for anything that resembles a password or a full connection string.
- [ ] Render holds only `DATABASE_URL` (crew_app) and the generated `FLASK_SECRET_KEY`.
- [ ] `crew_app`: `LOGIN`, no `BYPASSRLS`, no `CREATEROLE`, only the grants in the migration (select on all four tables, insert on scores and score_audit_log, no update or delete).
- [ ] RLS enabled on every `crew` table; `crew` not exposed in the Data API.
- [ ] The Supabase `service_role` key is not used anywhere on this path.
- [ ] Passwords rotated if they ever appeared in a terminal history, screenshot or chat (`alter role crew_app with password ...`, then update Render).
- [ ] Features are never logged by the service (data minimisation).

Regulatory link, one line each:
- **GDPR Art. 5(1)(c) and 32:** data minimisation and least-privilege access to client data.
- **EU AI Act Art. 12 (credit scoring is high-risk, Annex III):** `score_audit_log` is the automatic, append-only record of every scoring event.
- **SR 11-7:** the model version and `bundle_sha256` stored with every score tie each decision to an identified, validated model.

---

## E. Rollback

- **Service:** Render -> `crew-simple` -> Events / Deploys -> pick the last good deploy -> **Rollback**. Instant, no git change. To stop serving altogether: Settings -> **Suspend**.
- **Code:** the branch starts from tag `pre-fase-simple-20261108` (on `dev`, commit `63f818a`); `dev` and `main` are untouched by this path.
- **Database:** the Free plan has no point-in-time recovery. To undo the schema: SQL Editor `drop schema crew cascade; drop role crew_app;` then re-run A4 to A9. Write any schema fix as a new migration file instead of editing `0001`.
