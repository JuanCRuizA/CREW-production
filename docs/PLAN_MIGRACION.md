# CREW — Plan de Migración a Producción (v2, unificado)

**Versión**: v2.1 — 2026-09-26. Incorpora las mejoras de un borrador anterior y las decisiones tomadas en la sesión del 2026-09-26.

**Cambio v2.1**: la producción se construye en un **proyecto y repositorio nuevos** (repo `JuanCRuizA/CREW-production`). El proyecto de investigación `credit-risk-early-warning-system` (repo `bank4growth/credit-risk-early-warning`) y su app de Streamlit **no se tocan**. Del proyecto original solo se copió lo estrictamente necesario (ver la sección 3).

**Objetivo**: ver el flujo completo de una app de riesgo de crédito funcionando en producción real (GitHub → Vercel → Supabase) con una muestra pequeña de datos de Kaggle, montado con las prácticas de un banco (medallion, migraciones versionadas, RLS, audit trail, registro de modelos, CI) para aprender a hacerlo en la vida real.

---

## 1. Decisiones cerradas

| Tema | Decisión |
|---|---|
| Frontend | Next.js (Node.js, TypeScript, App Router) en **Vercel** |
| Backend de inferencia | **Funciones Python de Vercel** (serverless), en el mismo proyecto Vercel que el frontend. **Render queda solo como plan B** si la prueba de concepto (Fase 0.5) demuestra que el scoring no cabe en Vercel |
| Agente de vigilancia (NB05) | Corre en **GitHub Actions** (`workflow_dispatch`), disparado desde la vista avanzada; escribe sus resultados en Supabase. Evita los límites de tiempo de las funciones serverless y deja un log auditable por corrida |
| Base de datos | **Supabase** plan gratuito (500 MB DB, 1 GB Storage) |
| Datos | CSV de Kaggle *Home Credit Default Risk* en `data/raw/` (~2.5 GB, 7 tablas más el diccionario de columnas). Se usa **el mismo modelo relacional de Kaggle** (mismas tablas y llaves) |
| Tamaño de muestra | **Primero se muestrea, después se construyen las capas**: cohorte estratificada por `TARGET` de **~2,000 clientes** (`SK_ID_CURR`) para la primera versión, escalable a ~10,000 cambiando un parámetro. Las 6 tablas hijas se filtran por esos IDs |
| Arquitectura de datos | Medallion completo: `bronze` → `silver` → `gold`, aunque los datos sean estáticos (el objetivo es simular un escenario real) |
| Acceso | **Vista simple pública** (solo lectura, sin agente). **Vista avanzada** con login, **solo por invitación**: el admin asigna correo y usuario; el registro público está desactivado. El agente solo existe en la vista avanzada |
| Repositorio | Proyecto nuevo `CREW-production` (repo `JuanCRuizA/CREW-production`), independiente del proyecto de investigación |
| Streamlit | Sigue funcionando en su propio repo, sin cambios. Sirve como referencia numérica para los tests de paridad |
| Reentrenamiento | **Manual**. El workflow existe y está completo (listo para programarse), pero sin `schedule` activo. Un modelo reentrenado entra como `candidate`; la promoción a `active` es una acción manual del admin tras comparar champion vs challenger (SR 11-7, FINMA 2023/1) |
| Git | Rama `dev` para el trabajo diario; `main` protegida (solo recibe PRs con CI en verde). Se conserva la opción de reescribir historial: se desactiva la protección temporalmente, se reescribe y se vuelve a activar |
| Archivos obsoletos | No aplica: el proyecto original queda intacto como respaldo completo |
| Idioma | Código, nombres de tablas, columnas y comentarios en **inglés** (CLAUDE.md). Documentos de planificación en español |
| Entornos de Supabase | **Un solo proyecto** al inicio (`crew`). Un segundo proyecto `crew-dev` para los previews es opcional en la Fase 6 (el plan gratuito permite 2) |
| Desarrollo local de Supabase | **Sin Docker** al inicio: la CLI se usa para crear migraciones y hacer `db push` al proyecto en la nube. `supabase start` (Docker) queda como mejora opcional |

---

## 2. Arquitectura objetivo

```
                        ┌──────────────────── Vercel (un proyecto, Root Directory = web/) ───────────────────┐
 Navegador ──HTTPS──▶  │  Next.js (Node.js)                         Funciones Python (web/api/*.py)          │
                        │  • /            vista simple (pública)     • POST /api/score                        │
                        │  • /advanced    vista avanzada (login)       xgboost-cpu + numpy (sin shap/sklearn) │
                        │  • server routes: lee Supabase con         • SHAP vía pred_contribs=True            │
                        │    supabase-js; dispara el agente                                                   │
                        └──────────────┬──────────────────────────────────────┬───────────────────────────────┘
                                       │ supabase-js (anon / sesión)          │ service_role (solo servidor)
                                       ▼                                      ▼
                        ┌──────────────────────── Supabase ─────────────────────────────────────────┐
                        │ Postgres: bronze → silver → gold, ml, ops, app  │ Auth (solo invitación)  │
                        │ Schema expuesto por la API: solo `api` (views)  │ Storage: bucket privado │
                        │ RLS en todas las tablas                         │ `models` (bundles)      │
                        └──────────────▲────────────────────────────────────────────────────────────┘
                                       │ service_role (secret de GitHub)
                        ┌──────────────┴──────────── GitHub Actions ───────────────────────────────┐
                        │ ci.yml (PRs)  • retrain.yml (manual)  • surveillance.yml (agente, manual) │
                        │ • monitoring.yml (PSI, manual/opcional programado)                        │
                        └───────────────────────────────────────────────────────────────────────────┘
```

**Quién habla con qué** (fácil de explicar en una entrevista):
- **Next.js lee Supabase directamente** con `supabase-js`: con la anon key solo ve las views públicas del schema `api`; con la sesión del usuario ve lo que su rol permite por RLS.
- **Python solo existe donde hace falta el modelo**: la función `/api/score` y los jobs de GitHub Actions.
- **La `service_role` key nunca llega al navegador**: vive solo en las variables de entorno de servidor de Vercel y en los secretos de GitHub.

---

## 3. Estructura del repositorio (monorepo)

```
CREW-production/
├── web/                          # Proyecto Vercel (Root Directory = web/)
│   ├── app/                      # Next.js App Router (vista simple + /advanced)
│   ├── api/                      # Funciones Python de Vercel
│   │   ├── score.py
│   │   └── _crew/               # núcleo de scoring (el "_" evita que Vercel lo publique como endpoint)
│   │       ├── preprocess.py     # encoders, medianas e imputación desde JSON
│   │       ├── calibrate.py      # calibrador exportado a JSON y aplicado con numpy
│   │       └── model.py          # carga del bundle + predict + pred_contribs
│   ├── requirements.txt          # ligero: xgboost-cpu, numpy, supabase
│   └── package.json
├── pipelines/                    # Jobs Python "pesados" (local o GitHub Actions)
│   ├── sample_cohort.py          # muestreo estratificado desde data/raw/
│   ├── ingest_bronze.py          # aterrizaje append-only con checksum
│   ├── build_silver.py           # tipado, llaves, cuarentena de huérfanos
│   ├── build_gold.py             # features + scoring de la cartera
│   ├── export_model_bundle.py    # pkl → bundle JSON portable
│   ├── train.py                  # reentrenamiento → modelo candidate
│   ├── monitoring.py             # PSI vs baseline del modelo activo
│   ├── surveillance/             # agente NB05 (tools, pipeline, verification)
│   └── requirements.txt          # completo: pandas, xgboost, shap, anthropic, ...
├── supabase/
│   ├── config.toml
│   ├── migrations/               # TODO el esquema, versionado
│   └── seed/                     # datos de demo (usuarios de prueba, etc.)
├── tests/                        # pytest (scoring, paridad, RLS)
├── .github/workflows/            # ci.yml, retrain.yml, surveillance.yml, monitoring.yml
├── reference/                    # copiado del proyecto de investigación, solo lectura
│   ├── streamlit_app.py          # app.py original: lógica de KPIs, ECL, tiers y UI a portar
│   ├── notebooks/                # NB02 (features), NB03 (entrenamiento), NB05 (agente)
│   ├── agent_output_latest.json  # esquema de referencia para ops.agent_runs
│   └── case_studies.json         # esquema de referencia para ops.case_studies
├── models/legacy/                # ignorado por git: los .pkl que carga Streamlit (origen del bundle JSON)
├── docs/                         # este plan (docs/chats/ ignorado)
└── data/                         # ignorado por git
    ├── raw/                      # 7 tablas (más el diccionario de columnas) (~2.5 GB)
    ├── legacy/                   # portfolio_surveillance.db (snapshot de features para la v1)
    └── sample/                   # salida de sample_cohort.py
```

**Qué se copió del proyecto original y para qué** (todo lo demás se quedó allá):

| Origen | Destino | Se usa en |
|---|---|---|
| `docs/PLAN_MIGRACION.md` | `docs/` | Este plan |
| `app.py` | `reference/streamlit_app.py` | Fases 3–4 (portar lógica y UI) |
| NB02, NB03, NB05 | `reference/notebooks/` | Fase 4 (features), Fase 5 (`train.py` y agente) |
| `reports/agent_output_latest.json`, `case_studies.json` | `reference/` | Fase 1 (esquema) y Fase 5 |
| `src/news_service.py` | `pipelines/surveillance/` | Fase 5 |
| 6 `.pkl` + `population_medians.csv` + `model_card.json` | `models/legacy/` | Fase 0.5 (bundle JSON y tests de paridad) |
| `data/raw/*.csv` (sin `application_test` ni `sample_submission`) | `data/raw/` | Fase 1 (muestreo y bronze) |
| `data/portfolio_surveillance.db` | `data/legacy/` | Fase 1 (snapshot de features v1) |

---

## 4. Modelo de datos (Supabase Postgres)

### 4.1 Schemas

| Schema | Contenido | ¿Expuesto por la API REST? |
|---|---|---|
| `bronze` | 7 tablas (más el diccionario de columnas) **tal cual vienen** (columnas como texto), append-only, con `_batch_id`, `_ingested_at`, `_source_file` y `_source_checksum` | No |
| `silver` | Mismas tablas **tipadas y con llaves reales** (PK/FK) | No |
| `gold` | `customer_features`, `scores`, `watch_list` y `risk_migration` | No |
| `ml` | `model_registry` | No |
| `ops` | `ingestion_batches`, `dq_rejections`, `audit_trail`, `agent_runs`, `monitoring_alerts` y `case_studies` | No |
| `app` | `profiles` (rol y cuota del agente) | No |
| `api` | **Views** con exactamente lo que ve el frontend | **Sí, es el único** |

Exponer solo `api` es una defensa en profundidad: aunque una política RLS tenga un error, las tablas base no son alcanzables desde la API pública.

### 4.2 Modelo relacional de Kaggle (en `silver`)

```
applications (sk_id_curr PK, target)                 ← application_train.csv
 ├── bureau (sk_id_bureau PK, sk_id_curr FK)
 │    └── bureau_balance (sk_id_bureau FK, months_balance)
 └── previous_applications (sk_id_prev PK, sk_id_curr FK)
      ├── pos_cash_balance (sk_id_prev FK, sk_id_curr FK, months_balance)
      ├── installments_payments (sk_id_prev FK, sk_id_curr FK)
      └── credit_card_balance (sk_id_prev FK, sk_id_curr FK, months_balance)
```

- Las llaves compuestas o las llaves subrogadas (`bigint generated always as identity`) se definen **después de inspeccionar si hay duplicados** en la muestra; no se asumen.
- **Cuarentena de huérfanos**: los datos de Kaggle tienen filas hijas cuyo `SK_ID_PREV` no existe en `previous_application`. `build_silver.py` las envía a `ops.dq_rejections` con el motivo, en lugar de romper la FK. Es un control de calidad de datos real y un buen tema de entrevista (BCBS 239).
- `application_test.csv` (sin `TARGET`) se puede usar más adelante como "solicitudes nuevas" que se scorean en vivo. No entra en la primera versión.

### 4.3 Tablas `gold`, `ml`, `ops` y `app` (resumen; el SQL definitivo va en las migraciones de la Fase 1)

- `gold.customer_features`: `sk_id_curr` (PK, FK a `silver.applications`), `feature_set_version`, las columnas clave tipadas (los 6 ratios bancarios de NB02: `debt_to_income`, `payment_burden`, `credit_to_goods`, `annuity_to_credit`, `income_per_person`, `income_to_credit`, además de `ext_source_mean`, etc.) y un `features jsonb` con el vector completo que consume el modelo.
- `gold.scores`: `id`, `sk_id_curr` (FK), `pd` (calibrada), `raw_margin`, `risk_tier` (Green/Yellow/Orange/Red), `shap_top jsonb` (drivers principales), `model_version` (FK a `ml.model_registry.version`), `scored_at`, `requested_by uuid null` (FK a `auth.users`; es null en el scoring batch de la cartera).
- `ml.model_registry`: `version` (PK), `status` (`candidate` / `active` / `retired`), `metrics jsonb` (AUC, Gini, KS, Brier, PSI), `storage_path`, `trained_at`, `approved_by`, `approved_at`, `notes`. Garantía en la base de datos de que **solo hay un modelo activo**: `create unique index one_active_model on ml.model_registry (status) where status = 'active';`.
- `ops.audit_trail`: es **append-only**. Se revocan `update` y `delete` a todos los roles, incluido el acceso vía la API. Cada scoring y cada corrida del agente registra quién, cuándo, la versión del modelo, un hash del input y los drivers SHAP. Responde a SR 11-7, al Art. 12 del EU AI Act (registro de eventos; el scoring crediticio es de alto riesgo según el Anexo III) y al Art. 22 del GDPR (explicación de decisiones automatizadas).
- `ops.agent_runs`: `id`, `triggered_by`, `status`, `github_run_id`, `started_at`, `finished_at`, `summary` con columnas indexadas y `payload jsonb`. Reemplaza el `reports/agent_output_latest.json` que hoy se edita a mano.
- `app.profiles`: `id` (PK, FK a `auth.users`), `username`, `role` (`admin` / `analyst`), `agent_runs_limit`, `agent_runs_used`.

---

## 5. Seguridad (todos los huecos detectados en la revisión, cerrados)

1. **RLS habilitado en todas las tablas** de todos los schemas, sin excepción. Una migración de CI verifica que no quede ninguna tabla sin RLS (consulta sobre `pg_tables` con `rowsecurity = false`, que debe devolver cero filas).
2. **Solo el schema `api` está expuesto** por PostgREST. Las tablas base no son alcanzables con la anon key aunque exista un error en una política.
3. **Vista pública**: el rol `anon` solo tiene `select` sobre views específicas de `api` (KPIs agregados y un conjunto fijo de clientes de ejemplo). Sin acceso a `audit_trail`, `agent_runs` ni `profiles`.
4. **Registro solo por invitación**: se desactiva "Allow new users to sign up" en Supabase Auth. El admin invita desde el dashboard de Supabase o desde una server route de admin que usa `auth.admin.inviteUserByEmail`.
5. **Perfil automático**: un trigger en `auth.users` crea la fila en `app.profiles` con rol `analyst` por defecto. El rol `admin` solo se asigna por SQL o desde el panel de admin (con `service_role`).
6. **Nadie modifica su propio rol ni su cuota**: no existe ninguna política de `update` sobre `app.profiles` para `authenticated`. Solo `service_role` escribe ahí.
7. **Sin recursión en las políticas**: la función `app.is_admin()` se declara `security definer` con `set search_path = ''` y se usa en las políticas en lugar de subconsultas a `profiles`.
8. **Cuota atómica solo para el agente** (lo que cuesta dinero). El scoring no lleva cuota, porque su costo es despreciable. La cuota se consume con una función RPC `security definer`:
   ```sql
   update app.profiles
      set agent_runs_used = agent_runs_used + 1
    where id = auth.uid() and agent_runs_used < agent_runs_limit
   returning agent_runs_used;
   ```
   Si no devuelve fila, la corrida no se dispara.
9. **Secretos**: `SUPABASE_SERVICE_ROLE_KEY`, `ANTHROPIC_API_KEY` y el `GITHUB_WORKFLOW_TOKEN` (un PAT *fine-grained* con permiso solo para `actions:write` en este repo) viven únicamente en las variables de servidor de Vercel y en los secretos de GitHub Actions. Nunca llevan prefijo `NEXT_PUBLIC_`.
10. **Rate limit** en `/api/score` para la vista pública (por IP, sencillo), para evitar abuso.
11. **Sandbox `exec()`** de `execute_risk_analysis` (agente NB05): revisión dedicada antes de conectarlo. Como corre en GitHub Actions, en un runner efímero sin acceso a otros secretos que los del job, el radio de impacto es menor, pero se revisa igual.
12. **Tests de RLS** en CI: con la anon key, una consulta a `gold.scores` falla; un `analyst` no puede actualizar su `role`; un `analyst` solo ve sus propios scores en vivo.

---

## 6. Artefactos del modelo (bundle portable)

El modelo actual son varios `.pkl`: booster, calibrador, label encoders, thresholds, medianas y `feature_names`. `pipelines/export_model_bundle.py` los convierte en un **bundle por versión** que la función de Vercel carga sin `pandas`, `scikit-learn` ni `shap`:

```
models/crew-v{n}/
  booster.json          # xgb.Booster.save_model (formato nativo, portable)
  preprocessing.json    # feature_names, label encoders, medianas, modos categóricos
  calibration.json      # parámetros del calibrador (Platt: a, b; isotónico: x/y para np.interp)
  thresholds.json       # cortes de risk tiers
  model_card.json       # ya existe
```

- **SHAP sin la librería `shap`**: `booster.predict(dmatrix, pred_contribs=True)` usa el mismo algoritmo TreeSHAP. **Test de paridad obligatorio**: sobre N clientes, los valores deben coincidir con los de `shap.TreeExplainer` de NB04 (tolerancia 1e-6).
- **Test de paridad del scoring**: la PD del bundle JSON debe coincidir con la PD del pipeline en `.pkl` que usa Streamlit.
- **Nota de interpretabilidad**: SHAP explica el **margen crudo (log-odds)**, no la PD calibrada. Se documenta en la UI y en el model card.
- **Dónde vive el bundle**:
  - Fase 0.5: se empaqueta dentro de `web/api/_crew/` para medir tamaño y latencia.
  - Fase 2: pasa al bucket privado `models` de Supabase Storage; la función descarga el bundle del modelo con `status = 'active'` en el cold start y lo cachea a nivel de módulo.

---

## 7. Fases
 
**Alcance v1**: Fases 0 a 3 más el README con rollback y el tag (Fase 6). Las Fases 4 y 5 se deciden después.

**Regla de oro**: nunca se trabaja directo sobre `main`. Todo pasa por `dev`, y `main` solo recibe PRs con CI en verde. Antes de cada fase o paso destructivo se crea un tag `pre-fase-X-YYYYMMDD`. Si algo sale mal, se vuelve al tag; no se arregla hacia adelante bajo presión.

### Fase 0 — Proyecto nuevo (1–2 h)
- [x] Clonar `JuanCRuizA/CREW-production` en una carpeta local, el repo trae solo `LICENSE`, Apache 2.0
- [x] Esqueleto del monorepo, copia de lo estrictamente necesario (sección 3), `.gitignore`, `.env.example`, `README.md` y `CLAUDE.md`
- [x] Definir la identidad de git del repo (`user.name` / `user.email` locales) y la cuenta de GitHub con la que se hace push
- [x] Primeros commits en `main`
- [ ] Separar dependencias: `web/requirements.txt` (ligero) y `pipelines/requirements.txt` (completo)
- [ ] Hacer push y Crear la rama `dev`
- [ ] **Proteger `main`** en GitHub (Settings → Rules → Rulesets): exigir PR; el check de CI se agrega en la Fase 2, cuando exista ci.yml. Nota: en GitHub gratis requiere que el repo sea **público**
- **Hecho cuando**: el repo tiene la estructura en `main` y `dev`, `git status` está limpio y `main` está protegida.

### Fase 0.5 — Prueba de concepto de Vercel Python (1–2 h) ⚠ riesgo principal
- [ ] `export_model_bundle.py` genera el bundle JSON y se valida la paridad contra los `.pkl`
- [ ] `web/api/score.py` mínimo (sin base de datos): recibe features y devuelve PD, tier y top drivers SHAP
- [ ] Deploy a Vercel (preview) y medición: tamaño del bundle de la función, cold start y latencia en caliente
- **Hecho cuando**: `curl -X POST https://<preview>.vercel.app/api/score` devuelve una PD idéntica a la de Streamlit para el mismo cliente.
- **Si falla** (tamaño o tiempo): el scoring se mueve a FastAPI en Render (plan B) y el resto del plan no cambia.

### Fase 1 — Supabase: esquema, seguridad y datos (6–10 h)
- [ ] Crear el proyecto `crew` en supabase.com, en la región más cercana a la de las funciones de Vercel
- [ ] Instalar la CLI (`scoop install supabase` o `npx supabase`), luego `supabase init` y `supabase link --project-ref <ref>`
- [ ] Migraciones (una por tema, en orden): schemas y roles → bronze → silver (PK/FK) → gold/ml/ops/app → RLS, políticas y funciones → views `api` → grants
- [ ] `supabase db push` al proyecto en la nube
- [ ] En el dashboard: exponer solo el schema `api`, desactivar el registro público y crear el bucket privado `models`
- [ ] Pipeline de datos (local, con `data/raw/`):
  1. `sample_cohort.py`: ~2,000 `SK_ID_CURR` estratificados por `TARGET`, con semilla fija, filtrando las 6 tablas hijas → `data/sample/*.csv`
  2. `ingest_bronze.py`: registra el lote en `ops.ingestion_batches` con el SHA-256 de cada archivo. **Si el checksum ya existe, no vuelve a aterrizar nada**. Append-only
  3. `build_silver.py`: tipado, PK/FK y cuarentena de huérfanos en `ops.dq_rejections`
  4. `build_gold.py`: features de NB02 y scoring de la cartera con el modelo activo → `gold.customer_features` y `gold.scores`. **Simplificación v1**: las features se toman de `engineered_features` del SQLite (`data/legacy/portfolio_surveillance.db`) para los mismos `SK_ID_CURR` (documentado como "snapshot del feature store"). Portar las agregaciones de NB02 a `build_gold.py` queda como mejora en la Fase 4
- [ ] Registrar `crew-v1` en `ml.model_registry` como `active` y subir su bundle a Storage
- **Hecho cuando**: los conteos cuadran bronze = silver + rechazos, la tasa de default de la muestra es ~8%, y los tests de RLS pasan.

### Fase 2 — Scoring conectado a Supabase (4–6 h)
- [ ] `/api/score` carga el bundle `active` desde Storage (con caché en el módulo), escribe en `gold.scores` y en `ops.audit_trail`
- [ ] Rate limit para las llamadas anónimas
- [ ] `tests/`: paridad del scoring, paridad SHAP, aditividad SHAP (suma de contribuciones + bias = margen) y contrato del endpoint
- [ ] `ci.yml` en PRs a `dev` y `main`: `pytest` más `npm run build` y `lint` de `web/`, y agregar el check de CI.
- **Hecho cuando**: un score hecho desde curl aparece en `gold.scores` y en `ops.audit_trail`, y CI está en verde.

### Fase 3 — Next.js: vista simple y login (8–12 h)
- [ ] `create-next-app` en `web/` (TypeScript, App Router) e importar el repo en Vercel con Root Directory `web/`
- [ ] Variables de entorno en Vercel por entorno (Production / Preview / Development)
- [ ] Vista simple (`/`): KPIs de la cartera desde `api.*` y un scoring de ejemplo con drivers SHAP
- [ ] Login (`/login`) con Supabase Auth (email y contraseña, solo invitación) y la ruta `/advanced` protegida por middleware
- [ ] Invitar al primer usuario admin y a un analyst de prueba
- **Hecho cuando**: la URL de producción de Vercel muestra datos que vienen de Supabase de punta a punta, y el login funciona con un usuario invitado. **Aquí se cumple el objetivo de "ver el flujo completo funcionando".**

### Fase 4 — Vista avanzada con paridad con Streamlit (16–24 h)
- [ ] Pestañas restantes (desempeño del modelo, explicabilidad, watch list, compliance) y la calculadora de scoring
- [ ] Panel de admin: invitar usuarios, asignar rol y cuota, promover un modelo `candidate` a `active`
- [ ] Portar las agregaciones de NB02 a `build_gold.py` (reemplaza el snapshot del SQLite)
- **Hecho cuando**: hay paridad de funciones con Streamlit (excepto los PNG estáticos que se descartan a propósito).

### Fase 5 — Agente, reentrenamiento y monitoreo en GitHub Actions (16–24 h)
- [ ] Extraer de NB05 los `TOOL_SCHEMAS` (se conservan como diccionarios explícitos, por auditabilidad SR 11-7), el pipeline de las fases A–D y el `VerificationSubagent` (fase E, que sigue siendo un módulo separado) a `pipelines/surveillance/`
- [ ] Revisión de seguridad del sandbox `exec()`
- [ ] `surveillance.yml` (`workflow_dispatch` con inputs): el botón de la vista avanzada consume la cuota (RPC), crea la fila en `ops.agent_runs`, dispara el workflow vía la API de GitHub y la UI consulta el estado
- [ ] `retrain.yml`: **solo `workflow_dispatch`**, con el bloque `schedule` escrito y comentado, listo para activarse. Registra el modelo como `candidate` con sus métricas y su comparación contra el `active`
- [ ] `monitoring.yml`: calcula el PSI de la cartera contra el baseline del modelo activo y escribe en `ops.monitoring_alerts`
- **Hecho cuando**: una corrida real del agente desde la UI produce una fila en `ops.agent_runs` con la fase E en verde.

### Fase 6 — Cutover y endurecimiento (3–5 h)
- [ ] Opcional: segundo proyecto Supabase `crew-dev` para que los previews no toquen producción
- [ ] QA lado a lado entre Streamlit y la app nueva (KPIs, scores y SHAP para un conjunto fijo de clientes)
- [ ] README: URLs nuevas, arquitectura y **cómo hacer rollback**
- [ ] Tag `v1.0-produccion`; en el README del proyecto de investigación se agrega el link a la app nueva (Streamlit queda como referencia archivada)

---

## 8. Rollback (plan gratuito)

| Capa | Cómo se vuelve atrás |
|---|---|
| Código | `git checkout <tag>` o un PR de revert. `main` protegida: nada llega a producción sin PR |
| Deploy | Vercel → Deployments → **Promote to Production** sobre un deployment anterior (instantáneo) |
| Base de datos | Supabase **no tiene migraciones "down"** y el plan gratuito no tiene point-in-time recovery. Práctica: `supabase db dump` antes de cada migración riesgosa y, si algo sale mal, una **migración nueva que corrige** (forward-fix). Como los datos son reproducibles desde `data/raw/` y los scripts, el último recurso es `supabase db reset` en la nube y volver a correr el pipeline |
| Modelo | Cambiar el modelo `active` en `ml.model_registry` por el anterior (una línea de SQL o un botón del panel de admin) |

---

## 9. Consideraciones del plan gratuito

- **Supabase pausa el proyecto tras 7 días sin actividad**: antes de una demo, entrar al dashboard o hacer una consulta. Opcional: un workflow de GitHub Actions que haga un ping semanal.
- **Espacio**: con 2,000 clientes, el total de las 3 capas debería quedar muy por debajo de 100 MB; con 10,000, aproximadamente 300–400 MB. Se mide después de la Fase 1 antes de escalar.
- **Vercel Hobby**: límites de tamaño y duración de las funciones (la Fase 0.5 lo valida). El uso comercial requiere el plan Pro; un portafolio personal está dentro de Hobby.

---

## 10. Verificación general

- Cada capa se prueba sola antes de conectar la siguiente (primero curl contra `/api/score`, después la UI). Nunca se depuran dos capas a la vez.
- Paridad numérica contra Streamlit en cada fase que toca el scoring.
- CI en verde como condición para mergear a `main`.

## Estimación

~55–85 horas de trabajo efectivo. El **flujo completo funcionando** (Fases 0 → 3) son ~20–30 h. Camino crítico: 0 → 0.5 → 1 → 2 → 3 → 4 → 6; la Fase 5 puede ir en paralelo a la 4, pero se recomienda hacerlas en secuencia.
