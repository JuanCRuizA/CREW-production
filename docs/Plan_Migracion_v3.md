# CREW-production: plan de migración a producción (v3)

**Versión**: v3.0, 15 de noviembre de 2026. Reemplaza la v2.1 (26 de septiembre de 2026).

**Cambio v3**: el despliegue pasa de Vercel + Supabase a **una sola VM en Oracle Cloud Infrastructure (OCI)**, con Docker, Caddy y Postgres en la misma máquina. GitHub Actions se mantiene para CI/CD y jobs. Los cuatro proyectos del portafolio (CREW, SAFE, PRISM y CLASS) se desplegarán en esta misma VM.

**Por qué**: una VM propia con contenedores es más simple de mantener que dos plataformas serverless, se parece más a cómo despliega un banco, y el plan Always Free de OCI cubre los cuatro proyectos. La prueba de concepto de Vercel (Fase 0.5) no llegó a hacerse: la API se desplegó directamente en Oracle el 15 de noviembre de 2026.

**Objetivo** (sin cambios): ver el flujo completo de una app de riesgo de crédito funcionando en producción real, con una muestra pequeña de datos de Kaggle, montado con las prácticas de un banco (medallion, migraciones versionadas, control de acceso, audit trail, registro de modelos, CI/CD) para aprender a hacerlo en la vida real.

El proyecto de investigación `credit-risk-early-warning-system` y su app de Streamlit siguen sin tocarse y sirven como referencia numérica para los tests de paridad.

---

## 1. Decisiones cerradas

| Tema | Decisión |
|---|---|
| Infraestructura | Una VM en OCI, región Bogotá: `VM.Standard.A1.Flex` (Ampere ARM), 2 OCPU, 12 GB de RAM, disco de 100 GB, Ubuntu 24.04. Ampliable a 4 OCPU / 24 GB sin recrearla (límite Always Free) |
| Acceso a la VM | SSH solo con llaves: una por equipo personal y una dedicada a GitHub Actions (revocable por separado) |
| Red | Puertos abiertos solo 22, 80 y 443, en dos capas: Security List de OCI e iptables de la VM |
| Contenedores | Docker + Docker Compose. Cada proyecto vive en su carpeta (`~/CREW-production`, `~/SAFE-production`, etc.) con su propio `docker-compose.yml` |
| Entrada (edge) | Caddy como reverse proxy, con HTTPS automático de Let's Encrypt. Hoy usa `129-38-95-215.sslip.io`; un dominio propio es opcional |
| Backend de inferencia | API de scoring en Python (xgboost-cpu + numpy, SHAP vía `pred_contribs=True`), en un contenedor que solo es visible en la red interna de Docker. Reutiliza `web/api/score.py` sobre `ThreadingHTTPServer` |
| Base de datos | Postgres en un contenedor de la misma VM (reemplaza a Supabase). No se expone a internet: solo lo ven la API y los jobs |
| Modelo | El bundle JSON vive en el repo (`models/crew-v{n}/`) y se registra en `ml.model_registry`. Ya no hace falta un bucket de Storage |
| Frontend | Next.js en un contenedor detrás de Caddy, con vista simple pública y vista avanzada con login |
| Jobs (agente, reentrenamiento, PSI) | Corren en la VM (`docker compose run`), disparados de forma manual o desde GitHub Actions por SSH. Dejan registro en Postgres |
| CI/CD | GitHub Actions: cada push a `dev` corre los tests en un runner ARM y, solo si pasan, despliega por SSH y hace un smoke test de la API en vivo |
| Git | `dev` para el trabajo diario; `main` protegida, solo recibe PRs con el check de CI en verde |
| Reentrenamiento | Manual. Un modelo nuevo entra como `candidate`; la promoción a `active` es una acción manual tras comparar champion vs challenger (SR 11-7, FINMA 2023/1) |
| Idioma | Código, tablas, columnas y comentarios en inglés. Documentos de planificación en español |

## 2. Decisiones abiertas (se cierran en la fase indicada)

| Tema | Opciones | Fase |
|---|---|---|
| Framework de la API | Mantener el handler actual o migrar a FastAPI (documentación Swagger, validación con Pydantic) | 2 |
| Autenticación de la vista avanzada | Auth.js (NextAuth) con usuarios en Postgres y registro solo por invitación, o autenticación básica de Caddy como paso intermedio | 3 |
| Rate limit | En la propia API (por IP) o con el módulo de rate limit de Caddy | 2 |
| Migraciones de Postgres | Archivos SQL numerados aplicados con un script propio, o una herramienta como dbmate | 1 |

---

## 3. Arquitectura objetivo

```
                         VM en OCI (Ubuntu 24.04, ARM)
 Navegador --HTTPS-->  Caddy :443  (edge compartido por los 4 proyectos)
                         |-- /            --> web (Next.js)       vista simple y /advanced
                         |-- /api/score   --> api (Python)        xgboost-cpu + numpy, SHAP
                                                 |
                                                 v
                                          postgres (solo red interna)
                                          bronze -> silver -> gold, ml, ops, app
                                                 ^
                         jobs (docker compose run): pipelines, agente, reentrenamiento, PSI

 GitHub Actions: tests en ARM -> deploy por SSH -> smoke test
```

**Quién habla con qué** (fácil de explicar en una entrevista):
- **Solo Caddy recibe tráfico de internet.** La API, el frontend y Postgres están en la red interna de Docker.
- **Postgres nunca se expone.** El navegador no habla con la base de datos: todo pasa por el backend, que aplica los permisos.
- **Python solo existe donde hace falta el modelo**: la API de scoring y los jobs.
- **Los secretos** (contraseña de Postgres, `ANTHROPIC_API_KEY`) viven en el `.env` de la VM, que no entra a git, y en los secretos de GitHub Actions.

---

## 4. Estructura del repositorio

```
CREW-production/
├── web/
│   ├── api/                   # API de scoring (score.py + _crew/)
│   └── app/                   # Next.js (Fase 3)
├── pipelines/                 # jobs: muestreo, bronze, silver, gold, export, train, monitoring, agente
├── db/migrations/             # SQL versionado de Postgres (reemplaza supabase/)
├── models/crew-v1/            # bundle JSON del modelo activo
├── deploy/                    # serve.py, Caddyfile, make_sample.py
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/         # deploy.yml (hecho); retrain, surveillance y monitoring en la Fase 5
├── reference/                 # copiado del proyecto de investigación, solo lectura
├── docs/
└── data/                      # ignorado por git
```

---

## 5. Modelo de datos (Postgres en la VM)

Se mantiene el diseño de la v2.1: schemas `bronze`, `silver`, `gold`, `ml`, `ops` y `app`, el modelo relacional de Kaggle en `silver`, la cuarentena de huérfanos en `ops.dq_rejections` (BCBS 239), `ml.model_registry` con un único modelo `active` garantizado por índice, y `ops.audit_trail` append-only (SR 11-7, Art. 12 del EU AI Act, Art. 22 del GDPR).

Cambios frente a la v2.1:
- **Sin schema `api` ni PostgREST**: el frontend no consulta la base de datos directamente; el backend expone solo los endpoints necesarios.
- **Usuarios en `app.users`** (o las tablas de Auth.js), en lugar de `auth.users` de Supabase.
- **Roles de Postgres separados**: uno de solo lectura para la API pública, uno de escritura para los jobs y uno de migraciones. Nadie usa el superusuario en la operación diaria.
- El bundle se registra en `ml.model_registry` con su ruta en el repo y su SHA-256.

---

## 6. Seguridad

1. **Superficie mínima**: solo 22, 80 y 443 abiertos, en Security List e iptables (defensa en profundidad).
2. **SSH solo con llaves**, sin contraseña. Una llave por equipo y una dedicada a CI/CD, sin passphrase y revocable borrando su línea en `authorized_keys`.
3. **Huella del servidor fijada** en GitHub Actions (`DEPLOY_KNOWN_HOSTS`) para evitar ataques man-in-the-middle.
4. **HTTPS obligatorio** con certificados renovados automáticamente por Caddy.
5. **Postgres sin exposición pública** y con roles de mínimo privilegio.
6. **Registro solo por invitación** en la vista avanzada; nadie modifica su propio rol ni su cuota.
7. **Cuota atómica para el agente** (lo que cuesta dinero), con un `update ... where agent_runs_used < agent_runs_limit returning ...`.
8. **Rate limit** en `/api/score` para la vista pública.
9. **Sandbox `exec()`** del agente (NB05): revisión dedicada antes de conectarlo.
10. **Las features nunca se registran en logs** (minimización de datos).
11. **Pendiente**: actualizaciones de seguridad automáticas (`unattended-upgrades`) y `fail2ban` en la VM.

---

## 7. Fases

**Regla de oro** (sin cambios): nunca se trabaja directo sobre `main`. Antes de cada fase o paso destructivo se crea un tag `pre-fase-X-YYYYMMDD`. Si algo sale mal, se vuelve al tag; no se arregla hacia adelante bajo presión.

### Fase 0: proyecto nuevo (hecha)
- [x] Repo `JuanCRuizA/CREW-production`, esqueleto del monorepo, `.gitignore`, `.env.example`, `README.md`, `CLAUDE.md`
- [x] Dependencias separadas: `web/requirements.txt` (ligero) y `pipelines/requirements.txt` (completo)
- [x] Rama `dev` y `main` protegida

### Fase 0.5: API de scoring en Oracle Cloud (hecha el 15 de noviembre de 2026, salvo lo marcado)
- [x] Bundle JSON con paridad validada contra los `.pkl`
- [x] `web/api/score.py`: `pd_raw`, `pd_calibrated`, banderas de umbral y top drivers SHAP (sin tier: ver `docs/HALLAZGO_NAN_CALIBRACION.md`)
- [x] VM en OCI, Docker, puertos 80/443 en dos capas
- [x] API en contenedor detrás de Caddy con HTTPS: `https://129-38-95-215.sslip.io/api/score`
- [x] Paridad en producción: el mismo registro de prueba da el mismo `pd_raw` (0.3361572027206421) en local y en la VM
- [x] CI/CD: 85 tests en ARM, deploy por SSH y smoke test (`.github/workflows/deploy.yml`)
- [x] README con la arquitectura real
- [x] Medir latencia en caliente: p50 42 ms, p95 43 ms (23 nov 2026, medida desde la VM)
- [ ] PR de `dev` a `main` con el check de CI exigido en la protección de `main`, y tag `v0.5-api-oracle`

### Fase 1: Postgres, esquema y datos (6 a 10 h)
- [ ] Servicio `postgres` en `docker-compose.yml`, con volumen persistente, contraseña en `.env` y sin puertos publicados
- [ ] Migraciones (una por tema, en orden): schemas y roles, bronze, silver (PK/FK), gold/ml/ops/app, permisos
- [ ] Pipeline de datos, igual que en la v2.1: `sample_cohort.py` (~2,000 clientes estratificados por `TARGET`), `ingest_bronze.py` (append-only con SHA-256), `build_silver.py` (tipado y cuarentena de huérfanos), `build_gold.py` (features del snapshot SQLite y scoring de la cartera)
- [ ] Registrar `crew-v1` en `ml.model_registry` como `active`
- [ ] Backup con `pg_dump` antes de cada migración riesgosa
- **Hecho cuando**: los conteos cuadran (bronze = silver + rechazos), la tasa de default de la muestra es ~8% y los roles de solo lectura no pueden escribir.

### Fase 2: scoring conectado a Postgres (4 a 6 h)
- [ ] `/api/score` escribe en `gold.scores` y en `ops.audit_trail`
- [ ] Rate limit para las llamadas anónimas
- [ ] Tests: paridad SHAP y aditividad (suma de contribuciones + bias = margen), además de los actuales
- [ ] Decidir handler actual vs FastAPI
- **Hecho cuando**: un score hecho con curl aparece en `gold.scores` y en `ops.audit_trail`, y CI está en verde.

### Fase 3: Next.js, vista simple y login (8 a 12 h)
- [ ] App Next.js en `web/app/`, con su contenedor y su ruta en Caddy
- [ ] Vista simple (`/`): KPIs de la cartera y un scoring de ejemplo con drivers SHAP
- [ ] Login solo por invitación y ruta `/advanced` protegida
- **Hecho cuando**: la URL pública muestra datos que vienen de Postgres de punta a punta y el login funciona con un usuario invitado. **Aquí se cumple el objetivo de "ver el flujo completo funcionando".**

### Fase 4: vista avanzada con paridad con Streamlit (16 a 24 h)
- [ ] Pestañas de desempeño del modelo, explicabilidad, watch list, compliance y la calculadora de scoring
- [ ] Panel de admin: invitar usuarios, asignar rol y cuota, promover un modelo `candidate` a `active`
- [ ] Portar las agregaciones de NB02 a `build_gold.py`
- **Hecho cuando**: hay paridad de funciones con Streamlit.

### Fase 5: agente, reentrenamiento y monitoreo (16 a 24 h)
- [ ] Agente de NB05 en `pipelines/surveillance/`, con revisión del sandbox `exec()`
- [ ] Jobs en la VM: `surveillance`, `retrain` (solo manual) y `monitoring` (PSI contra el baseline del modelo activo), con registro en `ops.agent_runs` y `ops.monitoring_alerts`
- [ ] Disparo desde la vista avanzada (consumiendo la cuota) o desde GitHub Actions
- **Hecho cuando**: una corrida real del agente desde la UI deja su fila en `ops.agent_runs` con la fase E en verde.

### Fase 6: endurecimiento y cierre (3 a 5 h)
- [ ] Caddy sale de CREW-production a una carpeta propia (`~/edge`), como entrada compartida por los cuatro proyectos, con un subdominio por proyecto
- [ ] Backups automáticos de Postgres (`pg_dump` programado) hacia OCI Object Storage
- [ ] `unattended-upgrades`, `fail2ban` y renombrar la VM a `bds-prod`
- [ ] QA lado a lado con Streamlit (KPIs, scores y SHAP para un conjunto fijo de clientes)
- [ ] README con el procedimiento de rollback y tag `v1.0-produccion`

---

## 8. Rollback

| Capa | Cómo se vuelve atrás |
|---|---|
| Código y despliegue | `git revert` del commit en `dev` y push: el pipeline vuelve a probar y desplegar la versión anterior. En emergencia, en la VM: `git checkout <tag>` y `docker compose up -d --build` |
| Base de datos | `pg_dump` antes de cada migración riesgosa; si algo sale mal, se restaura el dump o se aplica una migración nueva que corrige. Como los datos son reproducibles desde `data/raw/`, el último recurso es recrear la base y volver a correr el pipeline |
| Modelo | Cambiar el modelo `active` en `ml.model_registry` por el anterior |
| VM completa | Recrear la instancia desde cero toma menos de una hora: el repo, el `.env` y los backups bastan |

---

## 9. Consideraciones del plan Always Free de OCI

- **Recuperación de instancias inactivas**: Oracle puede reclamar una instancia Always Free si durante 7 días su uso de CPU, red y memoria queda por debajo del 20%. Una API con poco tráfico cae en ese caso. **Acción**: pasar la cuenta a Pay As You Go (sigue en $0 dentro de los límites gratuitos) y crear una alerta de presupuesto de $1.
- **Límites A1**: 4 OCPU y 24 GB de RAM en total, y 200 GB de disco entre todos los volúmenes. Hoy se usan 2 OCPU, 12 GB y 100 GB.
- **Los cuatro proyectos en una VM**: se vigila el uso de memoria al sumar SAFE (590K transacciones). Si hace falta, la VM sube a 4 OCPU / 24 GB sin recrearla.

---

## 10. Verificación general

- Cada capa se prueba sola antes de conectar la siguiente (primero curl contra `/api/score`, después la UI). Nunca se depuran dos capas a la vez.
- Paridad numérica contra Streamlit en cada fase que toca el scoring.
- CI en verde como condición para desplegar y para mergear a `main`.

## Estimación

Quedan ~50 a 80 horas de trabajo efectivo. El **flujo completo funcionando** (Fases 1 a 3) son ~18 a 28 h. Camino crítico: 1, 2, 3, 4, 6; la Fase 5 puede ir en paralelo a la 4.
