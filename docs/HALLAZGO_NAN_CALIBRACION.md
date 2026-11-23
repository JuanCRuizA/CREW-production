# Hallazgo: política de NaN y calibración de crew-v1

**Fecha**: 2026-11-08. **Alcance**: documentar un hallazgo. No se recalibró nada y no se cambió el bundle (`models/crew-v1` queda idéntico).

## 1. Resumen

El pipeline legacy de Streamlit (y por paridad, `crew-v1`) rellena los NaN numéricos con 0, convierte infinitos en ±10 y manda el nulo categórico a `"MISSING"`. El entrenamiento (NB02 y NB03) hizo otra cosa: infinito a NaN, los NaN se dejan a XGBoost y el nulo categórico pasa por `astype(str)`, es decir, la clase `"nan"`.

Sobre las 307 511 filas del snapshot, y reconstruyendo la partición train/test del model card:

- Con NaN nativo se aproximan (PD por decil dentro de 0.003) las métricas y la tabla de deciles del model card; el AUC en test es 0.7775 contra 0.7778.
- Con el relleno a 0 del legacy el AUC en test baja a 0.7629 y la PD calibrada media es 0.0568 contra un default observado de 0.0807 (subestima un 30%).

Consecuencia para la API del paso 2: `pd_raw` no es una probabilidad (media 0.32) y `pd_calibrated`, bajo la política de `crew-v1`, subestima. Por eso se exponen las dos sin campo titular y sin tier, solo con las banderas `above_statistical` y `above_business` en escala cruda.

## 2. Evidencia

Todas las cifras salen de `data/legacy/portfolio_surveillance.db` (tabla `engineered_features`, abierta con `mode=ro`) y de los `.pkl` de `models/legacy`. El default observado es `TARGET`.

### Tabla 1: PD media contra default observado

| Variante | Subconjunto | n | AUC | PD cruda media | PD calibrada media | Observado | calibrada / observado | Brier (calibrada) |
|---|---|---|---|---|---|---|---|---|
| NaN a 0 (legacy, crew-v1) | train | 246 008 | 0.8157 | 0.3238 | 0.0570 | 0.0807 | 0.706 | 0.0654 |
| NaN a 0 (legacy, crew-v1) | test | 61 503 | 0.7629 | 0.3234 | 0.0568 | 0.0807 | 0.703 | 0.0682 |
| NaN nativo | train | 246 008 | 0.8372 | 0.3770 | 0.0788 | 0.0807 | 0.976 | 0.0628 |
| NaN nativo | test | 61 503 | 0.7775 | 0.3764 | 0.0783 | 0.0807 | 0.970 | 0.0667 |

Model card (test): AUC 0.7778, Brier 0.0668.

### Tabla 2: deciles en test, estilo model card

Los deciles se forman con `pd.qcut` sobre el valor calibrado (`duplicates="drop"`), lo que aproxima los conteos desiguales del card.

| Decil | Card n | Card pred | Card actual | NaN a 0: n | pred | actual | NaN nativo: n | pred | actual |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 6460 | 0.0007 | 0.0098 | 7382 | 0.0004 | 0.0110 | 6545 | 0.0007 | 0.0095 |
| 2 | 6157 | 0.0042 | 0.0182 | 5250 | 0.0026 | 0.0213 | 6185 | 0.0042 | 0.0186 |
| 3 | 6967 | 0.0096 | 0.0274 | 6640 | 0.0051 | 0.0280 | 7165 | 0.0096 | 0.0272 |
| 4 | 5791 | 0.0190 | 0.0300 | 6569 | 0.0103 | 0.0344 | 5778 | 0.0190 | 0.0317 |
| 5 | 5454 | 0.0313 | 0.0409 | 6077 | 0.0189 | 0.0479 | 5552 | 0.0313 | 0.0418 |
| 6 | 6817 | 0.0506 | 0.0616 | 5438 | 0.0312 | 0.0725 | 6801 | 0.0507 | 0.0628 |
| 7 | 5553 | 0.0745 | 0.0791 | 6311 | 0.0506 | 0.0776 | 5183 | 0.0737 | 0.0799 |
| 8 | 6197 | 0.1125 | 0.0971 | 5876 | 0.0784 | 0.1048 | 6522 | 0.1109 | 0.0981 |
| 9 | 6399 | 0.1769 | 0.1571 | 5934 | 0.1264 | 0.1508 | 6327 | 0.1772 | 0.1596 |
| 10 | 5708 | 0.3446 | 0.3041 | 6026 | 0.2584 | 0.2778 | 5445 | 0.3416 | 0.3098 |

Diferencia máxima de la PD predicha contra el card: 0.0862 con NaN a 0, 0.0030 con NaN nativo.

### Tabla 3: banderas y bandas solo en test (n = 61 503)

| Variante | `above_statistical` (>= 0.5157) | default observado | `above_business` (>= 0.79) | default observado |
|---|---|---|---|---|
| NaN a 0 | 19.29% | 21.56% | 2.19% | 40.73% |
| NaN nativo | 26.96% | 19.28% | 4.50% | 37.85% |

Bandas del model card (0.3 / 0.5 / 0.7 sobre la PD cruda, solo informativas): con NaN a 0 reparten 53.7 / 25.5 / 14.8 / 6.0% con default 2.9 / 8.5 / 16.3 / 32.2%; con NaN nativo, 43.5 / 27.7 / 18.5 / 10.3% con default 2.2 / 6.3 / 12.7 / 29.3%. Las banderas cambian de tamaño 1.4 veces (estadística) y 2 veces (negocio) según la política de NaN.

## 3. Por qué el decil 6 no coincide con el model card

El card predice 0.0506 en el decil 6; con el pipeline legacy sale 0.0312 y con NaN nativo 0.0507.

1. **Causa principal: el tratamiento de NaN.** El card se aproxima con NaN nativo y no con el relleno a 0.
2. **Causa menor: la segmentación.** Una primera tabla usaba deciles de igual tamaño por rango (`rank(method="first")`) y daba 0.0294 en el decil 6. Los deciles por valor (`qcut`) del card dan 0.0312. Es una diferencia de 0.002, pequeña frente a la anterior.
3. **Código de entrenamiento.** NB02 (línea 2458) hace `df.replace([np.inf, -np.inf], np.nan)`. NB03 declara que XGBoost maneja los faltantes de forma nativa y codifica con `X[col].astype(str)` seguido de `LabelEncoder` (el nulo queda como la clase `"nan"`).

El pipeline legacy y `crew-v1` difieren del entrenamiento en tres puntos: NaN a 0, infinito a ±10 y nulo categórico a `"MISSING"`. Solo el primero se aisló numéricamente aquí.

## 4. Método

Entorno: Python 3.11.14, xgboost 3.1.2, scikit-learn 1.8.0, numpy 2.3.5. Un pase completo sobre las 307 511 filas tarda unos 70 segundos. Se lee por bloques de 40 000 filas.

**Preprocesado de cada variante.** Ambas aplican los encoders legacy (`label_encoders_v2.pkl`, nulo categórico a `"MISSING"`) y convierten infinitos en ±10. Difieren solo en el numérico nulo: la variante legacy usa `fillna(0)`; la variante nativa deja el NaN a XGBoost. La PD calibrada es `np.interp(pd_raw.astype(float64), X_thresholds_, y_thresholds_)`.

**Partición.** Se reconstruye con `train_test_split(np.arange(n), test_size=0.2, stratify=TARGET, random_state=42)`, que son los parámetros del model card. Validación: da 246 008 y 61 503 filas, igual que el card, y con NaN nativo el AUC en test es 0.7775 contra 0.7778 del card. Se asume que el orden de las filas del snapshot es el del DataFrame de entrenamiento.

**Deciles estilo card.** `pd.qcut(pd_calibrada_test, 10, labels=False, duplicates="drop") + 1`, con conteo, media de la PD calibrada y media de `TARGET` por decil.

**Banderas.** `pd_raw >= umbral` con los umbrales de `thresholds.pkl` (`statistical_optimal` 0.5157342553138733, `business_optimal` 0.7900000000000001).

```python
import sqlite3, joblib, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import train_test_split

M = "models/legacy/"
model = joblib.load(M + "xgb_credit_model.pkl"); names = list(joblib.load(M + "feature_names.pkl"))
enc = joblib.load(M + "label_encoders_v2.pkl"); cal = joblib.load(M + "calibrator.pkl")
con = sqlite3.connect("file:data/legacy/portfolio_surveillance.db?mode=ro", uri=True)
cols = ", ".join(f'"{c}"' for c in ["TARGET"] + names)

fill0, native, ys = [], [], []
for chunk in pd.read_sql_query(f"SELECT {cols} FROM engineered_features", con, chunksize=40000):
    X = chunk[names].copy()
    for c, le in enc.items():  # categorical null -> "MISSING"
        X[c] = le.transform(X[c].astype(object).where(X[c].notna(), "MISSING"))
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], [10, -10])
    fill0.append(model.predict_proba(X.fillna(0))[:, 1])  # legacy / crew-v1
    native.append(model.predict_proba(X)[:, 1])           # NaN left to XGBoost
    ys.append(chunk["TARGET"].values)

y = np.concatenate(ys)
train, test = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
for name, raw in (("NaN->0", np.concatenate(fill0)), ("native", np.concatenate(native))):
    calibrated = np.interp(raw.astype(np.float64), cal.X_thresholds_, cal.y_thresholds_)
    for part, idx in (("train", train), ("test", test)):
        print(name, part, roc_auc_score(y[idx], raw[idx]), raw[idx].mean(), calibrated[idx].mean(),
              y[idx].mean(), brier_score_loss(y[idx], calibrated[idx]))
    d = pd.DataFrame({"cal": calibrated[test], "y": y[test]})
    d["dec"] = pd.qcut(d["cal"], 10, labels=False, duplicates="drop") + 1
    print(d.groupby("dec").agg(n=("y", "size"), pred=("cal", "mean"), actual=("y", "mean")))
```

## 5. Limitaciones

- **La paridad del paso 1 no valida el pipeline legacy.** Prueba que el bundle copia el pipeline legacy de Streamlit (diferencia máxima 0 con tolerancia 1e-6), no que ese pipeline sea fiel al entrenamiento. Este hallazgo muestra justamente que no lo es.
- **Incluso con NaN nativo, la calibración no es buena en los extremos.** Los deciles bajos se subestiman (decil 1: predice 0.0007 y se observa 0.0095) y los altos se sobrestiman (decil 10: predice 0.3416 y se observa 0.3098). Es compatible con un calibrador ajustado sobre train: con NaN nativo el AUC es 0.837 en train contra 0.7775 en test, o sea el modelo discrimina mejor en las filas con las que se ajustó el calibrador que en las de test. No se midió la forma de las predicciones de train ni se probó otra causa.
- **La partición train/test es una reconstrucción.** Está respaldada por los tamaños y por el AUC, pero no se leyó del código de entrenamiento.
- **Solo se aisló el efecto de NaN a 0.** El efecto de la clase `"nan"` contra `"MISSING"` en las categóricas y el de infinito a NaN contra ±10 no se cuantificaron. En el snapshot no hay infinitos, así que el segundo no se pudo medir con estos datos. Las categóricas podrían explicar parte de los conteos distintos por decil con NaN nativo (6545 contra 6460 en el decil 1); no se comprobó.
- **Train está en muestra y el origen del calibrador no está verificado.** El 80% de las filas se usó para entrenar el modelo, y el calibrador pudo ajustarse con esas mismas filas de train, pero no se comprobó (no se leyó el código que lo ajustó). Por eso las tablas separan train y test y las bandas se calculan solo con test.
- **Las bandas son descriptivas.** Los cortes 0.3 / 0.5 / 0.7 son los del model card; no se evaluaron como esquema de tiers.

## 6. Implicaciones

- **Modelo (SR 11-7).** Es un hallazgo de análisis de resultados: el pipeline de inferencia se desvía del de entrenamiento y degrada el rendimiento (AUC en test 0.7629 contra 0.7775). Debe quedar en el inventario y en el model card antes de usar la PD para decisiones.
- **IFRS 9.** El ECL necesita una PD calibrada; la calibrada actual subestima un 30% en media bajo la política de `crew-v1`.
- **EU AI Act (Art. 15, exactitud) y transparencia.** Documentar que la PD expuesta no coincide con el entrenamiento forma parte de la gestión de exactitud de un sistema de alto riesgo.

## 7. Opciones para un futuro crew-v2 (no ejecutadas)

1. Política de NaN alineada con el entrenamiento: NaN numérico nativo (`numeric_nan_value = null`, hoy rechazado por el cargador como no soportado), infinito a NaN y nulo categórico a la clase `"nan"` con encoders construidos como NB03.
2. Recalibrar sobre datos que no sean los de entrenamiento del modelo, una vez fijada la política de NaN.
3. Mantener `crew-v1` como referencia de paridad con Streamlit y no presentarlo como PD.

Cualquiera de las tres exige regenerar el bundle y su paridad, y queda fuera del alcance de este documento.
