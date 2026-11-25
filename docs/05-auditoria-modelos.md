# Auditoría de modelos, selección y análisis multivariado

Fecha de revisión: 2026-11-24. Alcance: código de entrenamiento de
`profitability_model.py` y `returns_model.py`, criterio de selección, módulo no
supervisado y viabilidad de un análisis multivariado. Los números citados
provienen del conjunto **simulado** `data/demo_simulado.csv` (1 500 ventas);
sirven para validar el pipeline, no como resultados de negocio.

## 1. Hallazgos de la auditoría y correcciones aplicadas

| # | Hallazgo | Riesgo | Corrección |
|---|---|---|---|
| 1 | `scripts/train.py` importaba `src` antes de añadir la raíz a `sys.path` | El CLI fallaba con `ModuleNotFoundError` | El `sys.path.insert` va antes de los imports |
| 2 | El `print` de devoluciones partía un f-string en dos líneas (solo válido en Python ≥ 3.12) | El script no corría en 3.11 | Línea reconstruida |
| 3 | Cada modelo elegía su "mejor" con su propia regla y **sin comprobar que superara al baseline** | Se podía publicar un modelo peor que la media/prior | Política única en `src/selection.py` |
| 4 | No se guardaba la justificación de la selección junto al `.joblib` | Imposible auditar después por qué se eligió un modelo | Se persiste `seleccion` en el bundle |
| 5 | `IsolationForest(contamination="auto")` marcaba **54 %** de las ventas como anomalía | La alerta de anomalías era inútil | `contamination=0.05` (5 % operativo) |
| 6 | El perfil de clústeres duplicaba columnas (`unidades_x`, `unidades_y`) y dejaba `cargo_venta` con signo negativo | Tabla confusa / lectura económica errónea | Se eliminó la duplicación y se usan magnitudes |
| 7 | El VIF devolvía `NaN` si existía una columna constante (`descuentos=0`) | El diagnóstico multivariado quedaba inservible | Se descartan columnas constantes/vacías y se imputa la mediana |
| 8 | Un fallo de XGBoost (u otra librería opcional) abortaba todo el entrenamiento | Pérdida total del análisis | Cada modelo se entrena en `try/except`; se reporta `errores` |
| 9 | `src/unsupervised_model.py` estaba **vacío** y el CLI ya lo importaba | El proyecto no arrancaba | Módulo implementado y probado |
| 10 | `features.py` usa `unidades` y `venta_publicidad`, que se conocen **al momento de la venta**, no estrictamente antes | La afirmación "solo datos pre-venta" es optimista para decisiones previas a publicar | Documentado como limitación; para decidir antes de publicar hay que excluirlas |

## 2. Criterio de selección (único y verificable)

Implementado en `src/selection.py` y usado por ambos modelos:

- **Rentabilidad (regresión):** menor **RMSE** en el 20 % más reciente
  (validación temporal). Debe mejorar al baseline en **≥ 1 % relativo**.
- **Devoluciones (clasificación):** mayor **ROC-AUC**. Debe mejorar al baseline
  en **≥ 0.01 absoluto**.
- El baseline (`DummyRegressor` media / `DummyClassifier` prior) siempre está en
  la tabla y sirve de referencia.
- Si ningún candidato supera el umbral, el modelo se conserva pero se marca
  `confiable = False` y la interfaz muestra una advertencia: **no hay evidencia
  de mejora real**.
- El desempate (p. ej. dos modelos con el mismo ROC-AUC) respeta el orden de la
  tabla, que prioriza modelos más simples/interpretables por construcción.

Nunca se selecciona con métricas de entrenamiento ni con el conjunto completo.

## 3. Resultado: ¿cuál es el mejor modelo?

Con los datos simulados actuales:

| Tarea | Mejor modelo | Métrica (prueba) | Baseline | Veredicto |
|---|---|---|---|---|
| Rentabilidad | **Random Forest** | RMSE 605.39 · MAE 343.29 · R² 0.438 | RMSE 814.28 | **Mejora clara (25.7 %)**; usable |
| Devoluciones | **Regresión logística** | ROC-AUC 0.614 · F1 0.410 · Recall 0.621 | ROC-AUC 0.500 | Mejora modesta; **solo triaje** |

Conclusión:

- **Mejor modelo del proyecto: Random Forest de rentabilidad.** Es el único con
  evidencia sólida de valor predictivo (R² 0.438 y 26 % menos error que el
  baseline). Su variable dominante es `ingreso_esperado` (unidades × precio),
  seguida del mes y el transportista.
- El modelo de devoluciones apenas supera el azar (ROC-AUC 0.61). Sirve para
  **priorizar revisiones** por bandas de riesgo, no para decidir automáticamente.
  La regresión logística empata con Random Forest en AUC (0.614) y se prefiere
  por interpretabilidad; con solo 66 positivos en prueba, la diferencia es
  estadísticamente frágil.
- XGBoost no se evaluó localmente (no está instalado); el código lo incluye
  cuando la dependencia existe (contenedor Docker).

## 4. Aprendizaje no supervisado

`src/unsupervised_model.py` implementa:

- **KMeans** sobre las mismas variables pre-venta que los modelos supervisados.
  `k` se elige por **silueta** (2–8 clústeres) y se reporta también la inercia.
- **IsolationForest** para anomalías, con 5 % de contaminación.
- **PCA de 2 componentes** para visualizar los segmentos.

Resultado con los datos simulados:

- Mejor `k = 2`, silueta **0.206** → estructura **débil**; los segmentos son
  exploratorios, no categorías estables.
- Segmento 0: 1 098 ventas, ticket medio 867 MXN, 17.4 % devoluciones.
- Segmento 1: 402 ventas, ticket medio 2 665 MXN, 15.2 % anomalías internas y
  18.6 % devoluciones. Es el grupo de **ticket alto y mayor valor**, pero también
  el más atípico.
- 75 ventas (5 %) marcadas como anomalía, concentradas en SKU-009 y
  `Mercado Envíos Flex`.

Los clústeres **no usan** el resultado económico ni la devolución para formarse;
esas columnas solo interpretan el segmento (evita fuga de información).

## 5. ¿Aplica un análisis multivariado?

**Sí, como análisis exploratorio**, no como inferencia formal:

- Hay varias variables numéricas por venta y están correlacionadas
  (`ingreso_esperado` ≈ `cargo_venta`, r ≈ 1.00).
- **PCA:** 3 componentes explican > 80 % de la varianza; PC1 sola explica 50.7 %.
  Es útil para resumir y visualizar.
- **VIF:** `ingreso_esperado`, `cargo_venta` y `total` superan 10 (redundancia
  severa). Para modelar no deben usarse juntas; para segmentar conviene PCA.
- **No aplica MANOVA/Hotelling** como prueba inferencial: los datos son un
  **registro completo** de ventas (no una muestra aleatoria), con colinealidad
  extrema y sin normalidad multivariada garantizada. Usar p-valores allí sería
  incorrecto.

## 6. Limitaciones y siguientes pasos

1. **Una sola partición temporal**: métricas con alta varianza. Añadir
   `TimeSeriesSplit` y reportar media ± desviación.
2. **Desbalance en devoluciones**: calibrar probabilidades (Platt/isotónica) y
   elegir el umbral por costo (falso negativo vs revisión manual).
3. **Silueta baja**: probar `DBSCAN`/`HDBSCAN` o reducir dimensionalidad antes de
   KMeans; validar estabilidad con remuestreo.
4. **Datos reales**: las métricas de demostración no son resultados de negocio.
5. **Variables al momento de la venta**: separar un modelo "pre-publicación" si
   la decisión debe tomarse antes de vender.
6. **Monitoreo**: deriva de datos (PSI) y recalibración periódica.

## 7. Reproducir

```bash
pip install -r requirements.txt
python scripts/train.py data/demo_simulado.csv
python -m unittest discover -s tests -v
streamlit run app.py     # sección «Segmentación y anomalías»
```
