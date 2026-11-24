# Documento de Negocio, Alcance y Limitaciones

## 1. Propósito del proyecto

El **Sistema Inteligente de Predicción de Rentabilidad y Riesgo de Devoluciones (Mercado Libre)** es una aplicación analítica desarrollada con Streamlit y modelos de *machine learning* (con soporte para ejecución local o en Docker) orientada a evaluar el resultado económico por venta y estimar el riesgo de devolución a partir de reportes de ventas de Mercado Libre.

Su objetivo práctico es ofrecer una herramienta visual y reproducible para explorar el desempeño de ventas, identificar patrones por SKU y transportista, y contar con una base metodológica clara sobre los alcances y restricciones del modelo predictivo.

---

## 2. Lo que el sistema SÍ hace

- **Ingesta y depuración de reportes de ventas:** carga archivos `.xlsx` y `.csv` con la estructura de dos filas de encabezado propia de los reportes de Mercado Libre (fila de sección y fila de columna), genera un registro de limpieza (`log`) y reporta indicadores de calidad de datos.
- **Cálculo de resultado económico por venta:** calcula el resultado como `Total (MXN)` de Mercado Libre menos el costo de adquisición cuando se aporta un catálogo `costos.csv` (`sku,costo_unitario`), considerando que `Total` ya integra cargos, envíos, descuentos y reembolsos sin doble conteo.
- **Análisis exploratorio e indicadores visuales:** organiza la información en 8 vistas dentro de Streamlit (*Inicio*, *Carga de datos*, *Análisis exploratorio*, *Rentabilidad*, *Predicción de devoluciones*, *Comparación de modelos*, *Recomendaciones inteligentes* y *Reportes*), incluyendo evolución mensual, top de SKUs, precios, envíos y exportación de resultados.
- **Modelado predictivo sin fuga de información (*no leakage*):** entrena modelos usando únicamente variables previas a la venta (como precio, unidades, SKU, tipo de publicación, envío y mes), evitando usar ingresos finales, cargos, reembolsos o costos reales como predictores.
- **Validación temporal y comparación contra *baseline*:** divide los datos cronológicamente (80 % más antiguo para entrenamiento y 20 % más reciente para prueba) y compara contra un modelo *baseline*, además de calcular importancia de variables por permutación.
- **Estimación de riesgo de devoluciones y recomendaciones:** etiqueta devoluciones (`1` con evidencia de devolución; `0` entregadas hace ≥ 30 días sin reclamo ni devolución), agrupa probabilidades en bandas de riesgo relativo y emite hallazgos orientados a SKU y transportista (por ejemplo, diferencias de comportamiento logístico como `DHL` frente a otros transportistas).
- **Privacidad local y persistencia:** procesa los datos localmente, elimina columnas personales o sensibles al cargar (comprador, domicilio, documentos, seguimiento, etc.) y conserva la sesión y modelos entrenados en `./data/current` y `./models`.

---

## 3. Lo que el sistema NO hace (alcance y limitaciones)

- **No es un sistema transaccional ni conecta en tiempo real con la API de Mercado Libre:** no crea, edita ni cancela publicaciones, no gestiona órdenes en vivo ni sincroniza automáticamente con cuentas de vendedor; opera sobre archivos exportados (`.xlsx` / `.csv`) o datos simulados.
- **No calcula rentabilidad completa sin catálogo de costos:** si no se proporciona `costos.csv` con `sku,costo_unitario`, el cálculo económico se etiqueta como **PARCIAL**, ya que no conoce el costo de adquisición real del producto.
- **No entrena el modelo de devoluciones si faltan casos positivos suficientes:** si existen menos de 30 casos positivos de devolución, el modelo de devoluciones no se entrena y el sistema informa el motivo en lugar de forzar un ajuste poco fiable.
- **No evalúa ventas ambiguas o recientes sin maduración:** las ventas entregadas hace menos de 30 días sin reclamo ni devolución quedan excluidas de la etiqueta negativa `0` para evitar tratar ventas aún en ventana de devolución como definitivas.
- **No establece relaciones causales:** la importancia de variables por permutación mide asociación predictiva dentro del conjunto de datos, no causalidad real ni efectos de intervenciones futuras.
- **No produce resultados reales con datos simulados:** el script `scripts/make_demo_data.py` genera datos sintéticos (`data/demo_simulado.csv`) únicamente para pruebas técnicas y demostración; sus métricas no reflejan operaciones reales.
- **No sustituye asesoría contable, fiscal ni logística:** los resultados y recomendaciones son de apoyo analítico y académico, no una auditoría financiera ni una garantía operativa.

---

## 4. Reglas clave de operación y criterios de calidad

- **Rentabilidad:** `Total (MXN) - costo_unitario` (etiquetada como **PARCIAL** sin catálogo de costos).
- **Partición temporal:** 80 % entrenamiento (más antiguo) / 20 % prueba (más reciente).
- **Umbral mínimo de devoluciones:** ≥ 30 casos positivos para entrenar el modelo de devoluciones; ventana de ≥ 30 días desde la entrega sin reclamo ni devolución para etiqueta `0`.
- **Privacidad:** eliminación de datos personales al ingestar el archivo.
