# Diccionario de Datos, Simulación y Privacidad

## Formato de reportes de Mercado Libre

- Los reportes cuentan con **dos filas de encabezado**: la primera fila agrupa por sección (`Ventas`, `Publicidad`, `Publicaciones`, `Facturación al comprador`, `Compradores`, `Envíos`, `Devoluciones`, `Reclamos`) y la segunda fila contiene los nombres de columna (por ejemplo: `# de venta`, `Fecha de venta`, `Estado`, `Descripción del estado`, `Unidades`, `Ingresos por productos (MXN)`, `Cargo por venta e impuestos (MXN)`, `Costos de envío (MXN)`, `Anulaciones y reembolsos (MXN)`, `Total (MXN)`, `SKU`, `# de publicación`, `Precio unitario de venta de la publicación (MXN)`, `Tipo de publicación`, `Forma de entrega`, `Fecha en camino`, `Fecha entregado`, `Transportista`).

## Simulación de datos (`scripts/make_demo_data.py`)

- **Semilla y tamaño:** `np.random.default_rng(7)` y `N` configurable por argumento (`1500` por defecto).
- **Entidades simuladas:** 10 SKUs (`SKU-001` a `SKU-010`) con precio base uniforme en `[150, 1500)` y probabilidad base de devolución en `[0.02, 0.18)`; 3 transportistas (`Estafeta` `0.5`, `DHL` `0.3`, `Mercado Envíos Flex` `0.2`), multiplicando por `1.8` la probabilidad de devolución cuando el transportista es `DHL`.
- **Fechas y montos:** fechas desde `2025-01-01` en un rango de `420` días; unidades en `[1, 1, 1, 2, 3]`; comisión `-round(ing * .17, 2)`; envío `-round(rng.uniform(60, 140), 2)`; cancelación del `3 %` cuando no hay devolución; `Total` como `round(ing + cargo + env + refund, 2)`.

## Privacidad y persistencia

- Se eliminan datos personales/sensibles al cargar (como `Comprador`, `Domicilio`, `Municipio/Alcaldía`, `Estado`, `Código postal`, `País`, `Número de seguimiento`, `URL de seguimiento`, entre otros presentes en el reporte).
- La sesión activa se guarda en `data/current/` y los modelos entrenados en `models/` (`profitability.joblib`, `returns.joblib`).
- `[Completar con la lista exacta de columnas eliminadas y campos de quality_report / metadata desde src/data_processing.py]`.
