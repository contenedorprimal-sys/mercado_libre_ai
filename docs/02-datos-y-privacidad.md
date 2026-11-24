# Diccionario de Datos, Simulación y Privacidad

## Formato de reportes de Mercado Libre

Los reportes de entrada utilizan **dos filas de encabezado**:

- **Fila 0 (secciones):** `Ventas` (16 columnas), `Publicidad` (1), `Publicaciones` (7), `Facturación al comprador` (6), `Compradores` (8), `Envíos` (6), `Devoluciones` (19) y `Reclamos` (3).
- **Fila 1 (columnas):** incluye campos como `# de venta`, `Fecha de venta`, `Estado`, `Descripción del estado`, `Paquete de varios productos`, `Pertenece a un kit`, `Unidades`, `Ingresos por productos (MXN)`, `Cargo por venta e impuestos (MXN)`, `Ingresos por envío (MXN)`, `Costos de envío (MXN)`, `Costo de envío basado en medidas y peso declarados`, `Cargo por diferencias en medidas y peso del paquete`, `Descuentos y bonificaciones`, `Anulaciones y reembolsos (MXN)`, `Total (MXN)`, `Orden de compra`, `Venta por publicidad`, `SKU`, `# de publicación`, `Título de la publicación`, `Variante`, `Precio unitario de venta de la publicación (MXN)`, `Tipo de publicación`, `Forma de entrega`, `Fecha en camino`, `Fecha entregado`, `Transportista`, entre otros.

## Simulación de datos (`scripts/make_demo_data.py`)

Genera `data/demo_simulado.csv` (`header=False, index=False`) únicamente para pruebas; sus métricas **no** representan resultados reales:

- **Semilla y tamaño:** `np.random.default_rng(7)` y `N` configurable por argumento CLI (`1500` por defecto).
- **Catálogo simulado:** 10 SKUs (`SKU-001` a `SKU-010`) con precio base `rng.uniform(150, 1500)` y probabilidad base de devolución `rng.uniform(0.02, 0.18)`.
- **Transportistas:** `Estafeta`, `DHL` y `Mercado Envíos Flex` con probabilidades `[0.5, 0.3, 0.2]`; cuando el transportista es `DHL`, la probabilidad de devolución se multiplica por `1.8`.
- **Fechas y cálculo económico:** fechas desde `2025-01-01` más `0..419` días y `0..23` horas; unidades en `[1, 1, 1, 2, 3]`; `ing = round(price * u, 2)`; `cargo = -round(ing * .17, 2)`; `env = -round(rng.uniform(60, 140), 2)`; probabilidad de cancelación del `3 %` si no hay devolución; `refund = -ing` si hay devolución (`0.0` en otro caso); `total = round(ing + cargo + env + refund, 2)`.

## Privacidad y persistencia

- El procesamiento es local y elimina columnas de datos personales o sensibles al cargar (como `Comprador`, `Domicilio`, `Municipio/Alcaldía`, `Estado`, `Código postal`, `País`, `Número de seguimiento`, `URL de seguimiento`, entre otras).
- La sesión activa se conserva en `data/current/` y los modelos entrenados en `models/` (`profitability.joblib`, `returns.joblib`).
