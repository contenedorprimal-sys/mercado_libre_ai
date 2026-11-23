# Diccionario de Datos, Calidad y Privacidad

## Anonimización (PII)
Al ingestar el reporte en `data_processing.py`, se descartan antes de persistir en disco las columnas sensibles: nombre del comprador, usuario/nickname, domicilio, código postal detallado, teléfono, identificación fiscal/documentos y número de guía/seguimiento.

## Esquema procesado (`data/current/`)
- `dataset.parquet` / `.csv`: registros limpios listos para modelado.
- `metadata.json`: fecha de carga, número de filas válidas, filas excluidas por ambigüedad en devoluciones y bandera `has_costs` (`true` = rentabilidad completa, `false` = etiqueta **PARCIAL**).

## Reglas de calidad y exclusión
- **Devoluciones ambiguas:** ventas entregadas hace menos de 30 días sin reclamo ni devolución confirmada se excluyen del entrenamiento de `returns_model.py` para evitar falsos negativos.
