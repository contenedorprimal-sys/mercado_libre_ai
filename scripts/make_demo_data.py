"""Genera un reporte SIMULADO con la misma estructura de Mercado Libre (2 filas de encabezado).
SOLO para probar el sistema. Los resultados obtenidos con estos datos NO son reales."""
import numpy as np, pandas as pd, sys
rng = np.random.default_rng(7)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
skus = {f"SKU-{i:03d}": (rng.uniform(150, 1500), rng.uniform(0.02, 0.18)) for i in range(1, 11)}
carriers = ["Estafeta", "DHL", "Mercado Envíos Flex"]
start = pd.Timestamp("2025-01-01")
rows = []
for i in range(N):
    sku = rng.choice(list(skus)); price, p_ret = skus[sku]
    car = rng.choice(carriers, p=[.5, .3, .2]); p_ret *= 1.8 if car == "DHL" else 1
    f = start + pd.Timedelta(days=int(rng.integers(0, 420)), hours=int(rng.integers(0, 24)))
    u = int(rng.choice([1, 1, 1, 2, 3])); ing = round(price * u, 2)
    cargo = -round(ing * .17, 2); env = -round(rng.uniform(60, 140), 2)
    ent = f + pd.Timedelta(days=int(rng.integers(2, 7)))
    ret = rng.random() < p_ret; cancel = (not ret) and rng.random() < .03
    refund = -ing if ret else 0.0
    total = round(ing + cargo + env + refund, 2)
    est = "Cancelada" if cancel else ("Devolución" if ret else "Entregado")
    r = [100000 + i, f, est, est, "No", "No", u, ing, cargo, 0, env, None, None, 0, refund, total,
         None, rng.choice(["Sí", "No"]), sku, f"MLM{i}", "Titulo", None, price / 1,
         rng.choice(["Clásica", "Premium"]), None, None, None, None, None, None, None, None,
         "Comprador X", None, None, None, "Mun", "Edo", "00000", "México",
         "Mercado Envíos", ent - pd.Timedelta(days=2), None if cancel else ent, car, "123", "url"]
    dev = [None] * 6 + [None] * 6 + [None] * 7 + [None] * 3
    if ret:
        dev[0] = u; dev[2] = ent + pd.Timedelta(days=3); dev[3] = ent + pd.Timedelta(days=6)
    rows.append(r + dev)
secs = (["Ventas"] * 16 + ["Publicidad"] + ["Publicaciones"] * 7 + ["Facturación al comprador"] * 6
        + ["Compradores"] * 8 + ["Envíos"] * 6 + ["Devoluciones"] * 19 + ["Reclamos"] * 3)
head = ("# de venta,Fecha de venta,Estado,Descripción del estado,Paquete de varios productos,Pertenece a un kit,"
 "Unidades,Ingresos por productos (MXN),Cargo por venta e impuestos (MXN),Ingresos por envío (MXN),Costos de envío (MXN),"
 "Costo de envío basado en medidas y peso declarados,Cargo por diferencias en medidas y peso del paquete,"
 "Descuentos y bonificaciones,Anulaciones y reembolsos (MXN),Total (MXN),Orden de compra,Venta por publicidad,SKU,"
 "# de publicación,Título de la publicación,Variante,Precio unitario de venta de la publicación (MXN),Tipo de publicación,"
 "Factura adjunta,Datos personales o de empresa,Tipo y número de documento,Dirección,Tipo de contribuyente,CFDI,"
 "Tipo de usuario,Régimen Fiscal,Comprador,Negocio,IFE,Domicilio,Municipio/Alcaldía,Estado,Código postal,País,"
 "Forma de entrega,Fecha en camino,Fecha entregado,Transportista,Número de seguimiento,URL de seguimiento,"
 "Unidades,Forma de entrega,Fecha en camino,Fecha entregado,Transportista,Número de seguimiento,URL de seguimiento,"
 "Revisado por Mercado Libre,Fecha de revisión,Dinero a favor,Resultado,Destino,Motivo del resultado,Unidades,"
 "Reclamo abierto,Reclamo cerrado,Con mediación").split(",")
n = len(head)
secs = (secs + [secs[-1]] * n)[:n]
rows = [(r + [None] * n)[:n] for r in rows]
out = pd.DataFrame([secs, head] + rows)
out.to_csv("data/demo_simulado.csv", header=False, index=False)
print("Generado data/demo_simulado.csv (SIMULADO)", out.shape)
