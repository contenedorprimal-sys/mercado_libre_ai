"""Pruebas del análisis multivariado (correlación, VIF y PCA)."""
import unittest

import numpy as np
import pandas as pd

from src import multivariate as mv


def _frame(n=120, seed=0):
    rng = np.random.default_rng(seed)
    precio = rng.normal(500, 100, n)
    unidades = rng.integers(1, 4, n).astype(float)
    return pd.DataFrame({
        "unidades": unidades,
        "precio_unitario": precio,
        "ingreso_esperado": precio * unidades,
        "costos_envio": rng.normal(50, 10, n),
        "descuentos": 0.0,  # columna constante: debe excluirse
    })


class TestMultivariate(unittest.TestCase):
    def test_report_aplica_y_limpia_constantes(self):
        rep = mv.multivariate_report(_frame())
        self.assertTrue(rep["aplica"])
        self.assertNotIn("descuentos", rep["correlacion"].columns)
        self.assertNotIn("descuentos", rep["vif"]["variable"].tolist())

    def test_vif_es_finito_y_detecta_colinealidad(self):
        rep = mv.multivariate_report(_frame())
        vif = rep["vif"]
        self.assertTrue(np.isfinite(vif["VIF"]).all())
        # ingreso_esperado es casi una copia de precio × unidades.
        fila = vif.set_index("variable").loc["ingreso_esperado", "VIF"]
        self.assertGreater(fila, 10)

    def test_pca_da_varianza_explicada(self):
        pca = mv.multivariate_report(_frame())["pca"]
        self.assertGreater(pca["varianza_explicada"][0], 0)
        self.assertLessEqual(sum(pca["varianza_explicada"]), 1.0001)
        self.assertGreaterEqual(pca["n_componentes_80"], 1)

    def test_datos_insuficientes_no_aplica(self):
        rep = mv.multivariate_report(pd.DataFrame({"a": [1, 2, 3]}))
        self.assertFalse(rep["aplica"])


if __name__ == "__main__":
    unittest.main()
