"""Pruebas del modelo no supervisado (KMeans + IsolationForest + PCA)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import unsupervised_model as um


def _frame(n=150, seed=1):
    rng = np.random.default_rng(seed)
    precio = rng.normal(800, 200, n)
    unidades = rng.integers(1, 4, n).astype(float)
    return pd.DataFrame({
        "id_venta": range(n),
        "fecha_venta": pd.date_range("2025-01-01", periods=n, freq="D"),
        "sku": rng.choice(["SKU-1", "SKU-2", "SKU-3"], n),
        "unidades": unidades,
        "precio_unitario": precio,
        "tipo_publicacion": rng.choice(["Clásica", "Premium"], n),
        "venta_publicidad": rng.choice(["Sí", "No"], n),
        "forma_de_entrega": rng.choice(["Mercado Envíos", "Flex"], n),
        "transportista": rng.choice(["DHL", "Estafeta"], n),
        "total": precio * unidades * rng.normal(1, 0.1, n),
        "ingresos_productos": precio * unidades,
        "costos_envio": -rng.normal(80, 10, n),
        "cargo_venta": -precio * unidades * 0.1,
    })


class TestUnsupervised(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = _frame()
        cls.res = um.fit_unsupervised(cls.df)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.tmp.name) / "unsupervised.joblib"
        um.save_model(cls.res, cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_viable_y_estructura(self):
        self.assertTrue(self.res["viable"])
        self.assertGreaterEqual(self.res["best_k"], 2)
        self.assertTrue(0 < self.res["pct_anomalias"] <= 6)

    def test_perfil_sin_columnas_duplicadas(self):
        cols = self.res["resumen_clusters"].columns
        self.assertEqual(len(cols), len(set(cols)))
        self.assertIn("n_registros", cols)
        self.assertEqual(int(self.res["resumen_clusters"]["n_registros"].sum()), len(self.df))

    def test_pca_y_multivariado(self):
        self.assertEqual(len(self.res["varianza_pca2"]), 2)
        self.assertTrue(self.res["multivariado"]["aplica"])
        self.assertIn("pca_1", self.res["pca_2d"].columns)

    def test_prediccion_sobre_datos_nuevos(self):
        out = um.predict_unsupervised(self.df.head(5), path=self.model_path)
        self.assertEqual(len(out), 5)
        self.assertTrue({"cluster", "es_anomalia", "score_anomalia"} <= set(out.columns))


if __name__ == "__main__":
    unittest.main()
