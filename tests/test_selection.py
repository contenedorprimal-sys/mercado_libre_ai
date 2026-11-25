"""Pruebas del criterio compartido de selección de modelos.

Ejecutar desde la raíz del proyecto:
    python -m unittest discover -s tests -v
"""
import unittest

import pandas as pd

from src.selection import choose_model


class TestChooseModel(unittest.TestCase):
    def test_regression_picks_lowest_rmse_and_beats_baseline(self):
        metrics = pd.DataFrame({
            "Modelo": ["Baseline (media)", "Linear Regression", "Random Forest"],
            "RMSE": [800.0, 650.0, 600.0],
        })
        sel = choose_model(metrics, "regression")
        self.assertEqual(sel["best"], "Random Forest")
        self.assertTrue(sel["confiable"])
        self.assertGreater(sel["improvement"], 0.01)

    def test_regression_flags_model_that_does_not_beat_baseline(self):
        metrics = pd.DataFrame({
            "Modelo": ["Baseline (media)", "Linear Regression", "Random Forest"],
            "RMSE": [600.0, 600.5, 650.0],
        })
        sel = choose_model(metrics, "regression")
        self.assertEqual(sel["best"], "Linear Regression")
        self.assertFalse(sel["confiable"])
        self.assertIn("no demuestra", sel["reason"])

    def test_classification_picks_highest_auc(self):
        metrics = pd.DataFrame({
            "Modelo": ["Baseline (prior)", "Logistic Regression", "Random Forest"],
            "ROC-AUC": [0.50, 0.62, 0.60],
        })
        sel = choose_model(metrics, "classification")
        self.assertEqual(sel["best"], "Logistic Regression")
        self.assertTrue(sel["confiable"])

    def test_classification_flags_tiny_gain(self):
        metrics = pd.DataFrame({
            "Modelo": ["Baseline (prior)", "Random Forest"],
            "ROC-AUC": [0.50, 0.505],
        })
        sel = choose_model(metrics, "classification")
        self.assertFalse(sel["confiable"])

    def test_empty_metrics_is_safe(self):
        sel = choose_model(pd.DataFrame(), "regression")
        self.assertIsNone(sel["best"])


if __name__ == "__main__":
    unittest.main()
