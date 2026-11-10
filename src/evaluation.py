"""Métricas, división temporal e importancia de variables."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score)


def temporal_split(df: pd.DataFrame, date_col="fecha_venta", frac=0.8):
    """Entrena con el pasado y prueba con el futuro (sin mezclar al azar)."""
    d = df.sort_values(date_col).reset_index(drop=True)
    cut = int(len(d) * frac)
    return d.iloc[:cut], d.iloc[cut:]


def regression_metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def classification_metrics(y, p, proba):
    m = {"Accuracy": accuracy_score(y, p),
         "Precision": precision_score(y, p, zero_division=0),
         "Recall": recall_score(y, p, zero_division=0),
         "F1": f1_score(y, p, zero_division=0),
         "ROC-AUC": roc_auc_score(y, proba) if len(set(y)) == 2 else np.nan}
    m["Matriz de confusión"] = str(confusion_matrix(y, p, labels=[0, 1]).tolist())
    return m


def importance_table(pipe, X_test, y_test, scoring):
    """Importancia por permutación sobre las variables ORIGINALES (no las dummies).
    Indica asociación con la predicción, no causalidad."""
    r = permutation_importance(pipe, X_test, y_test, scoring=scoring, n_repeats=5, random_state=42)
    return (pd.DataFrame({"variable": X_test.columns, "importancia": r.importances_mean,
                          "desv": r.importances_std})
            .sort_values("importancia", ascending=False).reset_index(drop=True))
