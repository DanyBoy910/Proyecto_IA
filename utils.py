"""
Funciones auxiliares: métricas, validación cruzada, etc.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from config import GAP_HOURS, MIN_TRAIN_SIZE, N_FOLDS, TARGET_VARS


def create_time_series_folds(
    df: pd.DataFrame,
    n_folds: int = N_FOLDS,
    gap: int = GAP_HOURS,
    min_train_size: int = MIN_TRAIN_SIZE,
):
    """
    Crea folds de validación cruzada temporal con gap.

    Args:
        df: DataFrame completo
        n_folds: Número de folds
        gap: Número de horas de gap entre train y val
        min_train_size: Tamaño mínimo del conjunto de entrenamiento

    Returns:
        Lista de dicts con train y val DataFrames
    """
    folds = []
    n_total = len(df)
    val_size = (n_total - min_train_size - (n_folds * gap)) // n_folds

    for fold_idx in range(n_folds):
        train_end = min_train_size + (fold_idx * val_size)
        gap_end = train_end + gap
        val_end = gap_end + val_size

        if val_end > n_total:
            break

        train_df = df.iloc[:train_end]
        val_df = df.iloc[gap_end:val_end]

        folds.append({"fold": fold_idx + 1, "train": train_df, "val": val_df})

    return folds


def create_normalized_cv_data(cv_folds, feature_cols, target_cols):
    """
    Normaliza los datos de cada fold de validación cruzada.

    Args:
        cv_folds: Lista de folds creados por create_time_series_folds
        feature_cols: Columnas de características
        target_cols: Columnas objetivo

    Returns:
        Lista de dicts con datos normalizados y scalers por fold
    """
    normalized_cv_data = []

    for fold_idx, fold_info in enumerate(cv_folds, 1):
        X_train = fold_info["train"][feature_cols]
        y_train = fold_info["train"][target_cols]
        X_val = fold_info["val"][feature_cols]
        y_val = fold_info["val"][target_cols]

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)

        X_train_scaled = pd.DataFrame(
            X_train_scaled, columns=feature_cols, index=X_train.index
        )
        y_train_scaled = pd.DataFrame(
            y_train_scaled, columns=target_cols, index=y_train.index
        )
        X_val_scaled = pd.DataFrame(
            X_val_scaled, columns=feature_cols, index=X_val.index
        )
        y_val_scaled = pd.DataFrame(
            y_val_scaled, columns=target_cols, index=y_val.index
        )

        normalized_cv_data.append(
            {
                "fold": fold_idx,
                "X_train": X_train_scaled,
                "y_train": y_train_scaled,
                "X_val": X_val_scaled,
                "y_val": y_val_scaled,
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
            }
        )

    return normalized_cv_data


# def compute_metrics(y_true, y_pred, var_names=["P2.5", "O3", "CO"]):
def compute_metrics(y_true, y_pred, var_names=None):
    """
    Calcula métricas de evaluación (MAE, RMSE, R²) para cada variable.

    Args:
        y_true: Valores verdaderos
        y_pred: Valores predichos
        var_names: Nombres de las variables

    Returns:
        Dict con métricas por variable y promedio
    """
    if var_names is None:
        var_names = [TARGET_VARS[0]]

    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    metrics = {}
    for i, var in enumerate(var_names):
        mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        metrics[var] = {"MAE": mae, "RMSE": rmse, "R2": r2}

    metrics["Average"] = {
        "MAE": np.mean([metrics[var]["MAE"] for var in var_names]),
        "RMSE": np.mean([metrics[var]["RMSE"] for var in var_names]),
        "R2": np.mean([metrics[var]["R2"] for var in var_names]),
    }
    return metrics


def print_cv_results(cv_results):
    """
    Imprime resultados de validación cruzada de forma legible.

    Args:
        cv_results: Lista de resultados por fold
    """
    with open('resultados.txt', 'a', encoding='utf-8') as fl:
        print("\n" + "=" * 80)
        print("RESULTADOS DE VALIDACIÓN CRUZADA")
        print("=" * 80)
        fl.write("\n" + "=" * 80 + "\n")
        fl.write("RESULTADOS DE VALIDACIÓN CRUZADA\n")

        for result in cv_results:
            fold = result["fold"]
            metrics = result["metrics"]

            print(f"\n--- Fold {fold} ---")
            fl.write(f"--- Fold {fold} ---\n")
            # for var in ["P2.5", "O3", "CO"]:
            for var in [TARGET_VARS[0]]:
                print(
                    f"{var:>6}: MAE={metrics[var]['MAE']:>8.4f}  "
                    f"RMSE={metrics[var]['RMSE']:>8.4f}  "
                    f"R²={metrics[var]['R2']:>7.4f}"
                )
                fl.write(
                    f"{var:>6}: MAE={metrics[var]['MAE']:>8.4f}  "
                    f"RMSE={metrics[var]['RMSE']:>8.4f}  "
                    f"R²={metrics[var]['R2']:>7.4f}\n"
                )

            # print(
            #     f"{'Promedio':>6}: MAE={metrics['Average']['MAE']:>8.4f}  "
            #     f"RMSE={metrics['Average']['RMSE']:>8.4f}  "
            #     f"R²={metrics['Average']['R2']:>7.4f}"
            # )

        # Calcular promedios generales
        avg_metrics = {
            "MAE": np.mean([r["metrics"]["Average"]["MAE"] for r in cv_results]),
            "RMSE": np.mean([r["metrics"]["Average"]["RMSE"] for r in cv_results]),
            "R2": np.mean([r["metrics"]["Average"]["R2"] for r in cv_results]),
        }

        print("\n" + "=" * 80)
        print("PROMEDIO GENERAL DE TODOS LOS FOLDS")
        print("=" * 80)
        print(f"MAE:  {avg_metrics['MAE']:.4f}")
        print(f"RMSE: {avg_metrics['RMSE']:.4f}")
        print(f"R²:   {avg_metrics['R2']:.4f}")
        print("=" * 80 + "\n")
        fl.write("\n" + "=" * 80 + "\n")
        fl.write("PROMEDIO GENERAL DE TODOS LOS FOLDS\n")
        fl.write(f"MAE:  {avg_metrics['MAE']:.4f}\n")
        fl.write(f"RMSE: {avg_metrics['RMSE']:.4f}\n")
        fl.write(f"R²:   {avg_metrics['R2']:.4f}\n")
        fl.write("=" * 80 + "\n\n")
