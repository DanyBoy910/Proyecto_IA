"""
Módulo para carga y procesamiento de datos de calidad del aire
"""

from typing import Any

import pandas as pd

from config import LAG_HOURS, NON_NEGATIVE_VARIABLES, START_DATE, VAR_COLS


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Carga el dataset JSON y lo convierte a DataFrame.

    Args:
        dataset_path: Ruta al archivo dataset.json

    Returns:
        DataFrame con columnas: Día, Hora, y todas las variables
    """
    dataset = pd.read_json(dataset_path)
    days: list[Any] = list(dataset["fecha"])
    hours: list[Any] = list(dataset.iloc[0]["horas"].keys())
    hours.sort(key=lambda x: int(x.split(":")[0]))
    variables: list[Any] = list(dataset.iloc[0]["horas"][hours[0]])

    # Crear DataFrame con todas las horas
    rows = []
    for day_index, date in enumerate(days):
        for hour in hours:
            row = [date, hour]
            for var in variables:
                value = dataset.iloc[day_index]["horas"][hour][var]
                row.append(float(value) if value != "" else None)
            rows.append(row)

    df = pd.DataFrame(rows, columns=["Día", "Hora"] + variables)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y corrige valores del DataFrame.

    Args:
        df: DataFrame con datos crudos

    Returns:
        DataFrame limpio
    """
    df = df.copy()

    # Eliminar columna T si existe
    if "T" in df.columns:
        df = df.drop(columns=["T"])

    # Reemplazar valores negativos por 0
    for var in NON_NEGATIVE_VARIABLES:
        if var in df.columns:
            df[var] = df[var].apply(lambda x: 0 if pd.notna(x) and x < 0 else x)

    # Corregir outliers en VV (velocidad del viento)
    if "VV" in df.columns:
        df.loc[df["VV"] > 500, "VV"] = None

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características temporales al DataFrame.

    Args:
        df: DataFrame con columnas Día y Hora

    Returns:
        DataFrame con características temporales agregadas
    """
    df = df.copy()
    df["Día_numérico"] = pd.to_datetime(df["Día"]).dt.day
    df["Mes"] = pd.to_datetime(df["Día"]).dt.month
    df["Hora_numérica"] = df["Hora"].str.split(":").str[0].astype(int)
    return df


def create_lagged_features(
    df: pd.DataFrame, lag_hours: int = LAG_HOURS
) -> pd.DataFrame:
    """
    Crea características de rezago (lag) y variables objetivo (next).

    Args:
        df: DataFrame con variables meteorológicas
        lag_hours: Número de horas de rezago a crear

    Returns:
        DataFrame con lags y targets (eliminando NaN)
    """
    df_lagged = df.copy()

    # Crear rezagos
    for var in VAR_COLS:
        for lag in range(1, lag_hours + 1):
            df_lagged[f"{var}_lag{lag}"] = df[var].shift(lag)

    # Crear targets (siguiente hora)
    for var in VAR_COLS:
        df_lagged[f"{var}_next"] = df[var].shift(-1)

    # Eliminar filas con NaN
    df_lagged = df_lagged.dropna()
    return df_lagged


def prepare_features_and_targets(df: pd.DataFrame, target_vars: list[str]):
    """
    Prepara columnas de features y targets.

    Args:
        df: DataFrame procesado con lags
        target_vars: Lista de variables a predecir

    Returns:
        Tupla (feature_cols, target_cols)
    """
    feature_cols = (
        VAR_COLS
        + [f"{var}_lag{i}" for var in VAR_COLS for i in range(1, LAG_HOURS + 1)]
        + ["Hora_numérica", "Día_numérico", "Mes"]
    )
    target_cols = [f"{var}_next" for var in target_vars]

    return feature_cols, target_cols


def load_and_preprocess_data(dataset_path: str, start_date: str = START_DATE):
    """
    Pipeline completo de carga y preprocesamiento de datos.

    Args:
        dataset_path: Ruta al archivo dataset.json
        start_date: Fecha de inicio para filtrar datos

    Returns:
        Tupla (df_filtered, feature_cols, target_cols)
    """
    # Cargar datos
    df = load_dataset(dataset_path)

    # Limpiar datos
    df = clean_data(df)

    # Agregar features temporales
    df = add_temporal_features(df)

    # Crear lags
    df_lagged = create_lagged_features(df)

    # Filtrar desde fecha especificada
    df_filtered = df_lagged[pd.to_datetime(df_lagged["Día"]) >= start_date].reset_index(
        drop=True
    )

    # Preparar columnas de features y targets
    from config import TARGET_VARS

    feature_cols, target_cols = prepare_features_and_targets(df_filtered, TARGET_VARS)

    return df_filtered, feature_cols, target_cols
