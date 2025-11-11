"""
Entrenamiento de Modelo MLP para Predicción de Contaminantes del Aire
Variables a predecir: P2.5, O3, CO (próxima hora t+1)
"""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# 1. CARGA Y PROCESAMIENTO DE DATOS
# ============================================================================

# Cargar datos
dataset = pd.read_json("dataset.json")
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

# Limpieza: eliminar T y corregir valores
df = df.drop(columns=["T"])

# Reemplazar valores negativos por 0
non_negative_variables = ["CO", "NO2", "NO", "NxOy", "O3", "P2.5", "PP"]
for var in non_negative_variables:
    if var in df.columns:
        df[var] = df[var].apply(lambda x: 0 if pd.notna(x) and x < 0 else x)

# Corregir outliers en VV
df.loc[df["VV"] > 500, "VV"] = None

# Features temporales
df["Día_numérico"] = pd.to_datetime(df["Día"]).dt.day
df["Mes"] = pd.to_datetime(df["Día"]).dt.month
df["Hora_numérica"] = df["Hora"].str.split(":").str[0].astype(int)

# Variables a usar
var_cols = ["CO", "NO2", "NO", "NxOy", "O3", "P2.5", "VV", "HR", "PB", "RS", "PP"]


# Crear rezagos y targets
def create_lagged_features(df, lag_hours=5):
    df_lagged = df.copy()

    for var in var_cols:
        for lag in range(1, lag_hours + 1):
            df_lagged[f"{var}_lag{lag}"] = df[var].shift(lag)

    for var in var_cols:
        df_lagged[f"{var}_next"] = df[var].shift(-1)

    df_lagged = df_lagged.dropna()
    return df_lagged


df_lagged = create_lagged_features(df, lag_hours=5)

# Filtrar desde Junio 2024
df_filtered = df_lagged[pd.to_datetime(df_lagged["Día"]) >= "2024-06-01"].reset_index(
    drop=True
)

# Features y targets
feature_cols = (
    var_cols
    + [f"{var}_lag{i}" for var in var_cols for i in range(1, 6)]
    + ["Hora_numérica", "Día_numérico", "Mes"]
)
target_vars = ["P2.5", "O3", "CO"]
target_cols = [f"{var}_next" for var in target_vars]

# ============================================================================
# 2. TIME SERIES CROSS-VALIDATION SPLITS
# ============================================================================

GAP_HOURS = 24
N_FOLDS = 5
MIN_TRAIN_SIZE = 2000


def create_time_series_folds(df, n_folds=5, gap=24, min_train_size=2000):
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


cv_folds = create_time_series_folds(
    df_filtered, n_folds=N_FOLDS, gap=GAP_HOURS, min_train_size=MIN_TRAIN_SIZE
)

# Test final
test_start_idx = len(df_filtered) - int(0.15 * len(df_filtered))
test_gap_start = test_start_idx - GAP_HOURS
test_final = df_filtered.iloc[test_start_idx:]
X_test_final = test_final[feature_cols]
y_test_final = test_final[target_cols]

# ============================================================================
# 3. NORMALIZACIÓN
# ============================================================================


def create_normalized_cv_data(cv_folds, feature_cols, target_cols):
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


cv_data_normalized = create_normalized_cv_data(cv_folds, feature_cols, target_cols)

# ============================================================================
# 4. MODELO MLP
# ============================================================================


class AirQualityMLP(nn.Module):
    def __init__(
        self, input_size=69, hidden_sizes=[128, 64, 32], output_size=3, dropout=0.2
    ):
        super(AirQualityMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================================
# 5. DATASET Y ENTRENAMIENTO
# ============================================================================


class AirQualityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def compute_metrics(y_true, y_pred, var_names=["P2.5", "O3", "CO"]):
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


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            total_loss += loss.item()
            all_preds.append(y_pred.cpu())
            all_targets.append(y_batch.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    return total_loss / len(val_loader), all_preds, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=0.002,
    weight_decay=0.01,
    device="cpu",
    patience=15,
):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_preds, val_targets = validate_epoch(
            model, val_loader, criterion, device
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model


# ============================================================================
# 6. ENTRENAMIENTO EN CROSS-VALIDATION
# ============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 15

# Hiperparámetros óptimos
HIDDEN_SIZES = [128, 64, 32]
DROPOUT = 0.2
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01

cv_results = []

for fold_data in cv_data_normalized:
    train_dataset = AirQualityDataset(fold_data["X_train"], fold_data["y_train"])
    val_dataset = AirQualityDataset(fold_data["X_val"], fold_data["y_val"])
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = AirQualityMLP(
        input_size=69, hidden_sizes=HIDDEN_SIZES, output_size=3, dropout=DROPOUT
    )

    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=device,
        patience=PATIENCE,
    )

    # Evaluar
    model.eval()
    all_preds_scaled = []
    all_targets_scaled = []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            all_preds_scaled.append(y_pred.cpu())
            all_targets_scaled.append(y_batch)

    all_preds_scaled = torch.cat(all_preds_scaled, dim=0).numpy()
    all_targets_scaled = torch.cat(all_targets_scaled, dim=0).numpy()

    all_preds_original = fold_data["scaler_y"].inverse_transform(all_preds_scaled)
    all_targets_original = fold_data["scaler_y"].inverse_transform(all_targets_scaled)

    metrics = compute_metrics(all_targets_original, all_preds_original)

    cv_results.append(
        {
            "fold": fold_data["fold"],
            "metrics": metrics,
            "model": model,
            "scaler_X": fold_data["scaler_X"],
            "scaler_y": fold_data["scaler_y"],
        }
    )

# ============================================================================
# 7. MODELO FINAL (ENTRENADO CON TODOS LOS DATOS)
# ============================================================================

# Combinar todos los datos de train+val para modelo final
all_train_data = []
all_train_labels = []

for fold_data in cv_data_normalized:
    all_train_data.append(fold_data["X_train"])
    all_train_labels.append(fold_data["y_train"])
    all_train_data.append(fold_data["X_val"])
    all_train_labels.append(fold_data["y_val"])

X_final = pd.concat(all_train_data, ignore_index=True)
y_final = pd.concat(all_train_labels, ignore_index=True)

# Scalers finales (con TODOS los datos)
scaler_X_final = StandardScaler()
scaler_y_final = StandardScaler()

X_final_scaled = scaler_X_final.fit_transform(X_final)
y_final_scaled = scaler_y_final.fit_transform(y_final)

X_final_scaled = pd.DataFrame(X_final_scaled, columns=feature_cols)
y_final_scaled = pd.DataFrame(y_final_scaled, columns=target_cols)

# Entrenar modelo final
final_dataset = AirQualityDataset(X_final_scaled, y_final_scaled)
final_loader = DataLoader(
    final_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

# Crear un loader vacío para validación (no se usa pero la función lo requiere)
dummy_loader = DataLoader(
    final_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

final_model = AirQualityMLP(
    input_size=69, hidden_sizes=HIDDEN_SIZES, output_size=3, dropout=DROPOUT
)

final_model = train_model(
    model=final_model,
    train_loader=final_loader,
    val_loader=dummy_loader,
    epochs=EPOCHS,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    device=device,
    patience=PATIENCE,
)

# ============================================================================
# 8. GUARDAR MODELO Y SCALERS
# ============================================================================

import pickle

# Guardar modelo
torch.save(final_model.state_dict(), "air_quality_model.pth")

# Guardar scalers
with open("scaler_X.pkl", "wb") as f:
    pickle.dump(scaler_X_final, f)

with open("scaler_y.pkl", "wb") as f:
    pickle.dump(scaler_y_final, f)

# Guardar configuración
config = {
    "input_size": 69,
    "hidden_sizes": HIDDEN_SIZES,
    "output_size": 3,
    "dropout": DROPOUT,
    "feature_cols": feature_cols,
    "target_vars": target_vars,
    "var_cols": var_cols,
}

with open("model_config.pkl", "wb") as f:
    pickle.dump(config, f)

print("Modelo y configuración guardados exitosamente.")
print("- air_quality_model.pth")
print("- scaler_X.pkl")
print("- scaler_y.pkl")
print("- model_config.pkl")
