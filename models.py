"""
Modelo de red neuronal, entrenamiento e inferencia
"""

import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import config
from utils import compute_metrics

# ============================================================================
# DEFINICIÓN DEL MODELO
# ============================================================================


class AirQualityMLP(nn.Module):
    """Red neuronal MLP para predicción de contaminantes del aire."""

    def __init__(
        self,
        input_size=config.INPUT_SIZE,
        hidden_sizes=None,
        output_size=config.OUTPUT_SIZE,
        dropout=None,
    ):
        super(AirQualityMLP, self).__init__()

        if dropout is None:
            dropout = config.DROPOUT
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES

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
# DATASET
# ============================================================================


class AirQualityDataset(Dataset):
    """Dataset de PyTorch para datos de calidad del aire."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================================
# FUNCIONES DE ENTRENAMIENTO
# ============================================================================


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Entrena el modelo por una época."""
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
    """Valida el modelo en el conjunto de validación."""
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
    epochs=config.EPOCHS,
    lr=None,
    weight_decay=None,
    device="cpu",
    patience=None,
    verbose=False,
):
    """
    Entrena el modelo con early stopping.

    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        epochs: Número máximo de épocas
        lr: Learning rate
        weight_decay: Regularización L2
        device: Dispositivo (cpu/cuda)
        patience: Épocas sin mejora antes de parar
        verbose: Si imprimir progreso

    Returns:
        Modelo entrenado
    """
    if lr is None:
        lr = config.LEARNING_RATE
    if weight_decay is None:
        weight_decay = config.WEIGHT_DECAY
    if patience is None:
        patience = config.PATIENCE

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        with open('resultados.txt', 'a', encoding='utf-8') as fl:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = validate_epoch(
                model, val_loader, criterion, device
            )

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
                )
                fl.write(
                    f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}\n"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter > patience:
                if verbose:
                    print(f"Early stopping en época {epoch + 1}")
                    fl.write(f"Early stopping en época {epoch + 1}\n")
                break

    model.load_state_dict(best_model_state)
    return model


def train_with_cross_validation(cv_data_normalized, device="cpu", verbose=True):
    """
    Entrena modelos usando validación cruzada.

    Args:
        cv_data_normalized: Datos normalizados de CV
        device: Dispositivo (cpu/cuda)
        verbose: Si imprimir progreso

    Returns:
        Lista de resultados por fold
    """
    cv_results = []

    for fold_data in cv_data_normalized:
        if verbose:
            with open('resultados.txt', 'a', encoding='utf-8') as fl:
                print(f"\n{'=' * 80}")
                print(f"Entrenando Fold {fold_data['fold']}")
                print(f"{'=' * 80}")
                fl.write(f"\n{'=' * 80}\n")
                fl.write(f"Entrenando Fold {fold_data['fold']}\n")

        train_dataset = AirQualityDataset(fold_data["X_train"], fold_data["y_train"])
        val_dataset = AirQualityDataset(fold_data["X_val"], fold_data["y_val"])
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
        )

        model = AirQualityMLP()
        model = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            verbose=verbose,
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
        all_targets_original = fold_data["scaler_y"].inverse_transform(
            all_targets_scaled
        )

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

    return cv_results


def train_final_model(cv_folds, feature_cols, target_cols, device="cpu", verbose=True):
    """
    Entrena el modelo final con todos los datos de CV.

    Args:
        cv_folds: Folds de validación cruzada con datos ORIGINALES (no normalizados)
        feature_cols: Columnas de características
        target_cols: Columnas objetivo
        device: Dispositivo (cpu/cuda)
        verbose: Si imprimir progreso

    Returns:
        Tupla (modelo, scaler_X, scaler_y)
    """
    from sklearn.preprocessing import StandardScaler

    if verbose:
        print("\n" + "=" * 80)
        print("ENTRENANDO MODELO FINAL CON TODOS LOS DATOS")
        print("=" * 80)

    # Combinar todos los datos ORIGINALES de train+val
    all_train_data = []
    all_train_labels = []

    for fold_data in cv_folds:
        all_train_data.append(fold_data["train"][feature_cols])
        all_train_labels.append(fold_data["train"][target_cols])
        all_train_data.append(fold_data["val"][feature_cols])
        all_train_labels.append(fold_data["val"][target_cols])

    X_final = pd.concat(all_train_data, ignore_index=True)
    y_final = pd.concat(all_train_labels, ignore_index=True)

    # Scalers finales (con TODOS los datos ORIGINALES)
    scaler_X_final = StandardScaler()
    scaler_y_final = StandardScaler()

    X_final_scaled = scaler_X_final.fit_transform(X_final)
    y_final_scaled = scaler_y_final.fit_transform(y_final)

    X_final_scaled = pd.DataFrame(X_final_scaled, columns=feature_cols)
    y_final_scaled = pd.DataFrame(y_final_scaled, columns=target_cols)

    # Entrenar modelo final
    final_dataset = AirQualityDataset(X_final_scaled, y_final_scaled)
    final_loader = DataLoader(
        final_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )

    # Crear un loader para validación (mismo que train, no se usa realmente)
    dummy_loader = DataLoader(
        final_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    final_model = AirQualityMLP()
    final_model = train_model(
        model=final_model,
        train_loader=final_loader,
        val_loader=dummy_loader,
        device=device,
        verbose=verbose,
    )

    return final_model, scaler_X_final, scaler_y_final


def save_model(model, scaler_X, scaler_y, feature_cols, var_cols):
    """
    Guarda el modelo, scalers y configuración.

    Args:
        model: Modelo entrenado
        scaler_X: Scaler de características
        scaler_y: Scaler de targets
        feature_cols: Columnas de características
        var_cols: Columnas de variables
    """
    # Guardar modelo
    torch.save(model.state_dict(), config.MODEL_PATH)

    # Guardar scalers
    with open(config.SCALER_X_PATH, "wb") as f:
        pickle.dump(scaler_X, f)

    with open(config.SCALER_Y_PATH, "wb") as f:
        pickle.dump(scaler_y, f)

    # Guardar configuración
    config = {
        "input_size": config.INPUT_SIZE,
        "hidden_sizes": config.HIDDEN_SIZES,
        "output_size": config.OUTPUT_SIZE,
        "dropout": config.DROPOUT,
        "feature_cols": feature_cols,
        "target_vars": config.TARGET_VARS,
        "var_cols": var_cols,
    }

    with open(config.CONFIG_PATH, "wb") as f:
        pickle.dump(config, f)

    print("\n" + "=" * 80)
    print("MODELO Y CONFIGURACIÓN GUARDADOS EXITOSAMENTE")
    print("=" * 80)
    print(f"- {config.MODEL_PATH}")
    print(f"- {config.SCALER_X_PATH}")
    print(f"- {config.SCALER_Y_PATH}")
    print(f"- {config.CONFIG_PATH}")
    print("=" * 80 + "\n")


# ============================================================================
# INFERENCIA
# ============================================================================


def load_model_for_inference():
    """
    Carga el modelo, scalers y configuración para inferencia.

    Returns:
        Tupla (modelo, scaler_X, scaler_y, config)
    """
    # Cargar configuración
    with open(config.CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    # Cargar scalers
    with open(config.SCALER_X_PATH, "rb") as f:
        scaler_X = pickle.load(f)

    with open(config.SCALER_Y_PATH, "rb") as f:
        scaler_y = pickle.load(f)

    # Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AirQualityMLP(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        output_size=config["output_size"],
        dropout=config["dropout"],
    )
    model.load_state_dict(
        torch.load(config.MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, scaler_X, scaler_y, config


def predict(input_data, model=None, scaler_X=None, scaler_y=None, configu=None):
    """
    Predice los contaminantes para la próxima hora.

    Args:
        input_data: DataFrame o dict con las features necesarias (69 columnas)
                   Debe incluir:
                   - 11 variables actuales (CO, NO2, NO, NxOy, O3, P2.5, VV, HR, PB, RS, PP)
                   - 55 rezagos (cada variable con lag1 a lag5)
                   - 3 temporales (Hora_numérica, Día_numérico, Mes)
        model: Modelo cargado (si None, se carga automáticamente)
        scaler_X: Scaler de X (si None, se carga automáticamente)
        scaler_y: Scaler de y (si None, se carga automáticamente)
        configu: Configuración (si None, se carga automáticamente)

    Returns:
        dict con predicciones: {'P2.5': float, 'O3': float, 'CO': float}
    """
    # Cargar modelo si no se proporcionó
    if model is None or scaler_X is None or scaler_y is None or configu is None:
        model, scaler_X, scaler_y, configu = load_model_for_inference()

    device = next(model.parameters()).device

    # Convertir a DataFrame si es dict
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Verificar que tenga todas las columnas
    if not all(col in input_data.columns for col in configu["feature_cols"]):
        raise ValueError(f"Faltan columnas. Se requieren: {configu['feature_cols']}")

    # Seleccionar solo las columnas necesarias en el orden correcto
    X = input_data[configu["feature_cols"]]

    # Normalizar
    X_scaled = scaler_X.transform(X)

    # Convertir a tensor
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    # Predicción
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    # Desnormalizar
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Retornar como diccionario
    if config.TARGET_VARS[0] == "P2.5":
        predictions = {
            "P2.5": float(y_pred[0, 0]),
        }
    elif config.TARGET_VARS[0] == "O3":
        predictions = {
            "O3": float(y_pred[0, 0]),
        }
    else:
        predictions = {
            "CO": float(y_pred[0, 0]),
        }
    # predictions = {
    #     "P2.5": float(y_pred[0, 0]),
    #     # "O3": float(y_pred[0, 1]),
    #     # "CO": float(y_pred[0, 2]),
    # }

    return predictions
