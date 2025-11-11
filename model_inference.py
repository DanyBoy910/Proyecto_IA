"""
Script de inferencia para predicción de contaminantes del aire
Carga el modelo entrenado y hace predicciones
"""

import pickle

import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# CARGAR MODELO Y CONFIGURACIÓN
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


# Cargar configuración
with open("model_config.pkl", "rb") as f:
    config = pickle.load(f)

# Cargar scalers
with open("scaler_X.pkl", "rb") as f:
    scaler_X = pickle.load(f)

with open("scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AirQualityMLP(
    input_size=config["input_size"],
    hidden_sizes=config["hidden_sizes"],
    output_size=config["output_size"],
    dropout=config["dropout"],
)
model.load_state_dict(torch.load("air_quality_model.pth", map_location=device))
model.to(device)
model.eval()

# ============================================================================
# FUNCIÓN DE PREDICCIÓN
# ============================================================================


def predict(input_data):
    """
    Predice los contaminantes para la próxima hora.

    Args:
        input_data: DataFrame o dict con las features necesarias (69 columnas)
                   Debe incluir:
                   - 11 variables actuales (CO, NO2, NO, NxOy, O3, P2.5, VV, HR, PB, RS, PP)
                   - 55 rezagos (cada variable con lag1 a lag5)
                   - 3 temporales (Hora_numérica, Día_numérico, Mes)

    Returns:
        dict con predicciones: {'P2.5': float, 'O3': float, 'CO': float}
    """
    # Convertir a DataFrame si es dict
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])

    # Verificar que tenga todas las columnas
    if not all(col in input_data.columns for col in config["feature_cols"]):
        raise ValueError(f"Faltan columnas. Se requieren: {config['feature_cols']}")

    # Seleccionar solo las columnas necesarias en el orden correcto
    X = input_data[config["feature_cols"]]

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
    predictions = {
        "P2.5": float(y_pred[0, 0]),
        "O3": float(y_pred[0, 1]),
        "CO": float(y_pred[0, 2]),
    }

    return predictions


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo: crear datos de prueba
    # En producción, estos vendrían de tu sistema de monitoreo

    input = {
        # Variables actuales
        "CO": 0.9,
        "NO2": 0.017,
        "NO": 0.001,
        "NxOy": 0.018000000000000002,
        "O3": 0.0,
        "P2.5": 50.0,
        "VV": 129.0,
        "HR": 84.0,
        "PB": 762.31,
        "RS": 1480.0,
        "PP": 0.0,
        "Día_numérico": 13,
        "Mes": 6,
        "Hora_numérica": 15,
        "CO_lag1": 0.8200000000000001,
        "CO_lag2": 0.8,
        "CO_lag3": 0.79,
        "CO_lag4": 0.8,
        "CO_lag5": 0.84,
        "NO2_lag1": 0.005,
        "NO2_lag2": 0.005,
        "NO2_lag3": 0.005,
        "NO2_lag4": 0.007,
        "NO2_lag5": 0.011,
        "NO_lag1": 0.0,
        "NO_lag2": 0.001,
        "NO_lag3": 0.001,
        "NO_lag4": 0.001,
        "NO_lag5": 0.002,
        "NxOy_lag1": 0.005,
        "NxOy_lag2": 0.005,
        "NxOy_lag3": 0.005,
        "NxOy_lag4": 0.007,
        "NxOy_lag5": 0.013000000000000001,
        "O3_lag1": 0.0,
        "O3_lag2": 0.07,
        "O3_lag3": 0.055,
        "O3_lag4": 0.056,
        "O3_lag5": 0.047,
        "P2.5_lag1": 25.0,
        "P2.5_lag2": 17.0,
        "P2.5_lag3": 7.0,
        "P2.5_lag4": 3.0,
        "P2.5_lag5": 0.0,
        "VV_lag1": 175.0,
        "VV_lag2": 192.0,
        "VV_lag3": 178.0,
        "VV_lag4": 146.0,
        "VV_lag5": 114.0,
        "HR_lag1": 66.0,
        "HR_lag2": 62.0,
        "HR_lag3": 48.0,
        "HR_lag4": 56.0,
        "HR_lag5": 60.0,
        "PB_lag1": 762.259,
        "PB_lag2": 762.112,
        "PB_lag3": 762.297,
        "PB_lag4": 762.584,
        "PB_lag5": 762.812,
        "RS_lag1": 1365.0,
        "RS_lag2": 1123.0,
        "RS_lag3": 674.0,
        "RS_lag4": 603.0,
        "RS_lag5": 816.0,
        "PP_lag1": 0.0,
        "PP_lag2": 0.0,
        "PP_lag3": 0.0,
        "PP_lag4": 0.0,
        "PP_lag5": 0.0,
    }

    ground_truth = {
        "P2.5": 50.0,
        "O3": 0.0,
        "CO": 0.87,
    }

    # Nota: El ejemplo anterior está incompleto, necesitas todas las 69 features
    # En producción, construyes esto desde tus datos históricos

    predictions = predict(input)
    print("Predicciones para la próxima hora:")
    print(f"  P2.5: {predictions['P2.5']:.2f} µg/m³")
    print(f"  O3:   {predictions['O3']:.4f} ppm")
    print(f"  CO:   {predictions['CO']:.4f} ppm")
