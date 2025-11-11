"""Diagnóstico del problema de predicción"""

import pickle

import numpy as np

# Cargar scalers
with open("output/scaler_y.pkl", "rb") as f:
    scaler_y = pickle.load(f)

print("Información del scaler_y:")
print(f"Mean: {scaler_y.mean_}")
print(f"Scale: {scaler_y.scale_}")
print(f"Var: {scaler_y.var_}")

# Probar transformación inversa
test_scaled = np.array([[0.0, 0.0, 0.0]])
test_original = scaler_y.inverse_transform(test_scaled)
print(f"\nPrueba: [0, 0, 0] normalizado -> {test_original[0]}")

test_scaled2 = np.array([[1.0, 1.0, 1.0]])
test_original2 = scaler_y.inverse_transform(test_scaled2)
print(f"Prueba: [1, 1, 1] normalizado -> {test_original2[0]}")

# Cargar configuración
with open("output/model_config.pkl", "rb") as f:
    config = pickle.load(f)

print(f"\nTarget vars en config: {config['target_vars']}")

# Verificar un dato real
from data import load_and_preprocess_data

df, fc, tc = load_and_preprocess_data("dataset.json")

print(f"\nTarget cols: {tc}")
print("\nPrimeros 5 valores reales de targets:")
print(df[tc].head())
print("\nEstadísticas de targets:")
print(df[tc].describe())
