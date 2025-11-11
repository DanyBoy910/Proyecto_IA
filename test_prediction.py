"""Script temporal para probar predicción con datos reales"""

from data import load_and_preprocess_data
from models import load_model_for_inference, predict

# Cargar datos
df, feature_cols, target_cols = load_and_preprocess_data("dataset.json")

# Tomar una muestra del dataset (convertir a DataFrame)
sample = df.iloc[[-10]][feature_cols]  # Doble corchete para mantener DataFrame

# Cargar modelo
model, scaler_X, scaler_y, config = load_model_for_inference()

# Hacer predicción
predictions = predict(sample, model, scaler_X, scaler_y, config)

# Obtener valor real
actual = df.iloc[-10][target_cols]

print("=" * 80)
print("PREDICCIÓN CON DATOS REALES DEL DATASET")
print("=" * 80)
print("\nPredicciones:")
print(f"  P2.5: {predictions['P2.5']:>8.2f} µg/m³")
print(f"  O3:   {predictions['O3']:>8.4f} ppm")
print(f"  CO:   {predictions['CO']:>8.4f} ppm")

print("\nValores reales (siguiente hora):")
print(f"  P2.5: {actual['P2.5_next']:>8.2f} µg/m³")
print(f"  O3:   {actual['O3_next']:>8.4f} ppm")
print(f"  CO:   {actual['CO_next']:>8.4f} ppm")

print("\nError absoluto:")
print(f"  P2.5: {abs(predictions['P2.5'] - actual['P2.5_next']):>8.2f} µg/m³")
print(f"  O3:   {abs(predictions['O3'] - actual['O3_next']):>8.4f} ppm")
print(f"  CO:   {abs(predictions['CO'] - actual['CO_next']):>8.4f} ppm")
print("=" * 80)
