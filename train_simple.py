"""Entrenar solo el modelo final sin validación cruzada"""

import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from config import BATCH_SIZE, DATASET_PATH, VAR_COLS
from data import load_and_preprocess_data
from models import AirQualityDataset, AirQualityMLP, save_model, train_model

print("Cargando datos...")
df, feature_cols, target_cols = load_and_preprocess_data(DATASET_PATH)
print(f"Datos cargados: {len(df)} filas")

# Usar 85% para train
train_size = int(0.85 * len(df))
df_train = df.iloc[:train_size]

X_train = df_train[feature_cols]
y_train = df_train[target_cols]

# Scalers
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train)

print("\nEntrenando modelo...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")

import pandas as pd

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols)
y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=target_cols)

train_dataset = AirQualityDataset(X_train_scaled_df, y_train_scaled_df)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

model = AirQualityMLP()
model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=train_loader,  # Usar el mismo para validación
    device=device,
    verbose=True,
)

print("\nGuardando modelo...")
save_model(model, scaler_X, scaler_y, feature_cols, VAR_COLS)
print("✓ Completado")
