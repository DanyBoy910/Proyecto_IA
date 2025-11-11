"""
Configuración y constantes del proyecto de predicción de calidad del aire
"""

# ============================================================================
# VARIABLES Y CARACTERÍSTICAS
# ============================================================================

# Variables meteorológicas y de contaminantes
VAR_COLS = ["CO", "NO2", "NO", "NxOy", "O3", "P2.5", "VV", "HR", "PB", "RS", "PP"]

# Variables que no pueden ser negativas
NON_NEGATIVE_VARIABLES = ["CO", "NO2", "NO", "NxOy", "O3", "P2.5", "PP"]

# Variables objetivo (a predecir)
TARGET_VARS = ["P2.5", "O3", "CO"]

# Número de horas de rezago
LAG_HOURS = 5

# ============================================================================
# PARÁMETROS DE DATOS
# ============================================================================

# Fecha de inicio para filtrado de datos
START_DATE = "2024-06-01"

# Parámetros de validación cruzada temporal
GAP_HOURS = 24
N_FOLDS = 5
MIN_TRAIN_SIZE = 2000

# Porcentaje de datos para test final
TEST_SIZE = 0.15

# ============================================================================
# HIPERPARÁMETROS DEL MODELO
# ============================================================================

# Arquitectura del modelo
INPUT_SIZE = 69  # 11 vars + 55 lags (11*5) + 3 temporales
HIDDEN_SIZES = [128, 64, 32]
OUTPUT_SIZE = 3  # P2.5, O3, CO
DROPOUT = 0.2

# Parámetros de entrenamiento
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.01
PATIENCE = 50

# ============================================================================
# RUTAS DE ARCHIVOS
# ============================================================================

import os

# Directorio de salida
OUTPUT_DIR = "output"

# Asegurar que existe el directorio de salida
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rutas de archivos
DATASET_PATH = "dataset.json"
MODEL_PATH = os.path.join(OUTPUT_DIR, "air_quality_model.pth")
SCALER_X_PATH = os.path.join(OUTPUT_DIR, "scaler_X.pkl")
SCALER_Y_PATH = os.path.join(OUTPUT_DIR, "scaler_y.pkl")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "model_config.pkl")
