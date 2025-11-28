# Modelo MLP para Predicción de Contaminantes del Aire

Modelo de red neuronal (MLP) para predecir concentraciones de contaminantes del aire en la próxima hora.

## Variables Predichas

- **P2.5**: Material Particulado 2.5 (µg/m³)
- **O3**: Ozono (ppm)
- **CO**: Monóxido de Carbono (ppm)

## Arquitectura

- **Input**: 69 features
  - 11 variables actuales
  - 55 rezagos (5 horas × 11 variables)
  - 3 features temporales (Hora, Día, Mes)
- **Hidden layers**: [128, 64, 32] con BatchNorm, ReLU y Dropout (0.2)
- **Output**: 3 contaminantes
- **Regularización**: L2 (weight_decay = 0.01)
- **Normalización**: StandardScaler

## Rendimiento (Cross-Validation)

| Contaminante | MAE        | RMSE       | R²       |
| ------------ | ---------- | ---------- | -------- |
| P2.5         | ~4.6 µg/m³ | ~6.1 µg/m³ | 0.34     |
| O3           | ~0.003 ppm | ~0.004 ppm | 0.75     |
| CO           | ~0.10 ppm  | ~0.13 ppm  | 0.03     |
| **Promedio** | ~1.6       | ~2.1       | **0.37** |

## Archivos

### Scripts de Entrenamiento

- `model_training.py`: Script completo para entrenar el modelo
  - Procesa datos desde `dataset.json`
  - Entrena con Time Series Cross-Validation (5 folds)
  - Guarda modelo y scalers

### Scripts de Inferencia

- `model_inference.py`: Script para hacer predicciones
  - Carga modelo entrenado
  - Función `predict()` para nuevas predicciones

### Archivos Generados (después del entrenamiento)

- `air_quality_model.pth`: Pesos del modelo entrenado
- `scaler_X.pkl`: StandardScaler para features
- `scaler_y.pkl`: StandardScaler para targets
- `model_config.pkl`: Configuración del modelo

## Uso

### 1. Entrenamiento

```python
python model_training.py
```

Esto:

1. Carga y procesa `dataset.json`
2. Entrena con Cross-Validation
3. Entrena modelo final con todos los datos
4. Guarda modelo y scalers

### 2. Predicción

```python
from model_inference import predict

# Preparar input con 69 features
input_data = {
    # Variables actuales (11)
    'CO': 1.5, 'NO2': 0.005, 'NO': 0.001, 'NxOy': 0.006,
    'O3': 0.012, 'P2.5': 15.0, 'VV': 250.0, 'HR': 70.0,
    'PB': 760.0, 'RS': 1200.0, 'PP': 0.0,

    # Rezagos (55): lag1 a lag5 para cada variable
    'CO_lag1': 1.6, 'CO_lag2': 1.7, ...

    # Temporales (3)
    'Hora_numérica': 14, 'Día_numérico': 15, 'Mes': 11
}

predictions = predict(input_data)
print(predictions)
# {'P2.5': 16.5, 'O3': 0.013, 'CO': 1.55}
```

## Requisitos

```
pandas
numpy
torch
scikit-learn
```

## Notas Importantes

1. **Normalización es CRÍTICA**: Sin StandardScaler el modelo no funciona
2. **Orden de features**: Debe respetarse el orden exacto (ver `model_config.pkl`)
3. **Rezagos**: Se requieren las últimas 5 horas de datos
4. **Time Series CV**: Se usó con GAP de 24 horas para evitar data leakage
5. **Datos de entrenamiento**: Jun 2024 - Feb 2025 (9 meses)

## Limitaciones

- Faltan datos de Mar, Abr, May en el entrenamiento
- Menor precisión esperada en esos meses
- Reentrenar cuando se tengan datos completos del año
