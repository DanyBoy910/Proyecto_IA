"""
Punto de entrada principal del proyecto de predicción de calidad del aire
Permite elegir entre entrenar el modelo o hacer predicciones
"""

import sys

import torch

import config
from data import load_and_preprocess_data
from models import (
    load_model_for_inference,
    predict,
    save_model,
    train_final_model,
    train_with_cross_validation,
)
from utils import create_normalized_cv_data, create_time_series_folds, print_cv_results


def train_pipeline():
    """Pipeline completo de entrenamiento."""
    print("\n" + "=" * 80)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO")
    print("=" * 80 + "\n")

    # 1. Cargar y preprocesar datos
    print("1. Cargando y preprocesando datos...")
    df_filtered, feature_cols, target_cols = load_and_preprocess_data(config.DATASET_PATH)
    print(f"   ✓ Datos cargados: {len(df_filtered)} filas")
    print(f"   ✓ Features: {len(feature_cols)} columnas")
    print(f"   ✓ Targets: {target_cols}")

    # 2. Crear folds de validación cruzada
    print("\n2. Creando folds de validación cruzada temporal...")
    cv_folds = create_time_series_folds(df_filtered)
    print(f"   ✓ Creados {len(cv_folds)} folds con gap de {config.GAP_HOURS} horas")

    # 3. Normalizar datos
    print("\n3. Normalizando datos de CV...")
    cv_data_normalized = create_normalized_cv_data(cv_folds, feature_cols, target_cols)
    print(f"   ✓ {len(cv_data_normalized)} folds normalizados")

    # 4. Entrenar con validación cruzada
    print("\n4. Entrenando modelos con validación cruzada...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Dispositivo: {device}")
    cv_results = train_with_cross_validation(
        cv_data_normalized, device=device, verbose=True
    )

    # 5. Mostrar resultados de CV
    print_cv_results(cv_results)

    # # 6. Entrenar modelo final
    # final_model, scaler_X_final, scaler_y_final = train_final_model(
    #     cv_folds, feature_cols, target_cols, device=device, verbose=True
    # )

    # # 7. Guardar modelo
    # save_model(final_model, scaler_X_final, scaler_y_final, feature_cols, config.VAR_COLS)

    print("\n✓ Pipeline de entrenamiento completado exitosamente\n")


def inference_pipeline():
    """Pipeline de inferencia con datos de ejemplo."""
    print("\n" + "=" * 80)
    print("INICIANDO PIPELINE DE INFERENCIA")
    print("=" * 80 + "\n")

    # Cargar modelo
    print("Cargando modelo entrenado...")
    model, scaler_X, scaler_y, config = load_model_for_inference()
    print("✓ Modelo cargado exitosamente\n")

    # Datos de ejemplo
    input_data = {
        # Variables actuales
        "CO": 0.9,
        "NO2": 0.017,
        "NO": 0.001,
        "NxOy": 0.018,
        "O3": 0.0,
        "P2.5": 50.0,
        "VV": 129.0,
        "HR": 84.0,
        "PB": 762.31,
        "RS": 1480.0,
        "PP": 0.0,
        # Temporales
        "Día_numérico": 13,
        "Mes": 6,
        "Hora_numérica": 15,
        # Rezagos CO
        "CO_lag1": 0.82,
        "CO_lag2": 0.8,
        "CO_lag3": 0.79,
        "CO_lag4": 0.8,
        "CO_lag5": 0.84,
        # Rezagos NO2
        "NO2_lag1": 0.005,
        "NO2_lag2": 0.005,
        "NO2_lag3": 0.005,
        "NO2_lag4": 0.007,
        "NO2_lag5": 0.011,
        # Rezagos NO
        "NO_lag1": 0.0,
        "NO_lag2": 0.001,
        "NO_lag3": 0.001,
        "NO_lag4": 0.001,
        "NO_lag5": 0.002,
        # Rezagos NxOy
        "NxOy_lag1": 0.005,
        "NxOy_lag2": 0.005,
        "NxOy_lag3": 0.005,
        "NxOy_lag4": 0.007,
        "NxOy_lag5": 0.013,
        # Rezagos O3
        "O3_lag1": 0.0,
        "O3_lag2": 0.07,
        "O3_lag3": 0.055,
        "O3_lag4": 0.056,
        "O3_lag5": 0.047,
        # Rezagos P2.5
        "P2.5_lag1": 25.0,
        "P2.5_lag2": 17.0,
        "P2.5_lag3": 7.0,
        "P2.5_lag4": 3.0,
        "P2.5_lag5": 0.0,
        # Rezagos VV
        "VV_lag1": 175.0,
        "VV_lag2": 192.0,
        "VV_lag3": 178.0,
        "VV_lag4": 146.0,
        "VV_lag5": 114.0,
        # Rezagos HR
        "HR_lag1": 66.0,
        "HR_lag2": 62.0,
        "HR_lag3": 48.0,
        "HR_lag4": 56.0,
        "HR_lag5": 60.0,
        # Rezagos PB
        "PB_lag1": 762.259,
        "PB_lag2": 762.112,
        "PB_lag3": 762.297,
        "PB_lag4": 762.584,
        "PB_lag5": 762.812,
        # Rezagos RS
        "RS_lag1": 1365.0,
        "RS_lag2": 1123.0,
        "RS_lag3": 674.0,
        "RS_lag4": 603.0,
        "RS_lag5": 816.0,
        # Rezagos PP
        "PP_lag1": 0.0,
        "PP_lag2": 0.0,
        "PP_lag3": 0.0,
        "PP_lag4": 0.0,
        "PP_lag5": 0.0,
    }

    # Hacer predicción
    predictions = predict(input_data, model, scaler_X, scaler_y, config)

    # Mostrar resultados
    print("=" * 80)
    print("PREDICCIONES PARA LA PRÓXIMA HORA")
    print("=" * 80)
    if config.TARGET_VARS[0] == "P2.5":
        print(f"  P2.5 (PM2.5): {predictions['P2.5']:>8.2f} µg/m³")
    elif config.TARGET_VARS[0] == "O3":
        print(f"  O3  (Ozono): {predictions['O3']:>8.4f} ppm")
    elif config.TARGET_VARS[0] == "CO":
        print(f"  CO (Monóxido de Carbono): {predictions['CO']:>8.4f} ppm")
    print("=" * 80 + "\n")


def main():
    """Función principal con menú de opciones."""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        config.TARGET_VARS[0] = sys.argv[2] if len(sys.argv) > 2 else "P2.5"
        config.PATIENCE = int(sys.argv[3].split('=')[1]) if len(sys.argv) > 3 else 50
        config.WEIGHT_DECAY = float(sys.argv[4].split('=')[1]) if len(sys.argv) > 4 else 0.01
        config.LEARNING_RATE = float(sys.argv[5].split('=')[1]) if len(sys.argv) > 5 else 0.002
        config.BATCH_SIZE = int(sys.argv[6].split('=')[1]) if len(sys.argv) > 6 else 64
        config.DROPOUT = float(sys.argv[7].split('=')[1]) if len(sys.argv) > 7 else 0.2
        config.HIDDEN_SIZES = [int(x) for x in sys.argv[8].split('=')[1].split("_")] if len(sys.argv) > 8 else [128, 64, 32]
    else:
        print("\n" + "=" * 80)
        print("SISTEMA DE PREDICCIÓN DE CALIDAD DEL AIRE")
        print("=" * 80)
        print("\nOpciones:")
        print("  1. Entrenar modelo")
        print("  2. Hacer predicción (inferencia)")
        print("  3. Salir")
        print("=" * 80)

        choice = input("\nSeleccione una opción (1-3): ").strip()

        if choice == "1":
            mode = "train"
        elif choice == "2":
            mode = "predict"
        elif choice == "3":
            print("\n¡Hasta luego!\n")
            return
        else:
            print("\n✗ Opción inválida\n")
            return

    if mode in ["train", "entrenar", "entrenamiento"]:
        train_pipeline()
    elif mode in ["predict", "inference", "prediccion", "predicción", "inferencia"]:
        inference_pipeline()
    else:
        print("\n✗ Modo inválido. Use: 'train' o 'predict'\n")
        print("Ejemplos:")
        print("  python main.py train")
        print("  python main.py predict")
        print()


if __name__ == "__main__":
    main()
