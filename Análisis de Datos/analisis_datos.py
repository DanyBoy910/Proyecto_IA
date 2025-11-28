import pandas as pd
import json
import os

# --- 1. Definir el archivo y las INSTRUCCIONES de procesamiento ---
archivo_excel = "Datos.xlsx"

# ¡IMPORTANTE! Aquí defines cómo se procesa cada hoja.
# tipo = "horario" -> Mapea usando la columna 'Hora' (ej. CO)
# tipo = "secuencial_variable" -> Mapea N filas a las últimas N horas (ej. PP)
hojas_a_procesar = [
    {"sheet_name": "Monóxido de Carbono (CO)", "var_name": "CO", "tipo": "horario"},
    {"sheet_name": "Dióxido de Nitrógeno (NO2)", "var_name": "NO2", "tipo": "horario"},
    {"sheet_name": "Óxido Nítrico (NO)", "var_name": "NO", "tipo": "secuencial_variable"},

   
    {"sheet_name": "Óxido de Nitrógeno (NxOy)", "var_name": "NxOy", "tipo": "secuencial_variable"},


    {"sheet_name": "Ozono (O3)", "var_name": "O3", "tipo": "horario"},
    {"sheet_name": "Partículas 2.5", "var_name": "P2.5", "tipo": "horario"},
    
    
    
    # trata de forma diferente a las hojas que no tienen hora
    {"sheet_name": "Velocidad del Viento (VV)", "var_name": "VV", "tipo": "secuencial_variable"},
    {"sheet_name": "Humedad Relativa (HR)", "var_name": "HR", "tipo": "secuencial_variable"},
    {"sheet_name": "Temperatura (TEMP)", "var_name": "T", "tipo": "secuencial_variable"},
    {"sheet_name": "Presión Barométrica (PB)", "var_name": "PB", "tipo": "secuencial_variable"},
    {"sheet_name": "Radiación Solar (RS)", "var_name": "RS", "tipo": "secuencial_variable"},
    {"sheet_name": "Precipitación Pluvial (PP)", "var_name": "PP", "tipo": "secuencial_variable"}
]

# --- FASE 1: Definir la Plantilla Maestra (Sin cambios) ---
print("Iniciando Fase 1: Definiendo plantilla de 366 días...")
fechas_rango = pd.date_range(start='2024-05-01', end='2025-05-01', freq='D')
fechas_fijas = fechas_rango.strftime('%Y-%m-%d').tolist()

horas_fijas = [
    "23:00 - 0:00", "0:00 - 1:00", "1:00 - 2:00", "2:00 - 3:00", "3:00 - 4:00",
    "4:00 - 5:00", "5:00 - 6:00", "6:00 - 7:00", "7:00 - 8:00", "8:00 - 9:00",
    "9:00 - 10:00", "10:00 - 11:00", "11:00 - 12:00", "12:00 - 13:00",
    "13:00 - 14:00", "14:00 - 15:00", "15:00 - 16:00", "16:00 - 17:00",
    "17:00 - 18:00", "18:00 - 19:00", "19:00 - 20:00", "20:00 - 21:00",
    "21:00 - 22:00", "22:00 - 23:00"
]
print(f"Plantilla definida: {len(fechas_fijas)} días y {len(horas_fijas)} horas.")

# --- FASE 2: Crear el DataFrame Maestro (Scaffold) ---
print("Iniciando Fase 2: Creando DataFrame maestro...")
df_master = pd.DataFrame(
    index=pd.MultiIndex.from_product(
        [fechas_fijas, horas_fijas], 
        names=['Fecha', 'Hora']
    )
).reset_index()

# Añadimos columnas vacías (usando la nueva lista de instrucciones)
for item in hojas_a_procesar:
    df_master[item['var_name']] = pd.NA
print(f"DataFrame maestro creado con {len(df_master)} filas.")


# --- FASE 3: (¡REPARADA!) Leer Excel y Rellenar la Plantilla ---
if not os.path.exists(archivo_excel):
    print(f"Error: No se encontró el archivo {archivo_excel}.")
else:
    print(f"Iniciando Fase 3: Leyendo {archivo_excel} y rellenando datos...")
    
    df_master_idx = df_master.set_index(['Fecha', 'Hora'])
    
    # Iteramos sobre la lista de INSTRUCCIONES
    for hoja_info in hojas_a_procesar:
        
        sheet_name = hoja_info['sheet_name']
        var_name = hoja_info['var_name']
        tipo_proceso = hoja_info['tipo']
        
        try:
            df = pd.read_excel(archivo_excel, sheet_name=sheet_name)
            
            # Encontrar la columna 'Valor' (mayúscula o minúscula)
            columna_valor_nombre = None
            if 'Valor' in df.columns: columna_valor_nombre = 'Valor'
            elif 'valor' in df.columns: columna_valor_nombre = 'valor'
            
            if not columna_valor_nombre:
                print(f"  > ADVERTENCIA (Hoja {sheet_name}): No se encontró 'Valor' o 'valor'. Se omitirá.")
                continue

            # ---
            # CASO "horario": Mapeo 1 a 1 usando la columna 'Hora'
            # ---
            if tipo_proceso == "horario":
                if 'Hora' not in df.columns or df['Hora'].isnull().all():
                    print(f"  > ADVERTENCIA (Hoja {sheet_name}): Se esperaba 'horario' pero no hay datos en la columna 'Hora'. Se omitirá.")
                    continue
                
                print(f"  > Procesando Hoja '{sheet_name}' (HORARIA)...")
                df_hourly = df[['Fecha', 'Hora', columna_valor_nombre]].copy()
                df_hourly.rename(columns={columna_valor_nombre: var_name}, inplace=True)
                df_hourly['Fecha'] = df_hourly['Fecha'].astype(str)
                df_hourly['Hora'] = df_hourly['Hora'].astype(str)
                df_hourly = df_hourly.dropna(subset=['Hora'])
                df_hourly = df_hourly[df_hourly['Hora'] != 'nan']
                df_hourly = df_hourly.drop_duplicates(subset=["Fecha", "Hora"], keep="last")
                
                df_hourly_idx = df_hourly.set_index(['Fecha', 'Hora'])
                df_master_idx.update(df_hourly_idx[var_name], overwrite=True)

            # ---
            # CASO "secuencial_variable": Mapeo de N filas a las últimas N horas
            # ---
            elif tipo_proceso == "secuencial_variable":
                print(f"  > Procesando Hoja '{sheet_name}' (SECUENCIAL VARIABLE)...")
                df_daily_seq = df[['Fecha', columna_valor_nombre]].rename(columns={columna_valor_nombre: var_name})
                df_daily_seq['Fecha'] = df_daily_seq['Fecha'].astype(str)

                # Agrupamos por fecha para procesar cada día
                for fecha_dia, grupo_dia in df_daily_seq.groupby('Fecha'):
                    
                    if fecha_dia not in df_master_idx.index:
                        continue # Ignoramos fechas que no estén en nuestra plantilla

                    valores_del_dia = grupo_dia[var_name].tolist()
                    num_valores = len(valores_del_dia)
                    
                    if num_valores == 0:
                        continue
                    
                    # Si hay más de 24, tomamos solo los últimos 24
                    if num_valores > 24:
                         print(f"  > ADVERTENCIA (Hoja {sheet_name}): {fecha_dia} tiene {num_valores} filas. Se usarán solo las últimas 24.")
                         valores_del_dia = valores_del_dia[-24:]
                         num_valores = 24
                         
                    # --- La Lógica Clave ---
                    horas_target = horas_fijas[-num_valores:] # Obtenemos las últimas N horas
                    
                    # Mapeamos los valores a las horas target
                    for i in range(num_valores):
                        hora_a_rellenar = horas_target[i]
                        valor_a_mapear = valores_del_dia[i]
                        df_master_idx.loc[(fecha_dia, hora_a_rellenar), var_name] = valor_a_mapear
            
            print(f"  > Hoja '{sheet_name}' procesada.")
            
        except Exception as e:
            print(f"ERROR: No se pudo procesar la hoja '{sheet_name}'. Error: {e}")

    # --- FASE 4: Limpieza Final (Sin cambios) ---
    print("\nIniciando Fase 4: Limpiando datos (NaN -> \"\")...")
    df_final = df_master_idx.reset_index()
    df_final.fillna("", inplace=True)


    # --- FASE 5: Construir y Guardar el JSON (Sin cambios) ---
    print("Iniciando Fase 5: Generando estructura JSON...")
    datos_json_final = []
    
    for fecha, grupo_dia in df_final.groupby("Fecha"):
        objeto_dia = {
            "fecha": fecha,
            "horas": {}
        }
        grupo_horas = grupo_dia.drop(columns="Fecha").set_index("Hora")
        grupo_horas = grupo_horas.reindex(horas_fijas)
        objeto_dia["horas"] = grupo_horas.to_dict('index')
        datos_json_final.append(objeto_dia)
        
    mapa_fechas = {dia['fecha']: dia for dia in datos_json_final}
    datos_json_final_ordenados = [mapa_fechas[fecha] for fecha in fechas_fijas if fecha in mapa_fechas]

    print("Estructura JSON generada y ordenada.")

    nombre_archivo_salida = "datos_consolidados.json"
    with open(nombre_archivo_salida, "w", encoding="utf-8") as f:
        json.dump(datos_json_final_ordenados, f, indent=2, ensure_ascii=False)

    print(f"--- ¡Éxito! ---")
    print(f"Archivo JSON guardado como: {nombre_archivo_salida}")