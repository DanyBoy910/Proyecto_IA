import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
import os

with open('resultados.txt', 'r') as fl:
    content = fl.read().replace('\t', ' ').split('\n')

datos = []
for i in range(len(content)):
    if 'prueba_1' in content[i]:
        datos.append(content[i].split('train "')[1].split(' prueba_1')[0].split('" '))
    elif content[i].startswith("MAE"):
        if len(datos[-1]) == 2:
            datos[-1].append(float(content[i].split(':')[1].strip()))
            datos[-1].append(float(content[i+1].split(':')[1].strip()))
            datos[-1].append(float(content[i+2].split(':')[1].strip()))
        else:
            datos[-1][2] += float(content[i].split(':')[1].strip())
            datos[-1][3] += float(content[i+1].split(':')[1].strip())
            datos[-1][4] += float(content[i+2].split(':')[1].strip())

# print(datos[0])
for it in datos[:]:
    if len(it) == 5:
        print(f'{it[0]}: {it[1]}, {it[2]/2:.4f}, {it[3]/2:.4f}, {it[4]/2:.4f}')
    else:
        del datos[datos.index(it)]

# Creamos una carpeta para guardar las imagenes
output_folder = "graficas_metricas"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Funciones de utilidad para parsear la data
def parse_line(line):
    pollutant = line[0]
    params_str = line[1]
    mae = round(line[2] / 2, 4)
    rmse = round(line[3] / 2, 4)
    r2 = round(line[4] / 2, 4)
    
    # Parsear parametros a un diccionario
    params = {}
    for item in params_str.split():
        key, value = item.split('=')
        # Intentamos convertir a numero si es posible, sino string
        try:
            if '_' in value: # Caso HIDDEN_SIZES
                params[key] = value.replace('_',',')
            else:
                params[key] = float(value)
        except ValueError:
             params[key] = value

    return {
        'Pollutant': pollutant,
        'Params': params,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'RawParams': params_str # Guardamos el string completo para debug
    }

# 1. Parsear los datos
data = []
for line in datos:
    parsed = parse_line(line)
    if parsed:
        data.append(parsed)

with open('algo.txt', 'w') as fk:
    # 2. Procesar por grupos de 8
    chunk_size = 8
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        if not chunk or len(chunk) < chunk_size: continue
        
        pollutant = chunk[0]['Pollutant']
        
        # Identificar que variable cambia en este grupo
        # Comparamos el primer registro con el segundo (o todos) para ver diffs
        keys = chunk[0]['Params'].keys()
        varying_param = None
        
        # Revisamos key por key cual tiene mas de 1 valor unico en el grupo
        for key in keys:
            values = set(row['Params'][key] for row in chunk)
            if len(values) > 1:
                varying_param = key
                break
                
        if not varying_param:
            varying_param = "UNKNOWN_GROUP"

        # Preparar datos para plotear
        x_values = [str(row['Params'][varying_param]) for row in chunk] # String para que HIDDEN_SIZES no rompa
        mae_values = [row['MAE'] for row in chunk]
        rmse_values = [row['RMSE'] for row in chunk]
        r2_values = [row['R2'] for row in chunk]
        
        # Encontrar el MEJOR desempeno (Criterio: Mayor R2)
        # Tambien podriamos buscar menor RMSE.
        best_idx = r2_values.index(max(r2_values))

        lista_valores = list(zip(x_values, r2_values))
        lista_valores.sort(key=lambda x: x[1], reverse=True)
        # Imprimir resultados en consola
        print(f"\n=== Resultados para {pollutant} variando {varying_param} ===")
        fk.write(f"{pollutant}={varying_param}:")
        for val, r2 in lista_valores[:4]:
            print(f"{pollutant} | {varying_param}={val} | R2={r2}")
            fk.write(f" {val},")
        fk.write("\n")
        
        # Crear la figura con 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        fig.suptitle(f'Analisis de {pollutant} variando {varying_param}', fontsize=16)
        
        # Plot MAE
        ax1.plot(x_values, mae_values, marker='o', color='skyblue', label='MAE')
        ax1.set_ylabel('MAE (Menor es mejor)')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot RMSE
        ax2.plot(x_values, rmse_values, marker='s', color='salmon', label='RMSE')
        ax2.set_ylabel('RMSE (Menor es mejor)')
        ax2.grid(True, linestyle='--', alpha=0.6)

        # Plot R2
        ax3.plot(x_values, r2_values, marker='^', color='lightgreen', label='R²')
        ax3.set_ylabel('R² (Mayor es mejor)')
        ax3.set_xlabel(f'Valor de {varying_param}')
        ax3.grid(True, linestyle='--', alpha=0.6)
        
        # Resaltar el mejor punto en R2
        best_x = x_values[best_idx]
        best_r2 = r2_values[best_idx]
        
        # Anotacion en el grafico de R2
        ax3.annotate(f'Mejor: {best_r2}', 
                    xy=(best_x, best_r2), 
                    xytext=(best_x, best_r2 + (max(r2_values)*0.05)),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center')
                    
        # Dibujar linea vertical en todos los plots para marcar el ganador
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=best_x, color='gold', linestyle='-', alpha=0.5, linewidth=2, label='Mejor Configuración (R²)')
            ax.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Guardar
        filename = f"{output_folder}/{pollutant}_{varying_param}.png"
        plt.savefig(filename)
        plt.close() # Importante cerrar para liberar memoria
        
        print(f"Generada grafica: {filename} -> Mejor {varying_param} = {best_x} (R²: {best_r2})")

print(f"\nProceso terminado. Las imagenes estan en la carpeta '{output_folder}'.")