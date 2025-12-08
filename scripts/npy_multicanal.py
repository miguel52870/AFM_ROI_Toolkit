from ultralytics import YOLO
import numpy as np
import os
import sys
from scipy.stats import pearsonr
from pathlib import Path

# =================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS GLOBALES
# =================================================================

# --- RUTAS PRINCIPALES ---
# Ruta al modelo entrenado
MODEL_PATH = 'runs/detect/Target_Area/weights/best.pt'

# Carpeta que contiene las IMÁGENES PNG del Canal 1 (Fuente para YOLO)
IMAGE_DIR_CANAL1 = 'C:/Users/migue/Desktop/training_afm/data/images/Test'

# Carpeta que contiene los ARRAYS NUMPY (.npy) del Canal 1 (Para Recortar)
NUMPY_DIR = 'C:/Users/migue/Desktop/training_afm/data/numpy_arrays/'

# Carpeta base donde se guardarán los arrays NumPy recortados
OUTPUT_NUMPY_DIR_BASE = 'C:/Users/migue/Desktop/training_afm/Resultados/numpy_recortes'
# Subcarpetas de salida por canal
OUTPUT_NUMPY_DIR_CANAL1 = os.path.join(OUTPUT_NUMPY_DIR_BASE, 'canal_1')
OUTPUT_NUMPY_DIR_CANAL2 = os.path.join(OUTPUT_NUMPY_DIR_BASE, 'canal_2')
OUTPUT_NUMPY_DIR_CANAL3 = os.path.join(OUTPUT_NUMPY_DIR_BASE, 'canal_3')

# Parámetros de Recorte y Detección
CROP_SIZE_PX = 80
CONFIDENCE_THRESHOLD = 0.85
#CROP_WIDTH_PX = 80      # utilizar para imagenes rectangulares
#CROP_HEIGHT_PX = 30     # utilizar para imagenes rectangulares
IMG_WIDTH = 256
IMG_HEIGHT = 128

# =================================================================
# 2. INICIALIZACIÓN
# =================================================================

# Asegurar que las carpetas de salida existan
os.makedirs(OUTPUT_NUMPY_DIR_CANAL1, exist_ok=True)
os.makedirs(OUTPUT_NUMPY_DIR_CANAL2, exist_ok=True)
os.makedirs(OUTPUT_NUMPY_DIR_CANAL3, exist_ok=True)

try:
    # Cargar el modelo entrenado
    model = YOLO(MODEL_PATH)
except (FileNotFoundError, ImportError, RuntimeError) as e:
    print(f"ERROR: No se pudo cargar el modelo en la ruta {MODEL_PATH}. {e}")
    sys.exit()

# La fuente de procesamiento ahora son las imágenes PNG
print(f"Iniciando procesamiento de recorte multi-canal (Fuente PNG) en: {IMAGE_DIR_CANAL1}\n")

# =================================================================
# 3. PROCESAMIENTO POR LOTE
# =================================================================

# Buscamos los archivos .png del Canal 1 (Fuente de detección)
canal1_png_files = sorted([f for f in os.listdir(IMAGE_DIR_CANAL1) if f.endswith('_Canal_1.png')])
total_files = len(canal1_png_files)
processed_counts = {"canal_1": 0, "canal_2": 0, "canal_3": 0}
processed_arrays_paths = {"canal_1": [], "canal_2": [], "canal_3": []}

for filename_canal1_png in canal1_png_files:

    # 3.1. Definir los Nombres de Archivo Pares y Rutas de Entrada
    base_name = filename_canal1_png.replace('_Canal_1.png', '')

    # Rutas para la Detección (PNG)
    path_canal1_png = os.path.join(IMAGE_DIR_CANAL1, filename_canal1_png)

    # Rutas para el Recorte (NumPy)
    path_canal1_npy = os.path.join(NUMPY_DIR, base_name + '_Canal_1.npy')
    path_canal2_npy = os.path.join(NUMPY_DIR, base_name + '_Canal_2.npy')
    path_canal3_npy = os.path.join(NUMPY_DIR, base_name + '_Canal_3.npy')

    # Verificamos que el archivo PNG de detección exista
    if not os.path.exists(path_canal1_png):
        print(f"ERROR: No se encontró la fuente PNG {path_canal1_png}. Saltando.")
        continue

    # 3.2. Detección en la Imagen PNG (Fuente directa)
    results = model.predict(source=path_canal1_png, save=False, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=IMG_WIDTH)

    # 3.3. Extracción de Coordenadas
    if results and len(results[0].boxes) > 0:

        # Tomamos la primera detección (única etiqueta: Target_Area o center_anchor)
        box = results[0].boxes[0]
        coords_px = box.xywh[0]

        center_x = int(coords_px[0].item())
        center_y = int(coords_px[1].item())
        # Cálculo del half-crop
        half_crop = CROP_SIZE_PX // 2

        # 3.4. Calcular el Área de Recorte Fija
        x_min = max(0, center_x - half_crop)
        y_min = max(0, center_y - half_crop)
        x_max = min(IMG_WIDTH, center_x + half_crop)
        y_max = min(IMG_HEIGHT, center_y + half_crop)

        # 3.5. Recortar y Guardar Canal 1, Canal 2 y Canal 3 (NumPy)

        # Definir los canales a procesar y sus rutas de entrada/salida
        canales_a_procesar = [
            (path_canal1_npy, OUTPUT_NUMPY_DIR_CANAL1, 1), # Canal 1 (Fuente)
            (path_canal2_npy, OUTPUT_NUMPY_DIR_CANAL2, 2), # Canal 2 (Análisis)
            (path_canal3_npy, OUTPUT_NUMPY_DIR_CANAL3, 3)  # Canal 3 (Análisis)
        ]

        for path_npy, output_dir, canal_num in canales_a_procesar:

            # Verificación de existencia del archivo NumPy
            if not os.path.exists(path_npy):
                continue

            try:
                # Cargar el array NumPy original
                array_canal = np.load(path_npy)

                # Aplicar Recorte: [filas (Y), columnas (X)]
                cropped_canal = array_canal[y_min:y_max, x_min:x_max]

                # Guardar el resultado
                out_filename = os.path.join(output_dir, f"{base_name}_recorte_{CROP_SIZE_PX}px_Canal_{canal_num}.npy")
                np.save(out_filename, cropped_canal)

                processed_counts[f'canal_{canal_num}'] += 1

                print(f"Recorte exitoso Canal {canal_num}: {os.path.basename(out_filename)} (Centro: {center_x}, {center_y})")
            except (IOError, OSError, ValueError) as e:
                print(f"AVISO: No se pudo recortar/guardar Canal {canal_num} para {base_name}: {e}")

    else:
        print(f"FALLO en Detección para la imagen PNG: {filename_canal1_png}. No se encontró Target_Area.")

print("\n--- Resumen del Lote Multi-Canal ---")
print(f"Total de archivos PNG examinados: {total_files}")
print(f"Recortes guardados Canal 1: {processed_counts['canal_1']}")
print(f"Recortes guardados Canal 2: {processed_counts['canal_2']}")
print(f"Recortes guardados Canal 3: {processed_counts['canal_3']}")
print(f"Carpetas de salida (base): {OUTPUT_NUMPY_DIR_BASE}")
