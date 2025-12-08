from ultralytics import YOLO
import cv2
import os
import sys

# =================================================================
# 1. CONFIGURACIÓN DE RUTAS Y PARÁMETROS GLOBALES
# =================================================================

# --- RUTAS PRINCIPALES ---
# Ruta del modelo entrenado
MODEL_PATH = 'runs/detect/Target_Area/weights/best.pt'

# Carpeta de PRUEBA que contiene TODAS las imágenes de entrada (Canal 1, Canal 2 y Canal 3)
TEST_DIR = 'C:/Users/migue/Desktop/training_afm/data/images/Test'

# Carpeta raíz donde se guardan los recortes por canal
OUTPUT_BASE_DIR = 'C:/Users/migue/Desktop/training_afm/Resultados/3_canales'
OUTPUT_DIR_CANAL1 = os.path.join(OUTPUT_BASE_DIR, 'canal_1')
OUTPUT_DIR_CANAL2 = os.path.join(OUTPUT_BASE_DIR, 'canal_2')
OUTPUT_DIR_CANAL3 = os.path.join(OUTPUT_BASE_DIR, 'canal_3')

# Parámetros de Recorte y Detección
CROP_SIZE_PX = 80
#CROP_WIDTH_PX = 80      # utilizar para imagenes rectangulares
#CROP_HEIGHT_PX = 30     # utilizar para imagenes rectangulares
CONFIDENCE_THRESHOLD = 0.85
IMG_WIDTH = 256
IMG_HEIGHT = 128

# =================================================================
# 2. INICIALIZACIÓN
# =================================================================
# Asegurar que las carpetas de salida existan
os.makedirs(OUTPUT_DIR_CANAL1, exist_ok=True)
os.makedirs(OUTPUT_DIR_CANAL2, exist_ok=True)
os.makedirs(OUTPUT_DIR_CANAL3, exist_ok=True)

try:
    # Cargar el modelo entrenado
    model = YOLO(MODEL_PATH)
except (FileNotFoundError, OSError) as e:
    print(f"ERROR: No se pudo cargar el modelo en la ruta {MODEL_PATH}. {e}")
    sys.exit()

print(f"Iniciando procesamiento en lote de imágenes en: {TEST_DIR}\n")

# =================================================================
# --- 3. PROCESAMIENTO POR LOTE ---
# =================================================================

# Buscar solo las imágenes del Canal 1 para iniciar la detección
canal1_files = [f for f in os.listdir(TEST_DIR) if f.endswith('_Canal_1.png')]
total_files = len(canal1_files)
processed_count = 0

for filename_canal1 in canal1_files:
    # 3.1. Definir los Nombres de Archivo Pares
    filename_canal2 = filename_canal1.replace('_Canal_1.png', '_Canal_2.png')
    filename_canal3 = filename_canal1.replace('_Canal_1.png', '_Canal_3.png')

    image_canal1_path = os.path.join(TEST_DIR, filename_canal1)
    image_canal2_path = os.path.join(TEST_DIR, filename_canal2)
    image_canal3_path = os.path.join(TEST_DIR, filename_canal3)

    # Verificar que las imágenes pares existan
    if not os.path.exists(image_canal2_path):
        print(f"AVISO: Se omite {filename_canal1}. No se encontró el par {filename_canal2}.")
        continue
    if not os.path.exists(image_canal3_path):
        print(f"AVISO: Se omite {filename_canal1}. No se encontró el par {filename_canal3}.")
        continue

    # 3.2. Detección en Canal 1
    results = model.predict(source=image_canal1_path, save=False, conf=CONFIDENCE_THRESHOLD, verbose=False, imgsz=(IMG_WIDTH, IMG_HEIGHT))

    if results and len(results[0].boxes) > 0:
        # 3.3. Extraer Coordenadas del Centro
        box = results[0].boxes[0]
        coords_px = box.xywh[0] # [x_center, y_center, width, height] en Píxeles

        center_x = int(coords_px[0].item())
        center_y = int(coords_px[1].item())

        # 3.4. Calcular el Área de Recorte Fija
        half_crop = CROP_SIZE_PX // 2

        #half_width = CROP_WIDTH_PX // 2    # utilizar para imagenes rectangulares
        #half_height = CROP_HEIGHT_PX // 2  # utilizar para imagenes rectangulares


        # Recorte y protección contra límites
        x_min = max(0, center_x - half_crop)               # modificar half_crop por half_width para imagenes rectangulares
        y_min = max(0, center_y - half_crop)               # modificar half_crop por half_height para imagenes rectangulares
        x_max = min(IMG_WIDTH, center_x + half_crop)       # modificar half_crop por half_width para imagenes rectangulares
        y_max = min(IMG_HEIGHT, center_y + half_crop)      # modificar half_crop por half_height para imagenes rectangulares

        # 3.5. Cargar y Recortar los tres Canales
        img_canal1 = cv2.imread(image_canal1_path, -1)
        img_canal2 = cv2.imread(image_canal2_path, -1)
        img_canal3 = cv2.imread(image_canal3_path, -1)

        if img_canal1 is not None and img_canal2 is not None and img_canal3 is not None:
            # Aplicar Recorte: [filas (Y), columnas (X)]
            cropped_canal1 = img_canal1[y_min:y_max, x_min:x_max]
            cropped_canal2 = img_canal2[y_min:y_max, x_min:x_max]
            cropped_canal3 = img_canal3[y_min:y_max, x_min:x_max]

            # 3.6. Guardar los Resultados en las carpetas correspondientes
            base_name_c1 = os.path.splitext(filename_canal1)[0]
            base_name_c2 = os.path.splitext(filename_canal2)[0]
            base_name_c3 = os.path.splitext(filename_canal3)[0]
            
            output_filename_c1 = os.path.join(OUTPUT_DIR_CANAL1, f"{base_name_c1}_recorte_{CROP_SIZE_PX}px.png")
            output_filename_c2 = os.path.join(OUTPUT_DIR_CANAL2, f"{base_name_c2}_recorte_{CROP_SIZE_PX}px.png")
            output_filename_c3 = os.path.join(OUTPUT_DIR_CANAL3, f"{base_name_c3}_recorte_{CROP_SIZE_PX}px.png")
            
            cv2.imwrite(output_filename_c1, cropped_canal1)
            cv2.imwrite(output_filename_c2, cropped_canal2)
            cv2.imwrite(output_filename_c3, cropped_canal3)

            print(f"Recorte exitoso para: {filename_canal1} (Centro: {center_x}, {center_y})")
            processed_count += 1
        else:
            if img_canal1 is None:
                print(f"ERROR: No se pudo cargar la imagen del Canal 1: {filename_canal1}")
            if img_canal2 is None:
                print(f"ERROR: No se pudo cargar la imagen del Canal 2: {filename_canal2}")
            if img_canal3 is None:
                print(f"ERROR: No se pudo cargar la imagen del Canal 3: {filename_canal3}")

    else:
        print(f"FALLO en Detección para: {filename_canal1}")

print("\n--- Resumen del Lote ---")
print(f"Total de pares de imágenes procesados (Canal 1): {total_files}")
print(f"Total de recortes exitosos guardados: {processed_count}")
print(f"Archivos guardados en:")
print(f"  - Canal 1: {OUTPUT_DIR_CANAL1}")
print(f"  - Canal 2: {OUTPUT_DIR_CANAL2}")
print(f"  - Canal 3: {OUTPUT_DIR_CANAL3}")
