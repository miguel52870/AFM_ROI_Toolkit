"""
batch_multicanal.py — Paso 2a del pipeline de recorte alineado
Tesis: Evolución de Dominios Ferroeléctricos con Deep Learning

Lee: Resultados/registro/coordenadas_registro.csv  (generado por calcular_registro.py)
Genera recortes PNG centrados en la estructura para:
  - Canal 1, 2, 3 preprocesados  → Resultados/3_canales/canal_X/
  - C2 diff                      → Resultados/3_canales/diff/
  - C3 mask                      → Resultados/3_canales/mask/

Soporta recortes cuadrados y rectangulares mediante CROP_MODE.

REQUISITO: correr calcular_registro.py antes que este script.
"""

import cv2
import os
import sys
import csv

# =================================================================
# 1. CONFIGURACIÓN
# =================================================================

# =================================================================
# RAIZ DEL PROYECTO
# =================================================================
# Unica ruta que hay que cambiar al mover el proyecto o al usarlo en
# otra maquina. Todo lo demas se deriva de aqui.
BASE_DIR = 'C:/Users/migue/Desktop/training_afm'

# Carpeta con imágenes PNG preprocesadas (_prep)
IMAGE_DIR       = f'{BASE_DIR}/data/images/Test'

# Carpetas con PNGs de diff y mask
DIFF_PNG_DIR    = f'{BASE_DIR}/data/diff/png'
MASK_PNG_DIR    = f'{BASE_DIR}/data/mask/png'

# CSV generado por calcular_registro.py
INPUT_CSV       = f'{BASE_DIR}/Resultados/registro/coordenadas_registro.csv'

# Carpetas de salida
OUTPUT_BASE_DIR = f'{BASE_DIR}/Resultados/3_canales'
OUTPUT_DIR_C1   = os.path.join(OUTPUT_BASE_DIR, 'canal_1')
OUTPUT_DIR_C2   = os.path.join(OUTPUT_BASE_DIR, 'canal_2')
OUTPUT_DIR_C3   = os.path.join(OUTPUT_BASE_DIR, 'canal_3')
OUTPUT_DIR_DIFF = os.path.join(OUTPUT_BASE_DIR, 'diff')
OUTPUT_DIR_MASK = os.path.join(OUTPUT_BASE_DIR, 'mask')

# --- MODO DE RECORTE ---
# 'cuadrado'    → usa CROP_SIZE para ancho y alto
# 'rectangular' → usa CROP_WIDTH y CROP_HEIGHT de forma independiente
CROP_MODE   = 'cuadrado'

# Parámetros cuadrado
CROP_SIZE   = 80

# Parámetros rectangular (solo se usan si CROP_MODE = 'rectangular')
# IMPORTANTE: ambos valores deben ser divisibles por 32 para compatibilidad
# con los modelos U-Net (EfficientNet-B0 hace downsampling ×32)
# Ejemplos válidos: 32, 64, 96, 128 / 32, 64, 96
CROP_WIDTH  = 80    # ancho del recorte en px (eje X)
CROP_HEIGHT = 64    # alto del recorte en px  (eje Y)

FILE_PREFIX = 'bifeo_training'

# Digitos del numero de frame en el nombre de archivo (relleno con ceros).
# Debe coincidir con FRAME_DIGITS de afm_a_gwy.py en AFM_ToolKit, que es
# quien genera los nombres. Este script los RECONSTRUYE para buscar los
# archivos, asi que un valor distinto no encontraria nada.
#   0 -> bifeo_training_21    sin relleno (formato historico, hasta 99 frames)
#   3 -> bifeo_training_021   hasta 999
#   4 -> bifeo_training_0021  hasta 9999
FRAME_DIGITS = 0

IMG_WIDTH   = 256
IMG_HEIGHT  = 128

# =================================================================
# 2. INICIALIZACIÓN
# =================================================================

for d in [OUTPUT_DIR_C1, OUTPUT_DIR_C2, OUTPUT_DIR_C3,
          OUTPUT_DIR_DIFF, OUTPUT_DIR_MASK]:
    os.makedirs(d, exist_ok=True)

if CROP_MODE not in ('cuadrado', 'rectangular'):
    print(f"ERROR: CROP_MODE='{CROP_MODE}' no válido. Usar 'cuadrado' o 'rectangular'.")
    sys.exit(1)

if not os.path.exists(INPUT_CSV):
    print(f"ERROR: No se encontró el CSV de registro: {INPUT_CSV}")
    print("       Corre primero calcular_registro.py")
    sys.exit(1)

# Calcular half según el modo seleccionado
if CROP_MODE == 'cuadrado':
    half_w = CROP_SIZE // 2
    half_h = CROP_SIZE // 2
    crop_label = f"{CROP_SIZE}px"
else:
    half_w = CROP_WIDTH  // 2
    half_h = CROP_HEIGHT // 2
    crop_label = f"{CROP_WIDTH}x{CROP_HEIGHT}px"
    # Advertencia si las dimensiones no son divisibles por 32
    if CROP_WIDTH % 32 != 0 or CROP_HEIGHT % 32 != 0:
        print(f"ADVERTENCIA: CROP_WIDTH={CROP_WIDTH} o CROP_HEIGHT={CROP_HEIGHT} "
              f"no son divisibles por 32.")
        print(f"  Los modelos U-Net requieren dimensiones divisibles por 32.")
        print(f"  Valores recomendados: 32, 64, 96, 128...\n")

# =================================================================
# 3. LEER CSV DE REGISTRO
# =================================================================

registro = {}
with open(INPUT_CSV, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['center_x'] and row['center_y']:
            registro[int(row['frame'])] = {
                'cx': int(row['center_x']),
                'cy': int(row['center_y']),
            }

print(f"Registro cargado  : {len(registro)} frames")
print(f"Modo de recorte   : {CROP_MODE}")
print(f"Tamaño de recorte : {crop_label}\n")

# =================================================================
# 4. PROCESAMIENTO POR FRAME
# =================================================================

processed = 0
failed    = 0

for frame, reg in sorted(registro.items()):

    cx = reg['cx']
    cy = reg['cy']

    # Calcular ROI centrado en la estructura
    x_min = max(0, cx - half_w)
    x_max = min(IMG_WIDTH,  cx + half_w)
    y_min = max(0, cy - half_h)
    y_max = min(IMG_HEIGHT, cy + half_h)

    # --- Canales prep (C1, C2, C3) ---
    canales_prep = [
        (os.path.join(IMAGE_DIR, f"{FILE_PREFIX}_{frame:0{FRAME_DIGITS}d}_Canal_1_prep.png"), OUTPUT_DIR_C1),
        (os.path.join(IMAGE_DIR, f"{FILE_PREFIX}_{frame:0{FRAME_DIGITS}d}_Canal_2_prep.png"), OUTPUT_DIR_C2),
        (os.path.join(IMAGE_DIR, f"{FILE_PREFIX}_{frame:0{FRAME_DIGITS}d}_Canal_3_prep.png"), OUTPUT_DIR_C3),
    ]

    # --- Diff: existe desde frame 22 ---
    diff_path = os.path.join(DIFF_PNG_DIR, f"{FILE_PREFIX}_{frame:0{FRAME_DIGITS}d}_Canal_2_diff.png")

    # --- Mask: existe para todos los frames ---
    mask_path = os.path.join(MASK_PNG_DIR, f"{FILE_PREFIX}_{frame:0{FRAME_DIGITS}d}_Canal_3_mask.png")

    ok = 0
    errores = []

    # Recortar canales prep
    for src_path, out_dir in canales_prep:
        if not os.path.exists(src_path):
            errores.append(os.path.basename(src_path))
            continue
        img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            errores.append(f"error_carga:{os.path.basename(src_path)}")
            continue
        cropped  = img[y_min:y_max, x_min:x_max]
        base     = os.path.splitext(os.path.basename(src_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{base}_recorte_{crop_label}.png"), cropped)
        ok += 1

    # Recortar diff
    if os.path.exists(diff_path):
        img_diff = cv2.imread(diff_path, cv2.IMREAD_UNCHANGED)
        if img_diff is not None:
            cropped_diff = img_diff[y_min:y_max, x_min:x_max]
            base_diff    = os.path.splitext(os.path.basename(diff_path))[0]
            cv2.imwrite(os.path.join(OUTPUT_DIR_DIFF,
                        f"{base_diff}_recorte_{crop_label}.png"), cropped_diff)
            ok += 1
        else:
            errores.append(f"error_carga:{os.path.basename(diff_path)}")
    else:
        if frame != 21:
            errores.append(f"sin_diff_frame_{frame}")

    # Recortar mask
    if os.path.exists(mask_path):
        img_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if img_mask is not None:
            cropped_mask = img_mask[y_min:y_max, x_min:x_max]
            base_mask    = os.path.splitext(os.path.basename(mask_path))[0]
            cv2.imwrite(os.path.join(OUTPUT_DIR_MASK,
                        f"{base_mask}_recorte_{crop_label}.png"), cropped_mask)
            ok += 1
        else:
            errores.append(f"error_carga:{os.path.basename(mask_path)}")
    else:
        errores.append(f"sin_mask_frame_{frame}")

    if errores:
        print(f"Frame {frame}: {ok} recortes  ⚠ {errores}")
        failed += 1
    else:
        print(f"Frame {frame}: ✓  centro=({cx}, {cy})  "
              f"ROI=[{y_min}:{y_max}, {x_min}:{x_max}]  {crop_label}")
        processed += 1

# =================================================================
# 5. RESUMEN
# =================================================================

print(f"\n{'─'*50}")
print(f"Frames completos : {processed}")
print(f"Frames con aviso : {failed}")
print(f"Modo             : {CROP_MODE} ({crop_label})")
print(f"PNGs guardados en:")
print(f"  Canal 1 → {OUTPUT_DIR_C1}")
print(f"  Canal 2 → {OUTPUT_DIR_C2}")
print(f"  Canal 3 → {OUTPUT_DIR_C3}")
print(f"  Diff    → {OUTPUT_DIR_DIFF}")
print(f"  Mask    → {OUTPUT_DIR_MASK}")