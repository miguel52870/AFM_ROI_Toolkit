"""
calcular_registro.py — Paso 1 del pipeline de recorte alineado
Tesis: Evolución de Dominios Ferroeléctricos con Deep Learning

Corre YOLO sobre Canal 1 de cada frame para detectar el centro físico
de la estructura ferroeléctrica. Aplica corrección de trayectoria según
el modo seleccionado.

Modos disponibles (SMOOTHING_MODE):
  'outlier'    → preserva valores YOLO correctos, solo corrige frames que
                 se apartan del promedio de sus vecinos
  'moving_avg' → suaviza toda la serie con media movil centrada

Genera: Resultados/registro/coordenadas_registro.csv
  Columnas: frame, center_x, center_y, center_x_raw, center_y_raw, conf, status
    center_x / center_y     -> coordenadas corregidas (enteras). Son las que
                               consumen batch_multicanal.py y npy_multicanal.py
    center_x_raw / _y_raw   -> float original de YOLO, sin truncar ni corregir.
                               Solo registro, para comparar cruda vs. suavizada
  status: 'ok' | 'suavizado' | 'fallo_deteccion' | 'sin_png'

Uso:
  1. Correr este script PRIMERO
  2. Luego correr batch_multicanal.py  (recortes PNG centrados en estructura)
  3. Luego correr npy_multicanal.py    (recortes NPY centrados en estructura)
"""

from ultralytics import YOLO
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

MODEL_PATH = 'runs/detect/Target_Area_prep/weights/best.pt'

# Imágenes PNG Canal 1 preprocesadas (fuente para YOLO)
IMAGE_DIR  = f'{BASE_DIR}/data/images/Test'

# Salida del CSV de registro
OUTPUT_DIR = f'{BASE_DIR}/Resultados/registro'
OUTPUT_CSV = os.path.join(OUTPUT_DIR, 'coordenadas_registro.csv')

# Parámetros de detección YOLO
FRAME_START  = 21
FRAME_END    = 60
FILE_PREFIX  = 'bifeo_training'
CONFIDENCE   = 0.85
IMG_WIDTH    = 256
IMG_HEIGHT   = 128

# =================================================================
# MODO DE CORRECCIÓN
# =================================================================
#
# 'outlier'    → Solo corrige frames cuyo center_x o center_y difiere
#                más de OUTLIER_THRESHOLD_X/Y px respecto al promedio
#                de sus vecinos. El resto mantiene el valor YOLO original.
#
# 'moving_avg' → Reemplaza el centro de TODOS los frames por el promedio
#                de una ventana centrada en ese frame. Suaviza la trayectoria
#                completa del ROI. Útil para estabilizar GIFs.
#
SMOOTHING_MODE = 'outlier'   # opciones: 'outlier' | 'moving_avg'

# --- PARÁMETROS COMPARTIDOS ---
#
# WINDOW (frames a cada lado):
#   'outlier'    → cuántos vecinos a cada lado para calcular el promedio de referencia
#   'moving_avg' → cuántos frames a cada lado entran en la ventana de suavizado
#   Recomendado: 2–4. No usar más de 4 en series cortas (<50 frames).
#
WINDOW = 4

# --- PARÁMETROS EXCLUSIVOS DE MODO 'outlier' ---
#
# OUTLIER_THRESHOLD_X / Y (px):
#   Umbral mínimo de diferencia para considerar un frame como outlier.
#   Con YOLO (enteros), el mínimo práctico es 1.0 px.
#   Valores bajos (1.0–2.0) → más frames corregidos (suavizado agresivo)
#   Valores altos (4.0–6.0) → solo outliers evidentes como saltos bruscos
#
OUTLIER_THRESHOLD_Y = 0.25
OUTLIER_THRESHOLD_X = 0.25

# =================================================================
# 2. INICIALIZACIÓN
# =================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

if SMOOTHING_MODE not in ('outlier', 'moving_avg'):
    print(f"ERROR: SMOOTHING_MODE='{SMOOTHING_MODE}' no válido. Usar 'outlier' o 'moving_avg'.")
    sys.exit(1)

try:
    model = YOLO(MODEL_PATH)
except (FileNotFoundError, OSError) as e:
    print(f"ERROR: No se pudo cargar el modelo: {e}")
    sys.exit(1)

# =================================================================
# 3. DETECCIÓN YOLO FRAME A FRAME
# =================================================================

print(f"Modelo  : {MODEL_PATH}")
print(f"Modo    : {SMOOTHING_MODE}  |  ventana={WINDOW}" +
      (f"  |  umbral_x={OUTLIER_THRESHOLD_X} px  umbral_y={OUTLIER_THRESHOLD_Y} px"
       if SMOOTHING_MODE == 'outlier' else ''))
print(f"Frames  : {FRAME_START}–{FRAME_END}\n")
print(f"{'Frame':<8} {'center_x':>10} {'center_y':>10} {'conf':>8} {'status'}")
print("─" * 50)

detecciones = {}  # frame -> {'cx': int, 'cy': int, 'conf': float, 'status': str}

for frame in range(FRAME_START, FRAME_END + 1):

    png_path = os.path.join(IMAGE_DIR, f"{FILE_PREFIX}_{frame}_Canal_1_prep.png")

    if not os.path.exists(png_path):
        print(f"{frame:<8} {'—':>10} {'—':>10} {'—':>8} sin_png")
        detecciones[frame] = {'cx': None, 'cy': None,
                              'cx_raw': None, 'cy_raw': None,
                              'conf': None, 'status': 'sin_png'}
        continue

    results = model.predict(source=png_path, save=False, conf=CONFIDENCE,
                             verbose=False, imgsz=(IMG_WIDTH, IMG_HEIGHT))

    if results and len(results[0].boxes) > 0:
        box    = results[0].boxes[0]
        coords = box.xywh[0]
        # YOLO devuelve float. Se conserva el valor subpixel porque la deriva
        # del escaner es continua: truncar a entero la convierte en una
        # escalera de saltos de 1 px que no corresponde a la trayectoria real.
        # El entero se mantiene como valor operativo: el recorte necesita
        # indices enteros y los scripts posteriores leen center_x/center_y.
        cx_raw = float(coords[0].item())
        cy_raw = float(coords[1].item())
        cx     = int(cx_raw)
        cy     = int(cy_raw)
        conf   = round(float(box.conf[0].item()), 4)

        print(f"{frame:<8} {cx:>10} {cy:>10} {conf:>8.4f} ok")
        detecciones[frame] = {'cx': cx, 'cy': cy,
                              'cx_raw': cx_raw, 'cy_raw': cy_raw,
                              'conf': conf, 'status': 'ok'}

    else:
        print(f"{frame:<8} {'—':>10} {'—':>10} {'—':>8} fallo_deteccion")
        detecciones[frame] = {'cx': None, 'cy': None,
                              'cx_raw': None, 'cy_raw': None,
                              'conf': None, 'status': 'fallo_deteccion'}

frames_sorted = sorted(detecciones.keys())

# =================================================================
# 4A. MODO OUTLIER — corrige solo frames anómalos
# =================================================================

def get_neighbors(frames_sorted, i, detecciones, axis, window):
    """Devuelve valores de 'axis' de hasta 'window' vecinos válidos a cada lado."""
    prev_vals = [detecciones[f][axis] for f in reversed(frames_sorted[max(0, i-window):i])
                 if detecciones[f][axis] is not None]
    next_vals = [detecciones[f][axis] for f in frames_sorted[i+1:i+1+window]
                 if detecciones[f][axis] is not None]
    return prev_vals, next_vals

def interpolate(prev_vals, next_vals, current_val):
    """Promedio de vecinos disponibles. Si no hay vecinos, mantiene el valor."""
    all_vals = prev_vals + next_vals
    if not all_vals:
        return current_val
    return int(round(sum(all_vals) / len(all_vals)))

if SMOOTHING_MODE == 'outlier':

    print(f"\nSuavizando cuantizacion (umbral_x={OUTLIER_THRESHOLD_X} px, "
          f"umbral_y={OUTLIER_THRESHOLD_Y} px, ventana={WINDOW})...")
    print("  Con umbrales por debajo de 1 px sobre coordenadas enteras, lo que")
    print("  se corrige es el escalonado del truncamiento, no fallos de YOLO.\n")

    outliers_found = 0

    for i, frame in enumerate(frames_sorted):

        d = detecciones[frame]
        if d['cx'] is None:
            continue

        outlier_x = outlier_y = False

        prev_cx, next_cx = get_neighbors(frames_sorted, i, detecciones, 'cx', WINDOW)
        all_cx = prev_cx + next_cx
        if all_cx:
            avg_cx = sum(all_cx) / len(all_cx)
            if abs(d['cx'] - avg_cx) > OUTLIER_THRESHOLD_X:
                outlier_x = True

        prev_cy, next_cy = get_neighbors(frames_sorted, i, detecciones, 'cy', WINDOW)
        all_cy = prev_cy + next_cy
        if all_cy:
            avg_cy = sum(all_cy) / len(all_cy)
            if abs(d['cy'] - avg_cy) > OUTLIER_THRESHOLD_Y:
                outlier_y = True

        if outlier_x or outlier_y:
            outliers_found += 1
            cx_orig, cy_orig = d['cx'], d['cy']
            cx_new = interpolate(prev_cx, next_cx, cx_orig) if outlier_x else cx_orig
            cy_new = interpolate(prev_cy, next_cy, cy_orig) if outlier_y else cy_orig

            print(f"  SUAVIZADO frame {frame}:")
            if outlier_x:
                print(f"    center_x: {cx_orig} → {cx_new} px  (ref. vecinos: {avg_cx:.1f} px)")
            if outlier_y:
                print(f"    center_y: {cy_orig} → {cy_new} px  (ref. vecinos: {avg_cy:.1f} px)")

            detecciones[frame]['cx']     = cx_new
            detecciones[frame]['cy']     = cy_new
            detecciones[frame]['status'] = 'suavizado'

    if outliers_found == 0:
        print("  Ningun frame requirio suavizado.")
    else:
        print(f"\n  Total frames suavizados: {outliers_found}")

# =================================================================
# 4B. MODO MOVING_AVG — suaviza toda la serie
# =================================================================

elif SMOOTHING_MODE == 'moving_avg':

    print(f"\nAplicando media móvil (ventana={WINDOW} frames a cada lado)...\n")

    # Guardar valores originales YOLO para mostrar en log
    originales = {f: {'cx': detecciones[f]['cx'], 'cy': detecciones[f]['cy']}
                  for f in frames_sorted}

    for i, frame in enumerate(frames_sorted):

        d = detecciones[frame]
        if d['cx'] is None:
            continue

        # Ventana centrada: frame-W hasta frame+W (inclusive), solo frames válidos
        window_frames = frames_sorted[max(0, i - WINDOW): i + WINDOW + 1]
        cx_vals = [originales[f]['cx'] for f in window_frames if originales[f]['cx'] is not None]
        cy_vals = [originales[f]['cy'] for f in window_frames if originales[f]['cy'] is not None]

        cx_new = int(round(sum(cx_vals) / len(cx_vals)))
        cy_new = int(round(sum(cy_vals) / len(cy_vals)))

        cx_orig = originales[frame]['cx']
        cy_orig = originales[frame]['cy']

        cambio_x = cx_new - cx_orig
        cambio_y = cy_new - cy_orig

        detecciones[frame]['cx']     = cx_new
        detecciones[frame]['cy']     = cy_new
        detecciones[frame]['status'] = 'suavizado'

        signo_x = f"{cambio_x:+d}" if cambio_x != 0 else "  0"
        signo_y = f"{cambio_y:+d}" if cambio_y != 0 else "  0"
        print(f"  Frame {frame:<4}  cx: {cx_orig} → {cx_new} ({signo_x})   "
              f"cy: {cy_orig} → {cy_new} ({signo_y})")

# =================================================================
# 5. GUARDAR CSV FINAL
# =================================================================

print(f"\n{'─'*50}")
print("Guardando CSV...\n")

ok_count   = 0
fail_count = 0

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    # center_x / center_y son las coordenadas CORREGIDAS que consumen
    # batch_multicanal.py y npy_multicanal.py. Conservan nombre y posicion.
    # Las columnas _raw guardan el float original de YOLO, solo como
    # registro para comparar la serie cruda contra la suavizada.
    writer.writerow(['frame', 'center_x', 'center_y',
                     'center_x_raw', 'center_y_raw', 'conf', 'status'])

    for frame in frames_sorted:
        d = detecciones[frame]

        if d['cx'] is not None:
            cx_raw = f"{d['cx_raw']:.2f}" if d.get('cx_raw') is not None else ''
            cy_raw = f"{d['cy_raw']:.2f}" if d.get('cy_raw') is not None else ''
            writer.writerow([frame, d['cx'], d['cy'],
                             cx_raw, cy_raw, d['conf'] or '', d['status']])
            ok_count += 1
        else:
            writer.writerow([frame, '', '', '', '', '', d['status']])
            fail_count += 1

print(f"Frames guardados        : {ok_count}")
print(f"Frames sin detección    : {fail_count}")
print(f"Modo aplicado           : {SMOOTHING_MODE}")
print(f"CSV guardado en         : {OUTPUT_CSV}")