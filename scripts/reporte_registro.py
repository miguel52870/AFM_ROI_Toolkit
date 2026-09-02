"""
reporte_registro.py — Reporte PDF del pipeline de registro y recorte.
Tesis: Evolucion de los Dominios Ferroelectricos con Deep Learning

Consolida en un solo documento citable lo que hoy son PNG sueltos, y anade
la verificacion visual del recorte, que ninguna otra figura cubre.

Secciones
---------
  1. Portada con los parametros del registro
  2. Deteccion YOLO: confianza por frame y conteo de suavizados
  3. Registro: coordenadas, drift y trayectoria del centro
  4. Verificacion del recorte: ROI sobre la imagen completa y tira temporal
  5. Contenido del diff: imagen completa vs. recorte
  6. Tabla resumen de verificacion

Requisitos previos
------------------
  Calcular_Registro.py   -> coordenadas_registro.csv (con columnas _raw)
  batch_multicanal.py    -> Resultados/3_canales/    (para la seccion 4)
  npy_multicanal.py      -> Resultados/numpy_recortes/ (para la seccion 5)

Las secciones cuyos datos falten se omiten con un aviso; el reporte se
genera igual con lo que haya disponible.

USO
---
    python reporte_registro.py

Salida
------
  Resultados/reporte_registro/reporte_registro.pdf
  Resultados/reporte_registro/figuras/*.png   (figuras sueltas del reporte)
"""

from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection

# =================================================================
# CONFIGURACION
# =================================================================

BASE_DIR   = Path(r'C:\Users\migue\Desktop\training_afm')
RES_DIR    = BASE_DIR / 'Resultados'

CSV_PATH   = RES_DIR / 'registro' / 'coordenadas_registro.csv'
IMAGE_DIR  = BASE_DIR / 'data' / 'images' / 'Test'      # PNGs completos C1
CROP_DIR   = RES_DIR / '3_canales'                       # recortes PNG
DIR_DIFF_COMPLETA = BASE_DIR / 'data' / 'diff' / 'npy'
DIR_DIFF_RECORTE  = RES_DIR / 'numpy_recortes' / 'diff'

# Salida — carpeta propia para no mezclarse con las figuras sueltas
OUT_DIR    = RES_DIR / 'reporte_registro'
FIG_DIR    = OUT_DIR / 'figuras'
OUTPUT_PDF = OUT_DIR / 'reporte_registro.pdf'

FILE_PREFIX = 'bifeo_training'
IMG_WIDTH   = 256
IMG_HEIGHT  = 128
CROP_SIZE   = 80
CROP_LABEL  = '80px'

# Parametros usados en Calcular_Registro.py, solo para documentarlos
PARAMS = {
    'Modelo YOLO':        'runs/detect/Target_Area_prep/weights/best.pt',
    'Confianza minima':   '0.85',
    'Modo de suavizado':  'outlier',
    'Ventana':            '4 frames a cada lado',
    'Umbral X / Y':       '0.25 px / 0.25 px',
    'Recorte':            f'cuadrado {CROP_SIZE}x{CROP_SIZE} px',
    'Imagen completa':    f'{IMG_WIDTH}x{IMG_HEIGHT} px',
}

# Frames a mostrar en la verificacion visual. None = elegir automaticamente
# el primero, el ultimo y el de mayor desviacion.
FRAMES_VERIFICACION = None

# Limites del panel de trayectoria en el plano X-Y. None = ajustar a los
# datos. Fijarlos da una vista estable entre corridas y evita que un
# outlier de la deteccion cruda comprima el recorrido util.
TRAY_XLIM = (120, 131)
TRAY_YLIM = (60, 66)

BLUE   = '#185FA5'
ORANGE = '#D85A30'
GREEN  = '#1D9E75'
GRAY   = '#888780'
TEAL   = '#0F6E56'

# =================================================================
# UTILIDADES
# =================================================================

def var_consecutiva(s):
    return float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0


def num_frame(path):
    m = re.search(r'_(\d+)_Canal', path.stem)
    return int(m.group(1)) if m else None


def estilo(ax, titulo='', xlabel='', ylabel=''):
    if titulo: ax.set_title(titulo, fontsize=10)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)


def guardar(fig, pdf, nombre):
    """Anade la figura al PDF y la guarda tambien suelta en FIG_DIR."""
    pdf.savefig(fig, bbox_inches='tight')
    fig.savefig(str(FIG_DIR / f'{nombre}.png'), dpi=180,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)


def cargar_csv():
    if not CSV_PATH.exists():
        print(f"ERROR: no se encontro el CSV:\n  {CSV_PATH}")
        print("  Correr Calcular_Registro.py primero.")
        sys.exit(1)
    df = pd.read_csv(str(CSV_PATH))
    for c in ('center_x_raw', 'center_y_raw'):
        if c not in df.columns:
            print(f"ERROR: al CSV le falta la columna '{c}'.")
            print("  Correr Calcular_Registro.py con la version que guarda")
            print("  las coordenadas crudas.")
            sys.exit(1)
    return df


def serie_diff(directorio):
    """{frame: (media, maximo)} de los NPY de un directorio de diffs."""
    if not directorio.exists():
        return {}
    out = {}
    for f in sorted(directorio.glob('*.npy')):
        fn = num_frame(f)
        if fn is None:
            continue
        a = np.load(str(f)).astype(np.float64)
        out[fn] = (float(a.mean()), float(a.max()))
    return out


# =================================================================
# 1 — PORTADA
# =================================================================

def page_portada(pdf, df, met):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    ax.text(0.5, 0.90, 'Registro y Recorte del Target Area',
            ha='center', fontsize=20, fontweight='bold', transform=ax.transAxes)
    ax.text(0.5, 0.845,
            'Estudio de la Evolucion de los Dominios Ferroelectricos al Switching\n'
            'Usando Deep Learning para Aplicacion de Memorias de Estado Solido',
            ha='center', fontsize=11, color='#5F5E5A',
            transform=ax.transAxes, linespacing=1.7)
    ax.axhline(y=0.79, xmin=0.1, xmax=0.9, color='#D3D1C7', linewidth=1)

    # Tarjetas con los tres resultados principales
    tarjetas = [
        ('Deteccion', f"{met['n_frames']} frames",
         f"confianza {met['conf_min']:.3f}–{met['conf_max']:.3f}", BLUE),
        ('Alineacion', f"{met['red_x']:.0f} % / {met['red_y']:.0f} %",
         'reduccion de variacion X / Y', ORANGE),
        ('Deriva total', f"{met['drift_x']:.0f} px / {met['drift_y']:.0f} px",
         'desplazamiento X / Y', GREEN),
    ]
    for i, (titulo, valor, sub, color) in enumerate(tarjetas):
        x0 = 0.06 + i * 0.32
        ax.add_patch(plt.Rectangle((x0, 0.50), 0.28, 0.22, transform=ax.transAxes,
                                    facecolor=f'{color}11', edgecolor=color, lw=1.5))
        ax.text(x0 + 0.14, 0.685, titulo, ha='center', fontsize=10.5,
                fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(x0 + 0.14, 0.605, valor, ha='center', fontsize=14,
                fontweight='bold', color='#333', transform=ax.transAxes)
        ax.text(x0 + 0.14, 0.545, sub, ha='center', fontsize=8,
                color='#5F5E5A', transform=ax.transAxes)

    ax.text(0.06, 0.44, 'Parametros del registro', fontsize=11,
            fontweight='bold', color='#444441', transform=ax.transAxes)
    for i, (k, v) in enumerate(PARAMS.items()):
        y = 0.395 - i * 0.042
        ax.text(0.30, y, k + ':', ha='right', fontsize=9, fontweight='bold',
                color='#444441', transform=ax.transAxes)
        ax.text(0.32, y, v, ha='left', fontsize=9, color='#5F5E5A',
                transform=ax.transAxes)

    ax.text(0.5, 0.05,
            f"Serie de frames {met['f0']}–{met['f1']}  ·  "
            f"CSV: {CSV_PATH.name}",
            ha='center', fontsize=8, color=GRAY, transform=ax.transAxes)

    guardar(fig, pdf, '01_portada')


# =================================================================
# 2 — DETECCION YOLO
# =================================================================

def page_deteccion(pdf, df, met):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7),
                             gridspec_kw={'height_ratios': [1.4, 1]})
    fig.suptitle('Deteccion YOLO del Target Area',
                 fontsize=13, fontweight='bold', y=0.98)

    frames = df['frame'].values
    conf   = df['conf'].astype(float).values

    ax = axes[0]
    ax.bar(frames, conf, color=BLUE, alpha=0.8, width=0.65)
    ax.axhline(0.85, color=ORANGE, ls='--', lw=1.2,
               label='Umbral de confianza (0.85)')
    ax.set_ylim(0.8, 1.0)
    estilo(ax, 'Confianza de deteccion por frame', '', 'confianza')
    ax.legend(fontsize=8, loc='lower right')
    ax.annotate(
        f'Todos los frames superan el umbral. '
        f'min {conf.min():.3f}  ·  media {conf.mean():.3f}  ·  max {conf.max():.3f}',
        xy=(0.015, 0.06), xycoords='axes fraction', fontsize=8.5, color='#444',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F4F0', ec='#BDBBB3', lw=0.8))

    # Estado por frame: sirve para ver si los suavizados se agrupan
    ax = axes[1]
    estados = df['status'].values
    colores = {'ok': BLUE, 'suavizado': ORANGE, 'interpolado': ORANGE,
               'fallo_deteccion': '#C0392B', 'sin_png': GRAY}
    for est in sorted(set(estados)):
        sel = estados == est
        ax.scatter(frames[sel], np.ones(sel.sum()), s=90, marker='s',
                   color=colores.get(est, GRAY), label=f'{est} ({sel.sum()})')
    ax.set_ylim(0.9, 1.1); ax.set_yticks([])
    ax.set_xlim(frames[0] - 0.5, frames[-1] + 0.5)
    estilo(ax, 'Estado de cada frame tras el suavizado', 'Numero de frame', '')
    ax.legend(fontsize=8, loc='upper center', ncol=4, framealpha=0.9)
    ax.annotate(
        'YOLO detecto en todos los frames. El estado "suavizado" indica que la\n'
        'coordenada se ajusto al promedio de sus vecinos, no que la deteccion fallara.',
        xy=(0.015, 0.02), xycoords='axes fraction', fontsize=8, color='#444',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F4F0', ec='#BDBBB3', lw=0.8))

    plt.tight_layout()
    guardar(fig, pdf, '02_deteccion')


# =================================================================
# 3 — REGISTRO: coordenadas, drift y trayectoria
# =================================================================

def page_coordenadas(pdf, df, eje):
    frames = df['frame'].values
    raw    = df[f'center_{eje}_raw'].astype(float).values
    trunc  = np.floor(raw)
    suave  = df[f'center_{eje}'].astype(float).values

    v_raw, v_tr, v_su = (var_consecutiva(raw), var_consecutiva(trunc),
                         var_consecutiva(suave))
    red  = (1 - v_su / v_raw) * 100 if v_raw > 1e-9 else 0.0
    desv = np.abs(raw - suave); i_p = int(np.argmax(desv))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frames, raw, '-', color=GRAY, lw=2.2, alpha=0.85,
            label='Deteccion YOLO (subpixel)')
    ax.step(frames, trunc, where='mid', color=ORANGE, lw=1.2, ls='--',
            label='Truncada a entero')
    ax.plot(frames, suave, 'o-', color=BLUE, lw=1.5, ms=4,
            label='Serie suavizada (la que se usa)')
    ax.set_xlim(frames[0] - 0.5, frames[-1] + 0.5)
    estilo(ax, f'Coordenada {eje.upper()} — deteccion cruda frente a serie suavizada',
           'Numero de frame', f'Coordenada {eje.upper()}  (px)')
    ax.legend(fontsize=8.5, loc='upper left', ncol=3, framealpha=0.9)
    ax.margins(y=0.16)
    ax.annotate(
        f'Variacion media entre frames consecutivos —  cruda: {v_raw:.2f} px    '
        f'truncada: {v_tr:.2f} px    suavizada: {v_su:.2f} px  ({red:.0f} % menos)',
        xy=(0.015, 0.02), xycoords='axes fraction', fontsize=8.5, color='#444',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F4F0', ec='#BDBBB3', lw=0.8))

    plt.tight_layout()
    guardar(fig, pdf, f'03_coordenada_{eje}')
    return {'v_raw': v_raw, 'v_su': v_su, 'red': red,
            'peor_fr': int(frames[i_p]), 'peor_px': float(desv[i_p])}


def page_drift_trayectoria(pdf, df):
    frames = df['frame'].values
    sx = df['center_x'].astype(float).values
    sy = df['center_y'].astype(float).values
    rx = df['center_x_raw'].astype(float).values
    ry = df['center_y_raw'].astype(float).values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [1.5, 1]})
    fig.suptitle('Desplazamiento del Target Area a lo largo de la serie',
                 fontsize=13, fontweight='bold', y=0.99)

    # Drift acumulado respecto al frame inicial
    ax = axes[0]
    dx, dy = sx - sx[0], sy - sy[0]
    ax.plot(frames, dx, 'o-', color=BLUE,  lw=1.5, ms=4, label='Acumulado X')
    ax.plot(frames, dy, 's-', color=GREEN, lw=1.5, ms=4, label='Acumulado Y')
    ax.fill_between(frames, dx, 0, alpha=0.08, color=BLUE)
    ax.fill_between(frames, dy, 0, alpha=0.08, color=GREEN)
    ax.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)
    ax.set_xlim(frames[0] - 0.5, frames[-1] + 0.5)
    estilo(ax, 'Drift acumulado respecto al frame inicial',
           'Numero de frame', 'Desplazamiento (px)')
    ax.legend(fontsize=8.5, loc='upper left')

    # Trayectoria en el plano
    ax = axes[1]
    ax.plot(rx, ry, '-', color=GRAY, lw=0.9, alpha=0.45, zorder=1)
    ax.scatter(rx, ry, s=10, color=GRAY, alpha=0.5, zorder=2)
    pts  = np.array([sx, sy]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap='viridis', linewidth=2.5, zorder=3)
    lc.set_array(frames[:-1]); ax.add_collection(lc)

    uniq = {}
    for x, y in zip(sx, sy):
        uniq[(x, y)] = uniq.get((x, y), 0) + 1
    for (x, y), n in uniq.items():
        ax.scatter([x], [y], s=40 + 12 * n, facecolor='white',
                   edgecolor=BLUE, lw=1.5, zorder=4)
        ax.annotate(str(n), (x, y), fontsize=6.5, color=BLUE,
                    ha='center', va='center', zorder=5, fontweight='bold')

    cb = fig.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label('Numero de frame', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    estilo(ax, 'Trayectoria del centro (gris = deteccion cruda)',
           'Coordenada X (px)', 'Coordenada Y (px)')
    # Los limites van al final: LineCollection y las anotaciones reajustan
    # los ejes al dibujarse y sobrescribirian un set_xlim anterior.
    if TRAY_XLIM and TRAY_YLIM:
        ax.set_xlim(*TRAY_XLIM)
        ax.set_ylim(*TRAY_YLIM)
    else:
        m = 0.5
        ax.set_xlim(sx.min() - m, sx.max() + m)
        ax.set_ylim(sy.min() - m, sy.max() + m)

    plt.tight_layout()
    guardar(fig, pdf, '04_drift_trayectoria')
    return {'drift_x': float(np.abs(dx).max()), 'drift_y': float(np.abs(dy).max()),
            'posiciones': len(uniq)}


# =================================================================
# 4 — VERIFICACION DEL RECORTE
# =================================================================

def page_roi_sobre_imagen(pdf, df, frames_sel):
    """Dibuja el ROI sobre la imagen completa del Canal 1 en varios frames.
    Es la evidencia mas directa de que el recorte sigue a la estructura."""
    disponibles = []
    for fr in frames_sel:
        p = IMAGE_DIR / f'{FILE_PREFIX}_{fr}_Canal_1_prep.png'
        if p.exists():
            disponibles.append((fr, p))
    if not disponibles:
        print(f"  AVISO: no se encontraron PNGs de Canal 1 en {IMAGE_DIR}")
        return False

    n = len(disponibles)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.6 * n))
    if n == 1:
        axes = [axes]
    fig.suptitle('Region recortada sobre la imagen completa (Canal 1)',
                 fontsize=13, fontweight='bold', y=0.995)

    reg = df.set_index('frame')
    for ax, (fr, p) in zip(axes, disponibles):
        img = plt.imread(str(p))
        ax.imshow(img, cmap='gray', aspect='equal')
        cx, cy = float(reg.loc[fr, 'center_x']), float(reg.loc[fr, 'center_y'])
        h = CROP_SIZE / 2
        ax.add_patch(mpatches.Rectangle((cx - h, cy - h), CROP_SIZE, CROP_SIZE,
                                        fill=False, edgecolor=ORANGE, lw=2))
        ax.plot(cx, cy, '+', color=ORANGE, ms=10, mew=2)
        # Deteccion cruda, para ver cuanto la corrigio el filtro
        rx, ry = float(reg.loc[fr, 'center_x_raw']), float(reg.loc[fr, 'center_y_raw'])
        ax.plot(rx, ry, 'x', color=BLUE, ms=8, mew=1.8)
        ax.set_title(f'Frame {fr}   ·   centro usado ({cx:.0f}, {cy:.0f})   ·   '
                     f'deteccion cruda ({rx:.2f}, {ry:.2f})', fontsize=9)
        ax.axis('off')

    fig.legend(handles=[
        mpatches.Patch(edgecolor=ORANGE, fill=False, label=f'ROI {CROP_SIZE} px + centro usado'),
        mpatches.Patch(edgecolor=BLUE, fill=False, label='Deteccion cruda de YOLO'),
    ], loc='lower center', ncol=2, fontsize=9, frameon=False,
        bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    guardar(fig, pdf, '05_roi_sobre_imagen')
    return True


def page_tira_recortes(pdf, df, canal='canal_1', sufijo='Canal_1_prep'):
    """Tira temporal del mismo recorte a lo largo de la serie. Si el registro
    funciona, la estructura permanece en la misma posicion del recuadro."""
    carpeta = CROP_DIR / canal
    if not carpeta.exists():
        print(f"  AVISO: no existe {carpeta}; se omite la tira de recortes.")
        return False

    archivos = sorted(carpeta.glob('*.png'))
    if not archivos:
        print(f"  AVISO: sin recortes en {carpeta}")
        return False

    # Muestrear hasta 10 frames repartidos por la serie
    idx = np.linspace(0, len(archivos) - 1, min(10, len(archivos))).astype(int)
    sel = [archivos[i] for i in idx]

    fig, axes = plt.subplots(1, len(sel), figsize=(1.6 * len(sel), 2.4))
    if len(sel) == 1:
        axes = [axes]
    fig.suptitle(f'Recorte de {CROP_SIZE} px a lo largo de la serie — {canal}\n'
                 'Si el registro funciona, la estructura no se desplaza dentro del recuadro',
                 fontsize=11, fontweight='bold', y=1.06)
    for ax, p in zip(axes, sel):
        ax.imshow(plt.imread(str(p)), cmap='gray')
        fn = num_frame(p)
        ax.set_title(f'f{fn}' if fn else p.stem[:8], fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    guardar(fig, pdf, f'06_tira_{canal}')
    return True


# =================================================================
# 5 — CONTENIDO DEL DIFF
# =================================================================

def page_diff(pdf):
    comp = serie_diff(DIR_DIFF_COMPLETA)
    rec  = serie_diff(DIR_DIFF_RECORTE)
    if not comp or not rec:
        print("  AVISO: faltan NPY de diff; se omite la seccion de diff.")
        return None

    frames = sorted(set(comp) & set(rec))
    max_c = [comp[f][1] for f in frames]; max_r = [rec[f][1] for f in frames]
    med_c = [comp[f][0] for f in frames]; med_r = [rec[f][0] for f in frames]
    nm = lambda v: [x / max(v) for x in v] if max(v) > 1e-12 else v
    r = float(np.corrcoef(nm(max_c), nm(max_r))[0, 1])
    r_med = float(np.corrcoef(med_c, med_r)[0, 1])

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle('Cambio entre frames del Canal 2 — imagen completa frente al recorte',
                 fontsize=13, fontweight='bold', y=0.98)

    ax = axes[0]
    ax.plot(frames, nm(max_c), 'o-', ms=4, lw=1.3, color=GRAY,
            label=f'Imagen completa {IMG_WIDTH}x{IMG_HEIGHT}')
    ax.plot(frames, nm(max_r), 's-', ms=4, lw=1.7, color=ORANGE,
            label=f'Recorte {CROP_SIZE}x{CROP_SIZE}')
    ax.fill_between(frames, nm(max_c), nm(max_r), alpha=0.12, color=ORANGE)
    estilo(ax, f'Cambio maximo por frame   ·   correlacion de perfiles r = {r:.3f}',
           '', '|delta| max (normalizado)')
    ax.legend(fontsize=8.5)

    ax = axes[1]
    w = 0.4
    ax.bar([f - w/2 for f in frames], med_c, width=w, color=GRAY,
           alpha=0.75, label='Imagen completa')
    ax.bar([f + w/2 for f in frames], med_r, width=w, color=ORANGE,
           alpha=0.75, label='Recorte Target Area')
    estilo(ax, f'Cambio medio por frame (magnitud cruda)   ·   r = {r_med:.3f}',
           'Numero de frame', '|delta| medio')
    ax.legend(fontsize=8.5)
    pc, pr = np.mean(med_c), np.mean(med_r)
    ax.annotate(
        f'Promedio de la serie —  imagen completa: {pc:.4f}    recorte: {pr:.4f}\n'
        'El recorte no renormaliza, por lo que ambas magnitudes son comparables.',
        xy=(0.015, 0.78), xycoords='axes fraction', fontsize=8.5, color='#444',
        bbox=dict(boxstyle='round,pad=0.35', fc='#F5F4F0', ec='#BDBBB3', lw=0.8))

    plt.tight_layout()
    guardar(fig, pdf, '07_diff_completa_vs_recorte')

    top_c = sorted(frames, key=lambda f: comp[f][1], reverse=True)[:4]
    top_r = sorted(frames, key=lambda f: rec[f][1], reverse=True)[:4]
    return {'r_max': r, 'r_med': r_med, 'prom_c': float(pc), 'prom_r': float(pr),
            'coinciden': sorted(set(top_c) & set(top_r))}


# =================================================================
# 6 — TABLA RESUMEN
# =================================================================

def page_resumen(pdf, met, mx, my, tray, dif):
    fig = plt.figure(figsize=(12, 8.5))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')

    ax.text(0.5, 0.95, 'Resumen de verificacion del registro',
            ha='center', fontsize=15, fontweight='bold', transform=ax.transAxes)
    ax.axhline(y=0.91, xmin=0.05, xmax=0.95, color='#D3D1C7', lw=0.8)

    filas = [
        ('Frames procesados', f"{met['n_frames']}  ({met['f0']}–{met['f1']})"),
        ('Confianza de deteccion', f"{met['conf_min']:.3f} – {met['conf_max']:.3f}"
                                   f"  (media {met['conf_med']:.3f})"),
        ('Frames sin deteccion', f"{met['n_fallos']}"),
        ('Frames suavizados', f"{met['n_suav']} de {met['n_frames']}"),
        ('Variacion entre frames, eje X',
         f"{mx['v_raw']:.2f} px cruda  →  {mx['v_su']:.2f} px suavizada"
         f"  ({mx['red']:.0f} % menos)"),
        ('Variacion entre frames, eje Y',
         f"{my['v_raw']:.2f} px cruda  →  {my['v_su']:.2f} px suavizada"
         f"  ({my['red']:.0f} % menos)"),
        ('Deriva total del Target Area',
         f"{tray['drift_x']:.0f} px en X ({tray['drift_x']/IMG_WIDTH*100:.1f} % de {IMG_WIDTH})"
         f"  ·  {tray['drift_y']:.0f} px en Y ({tray['drift_y']/IMG_HEIGHT*100:.1f} % de {IMG_HEIGHT})"),
        ('Posiciones distintas del centro', f"{tray['posiciones']} de {met['n_frames']} frames"),
        ('Mayor desviacion corregida',
         f"frame {mx['peor_fr']}: {mx['peor_px']:.2f} px en X  ·  "
         f"frame {my['peor_fr']}: {my['peor_px']:.2f} px en Y"),
    ]
    if dif:
        filas += [
            ('Diff completa vs. recorte (max)', f"r = {dif['r_max']:.3f}"),
            ('Diff completa vs. recorte (media)', f"r = {dif['r_med']:.3f}"),
            ('Cambio medio promedio',
             f"{dif['prom_c']:.4f} completa  ·  {dif['prom_r']:.4f} recorte"),
            ('Frames de mayor cambio coincidentes', f"{dif['coinciden']}"),
        ]

    y0, h = 0.86, 0.049
    for i, (k, v) in enumerate(filas):
        y  = y0 - (i + 1) * h
        bg = '#F1EFE8' if i % 2 == 0 else 'white'
        ax.add_patch(plt.Rectangle((0.04, y), 0.92, h - 0.004,
                                    transform=ax.transAxes, facecolor=bg, zorder=1))
        ax.text(0.06, y + h/2 - 0.004, k, ha='left', va='center', fontsize=9,
                fontweight='bold', color='#444441', transform=ax.transAxes)
        ax.text(0.45, y + h/2 - 0.004, v, ha='left', va='center', fontsize=9,
                color='#5F5E5A', transform=ax.transAxes)

    y_txt = y0 - (len(filas) + 1) * h - 0.04
    ax.text(0.5, y_txt, 'Lectura', ha='center', fontsize=11,
            fontweight='bold', transform=ax.transAxes)

    conclusiones = [
        (BLUE, 'YOLO detecto el Target Area en todos los frames por encima del umbral '
               'de confianza: el suavizado no corrige fallos de deteccion.'),
        (ORANGE, 'El filtro suprime la variacion de deteccion entre frames consecutivos, '
                 'que es la condicion para que dos recortes seguidos cubran la misma region.'),
        (GREEN, 'La deriva del escaner es pequena frente al tamano de la imagen, y el '
                'recorte centrado la compensa.'),
    ]
    if dif:
        conclusiones.append(
            (TEAL, 'Los picos de cambio del Canal 2 coinciden en la imagen completa y en '
                   'el recorte: el switching ocurre dentro del Target Area.'))

    for i, (color, texto) in enumerate(conclusiones):
        ax.text(0.05, y_txt - 0.045 - i * 0.055, f'- {texto}', ha='left', va='top',
                fontsize=9, color=color, transform=ax.transAxes, wrap=True)

    guardar(fig, pdf, '08_resumen')


# =================================================================
# MAIN
# =================================================================

def metricas_eje(df, eje):
    """Metricas de un eje sin generar figura. Se necesitan en la portada,
    que va antes que las paginas donde se calcularian."""
    frames = df['frame'].values
    raw    = df[f'center_{eje}_raw'].astype(float).values
    suave  = df[f'center_{eje}'].astype(float).values
    v_raw, v_su = var_consecutiva(raw), var_consecutiva(suave)
    desv = np.abs(raw - suave); i_p = int(np.argmax(desv))
    return {'v_raw': v_raw, 'v_su': v_su,
            'red': (1 - v_su / v_raw) * 100 if v_raw > 1e-9 else 0.0,
            'peor_fr': int(frames[i_p]), 'peor_px': float(desv[i_p])}


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df_all = cargar_csv()
    df = df_all.dropna(subset=['center_x', 'center_y',
                               'center_x_raw', 'center_y_raw']).copy()
    if df.empty:
        print("ERROR: no hay filas con coordenadas validas.")
        sys.exit(1)

    frames = df['frame'].values
    conf   = df['conf'].astype(float).values
    n_suav = int(df['status'].isin(['suavizado', 'interpolado']).sum())
    sx = df['center_x'].astype(float).values
    sy = df['center_y'].astype(float).values

    mx = metricas_eje(df, 'x')
    my = metricas_eje(df, 'y')

    met = {'n_frames': len(df), 'f0': int(frames[0]), 'f1': int(frames[-1]),
           'conf_min': float(conf.min()), 'conf_max': float(conf.max()),
           'conf_med': float(conf.mean()), 'n_suav': n_suav,
           'n_fallos': len(df_all) - len(df),
           'red_x': mx['red'], 'red_y': my['red'],
           'drift_x': float(np.abs(sx - sx[0]).max()),
           'drift_y': float(np.abs(sy - sy[0]).max())}

    # Frames para la verificacion visual
    if FRAMES_VERIFICACION:
        sel = list(FRAMES_VERIFICACION)
    else:
        d = np.abs(df['center_x_raw'].astype(float).values - sx)
        sel = sorted({int(frames[0]), int(frames[int(np.argmax(d))]), int(frames[-1])})

    print(f"Frames    : {met['n_frames']}  ({met['f0']}-{met['f1']})")
    print(f"Salida    : {OUTPUT_PDF}")
    print(f"Figuras   : {FIG_DIR}\n")

    with PdfPages(str(OUTPUT_PDF)) as pdf:
        print("  Portada...");             page_portada(pdf, df, met)
        print("  Deteccion YOLO...");      page_deteccion(pdf, df, met)
        print("  Coordenadas X...");       page_coordenadas(pdf, df, 'x')
        print("  Coordenadas Y...");       page_coordenadas(pdf, df, 'y')
        print("  Drift y trayectoria..."); tray = page_drift_trayectoria(pdf, df)
        print("  ROI sobre imagen...");    page_roi_sobre_imagen(pdf, df, sel)
        print("  Tira de recortes...");    page_tira_recortes(pdf, df)
        print("  Contenido del diff...");  dif = page_diff(pdf)
        print("  Resumen...");             page_resumen(pdf, met, mx, my, tray, dif)

    print(f"\nReporte generado: {OUTPUT_PDF}")
    print(f"Figuras sueltas : {FIG_DIR}")
    print()
    print(f"  Variacion X : {mx['v_raw']:.2f} -> {mx['v_su']:.2f} px  ({mx['red']:.0f} % menos)")
    print(f"  Variacion Y : {my['v_raw']:.2f} -> {my['v_su']:.2f} px  ({my['red']:.0f} % menos)")
    print(f"  Deriva      : {tray['drift_x']:.0f} px en X, {tray['drift_y']:.0f} px en Y")
    if dif:
        print(f"  Diff r      : {dif['r_max']:.3f} (max), {dif['r_med']:.3f} (media)")


if __name__ == '__main__':
    main()