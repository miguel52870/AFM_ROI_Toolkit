"""
grafica_coordenadas.py — Visualizacion de coordenadas detectadas por YOLO
con la variacion de deteccion suavizada.

Entrada:  coordenadas_registro.csv  (generado por Calcular_Registro.py)
          Columnas: frame, center_x, center_y, center_x_raw,
                    center_y_raw, conf, status
          status: 'ok' | 'suavizado' | 'fallo_deteccion' | 'sin_png'
                  ('interpolado' se acepta por compatibilidad con CSV viejos)

Salida:   coordenadas_registro.png  (en la misma carpeta que el CSV)

NOTA sobre grafica_registro_crudo.py
------------------------------------
Los paneles 1 y 2 quedan cubiertos por ese script, que grafica las mismas
series suavizadas ademas de la deteccion cruda y la truncada. El aporte
propio de esta figura es el panel 3 —el drift acumulado respecto al frame
inicial— y los insets que situan el recorte de 80 px dentro de la imagen
completa. Ninguna otra figura da esas dos lecturas.

Uso:
    python grafica_coordenadas.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

# =================================================================
# CONFIGURACIÓN
# =================================================================

# =================================================================
# RAIZ DEL PROYECTO
# =================================================================
# Unica ruta que hay que cambiar al mover el proyecto o al usarlo en
# otra maquina. Todo lo demas se deriva de aqui.
BASE_DIR = Path(r'C:\Users\migue\Desktop\training_afm')

CSV_PATH   = BASE_DIR / 'Resultados' / 'registro' / 'coordenadas_registro.csv'
OUT_PNG    = CSV_PATH.parent / 'coordenadas_registro.png'

IMG_WIDTH  = 256   # ancho real de la imagen AFM en px (eje X)
IMG_HEIGHT = 128   # alto real de la imagen AFM en px  (eje Y)
CROP_HALF  = 40    # mitad del recorte (80 px / 2)
MARGIN_PX  = 2     # margen extra alrededor del rango de datos en los paneles 1 y 2

# Colores
COLOR_OK    = '#185FA5'
COLOR_FIXED = '#D85A30'
COLOR_LINE  = '#185FA5'

# =================================================================
# CARGA
# =================================================================

print(f"Leyendo: {CSV_PATH}")
if not CSV_PATH.exists():
    print("ERROR: archivo no encontrado.")
    sys.exit(1)

df = pd.read_csv(str(CSV_PATH))
df_valid = df[df['center_x'].notna()].copy()
df_valid['center_x'] = df_valid['center_x'].astype(float)
df_valid['center_y'] = df_valid['center_y'].astype(float)

STATUS_FIXED = {'interpolado', 'suavizado'}
mask_ok    = df_valid['status'] == 'ok'
mask_fixed = df_valid['status'].isin(STATUS_FIXED)

frames_all = df_valid['frame'].values
cx_all     = df_valid['center_x'].values
cy_all     = df_valid['center_y'].values

n_sin = len(df) - len(df_valid)
print(f"  Detecciones sin corregir : {mask_ok.sum()}")
print(f"  Frames suavizados        : {mask_fixed.sum()}")
if n_sin:
    print(f"  Frames sin deteccion     : {n_sin}  (excluidos de la figura)")

# =================================================================
# FIGURA — 3 paneles
# =================================================================

fig, axes = plt.subplots(3, 1, figsize=(12, 10),
                         gridspec_kw={'height_ratios': [1, 1, 1.1]})
fig.suptitle(
    'Coordenadas del Target Area detectadas por YOLO\n'
    'con suavizado de la variacion de deteccion',
    fontsize=13, fontweight='bold', y=0.99
)

# ------------------------------------------------------------------
# Función auxiliar para paneles 1 y 2
# ------------------------------------------------------------------
def plot_coord_panel(ax, vals, col, ylabel, title, img_dim):
    """
    Escala ajustada al rango visible de los datos + MARGIN_PX.
    Inset miniatura en la esquina superior derecha mostrando la posición
    del rango visible dentro de la imagen completa.
    """
    # ── serie principal ──────────────────────────────────────────
    ax.plot(frames_all, vals, color=COLOR_LINE, linewidth=1.2, zorder=1)
    ax.scatter(
        df_valid.loc[mask_ok, 'frame'],
        df_valid.loc[mask_ok, col],
        color=COLOR_OK, s=30, zorder=3, label='Detección YOLO válida'
    )
    if mask_fixed.any():
        ax.scatter(
            df_valid.loc[mask_fixed, 'frame'],
            df_valid.loc[mask_fixed, col],
            color=COLOR_FIXED, s=50, marker='D', zorder=4,
            label='Valor suavizado'
        )

    # ── escala ajustada al rango visible ────────────────────────
    y_min = vals.min() - MARGIN_PX
    y_max = vals.max() + MARGIN_PX
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(frames_all[0] - 0.5, frames_all[-1] + 0.5)

    # ── anotación del rango total de variación ───────────────────
    rango = vals.max() - vals.min()
    pct   = rango / img_dim * 100
    ax.annotate(
        f'Rango de variación: {rango:.0f} px  ({pct:.1f} % de {img_dim} px)',
        xy=(0.02, 0.06), xycoords='axes fraction',
        fontsize=8.5, color='#444',
        bbox=dict(boxstyle='round,pad=0.3', fc='#F5F4F0', ec='#BDBBB3', lw=0.8)
    )

    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis='both', alpha=0.2, linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9)

    # ── inset: barra de contexto (posición relativa en la imagen) ─
    # Rectángulo horizontal que representa la imagen completa,
    # con un segmento coloreado mostrando dónde está el rango visible.
    inset = ax.inset_axes([0.02, 0.72, 0.18, 0.22])   # [x, y, w, h] en fracción de ejes
    inset.set_xlim(0, img_dim)
    inset.set_ylim(0, 1)
    # imagen completa: fondo gris
    inset.barh(0.5, img_dim, left=0, height=0.6, color='#DEDAD4', align='center')
    # rango visible: color primario
    inset.barh(0.5, vals.max() - vals.min(), left=vals.min(),
               height=0.6, color=COLOR_OK, alpha=0.75, align='center')
    # Recorte de 80 px situado en la posicion MEDIA de la serie. Con deriva
    # el centro cambia frame a frame, asi que esta barra representa la
    # cobertura tipica, no la de un frame concreto.
    media = np.mean(vals)
    inset.barh(0.5, CROP_HALF * 2, left=media - CROP_HALF,
               height=0.6, color='#F5A623', alpha=0.5, align='center',
               label='Recorte 80 px (posicion media)')
    inset.set_xticks([0, img_dim // 2, img_dim])
    inset.set_xticklabels(['0', f'{img_dim//2}', f'{img_dim}'], fontsize=6)
    inset.set_yticks([])
    inset.set_title('Posición en imagen', fontsize=6.5, pad=2)
    inset.spines['top'].set_visible(False)
    inset.spines['right'].set_visible(False)
    inset.spines['left'].set_visible(False)

# ------------------------------------------------------------------
# Panel 1: center_x
# ------------------------------------------------------------------
plot_coord_panel(
    axes[0], cx_all, 'center_x',
    ylabel='Coordenada X  (px)',
    title='Center X — rango de variación a lo largo de la serie',
    img_dim=IMG_WIDTH
)

# ------------------------------------------------------------------
# Panel 2: center_y
# ------------------------------------------------------------------
plot_coord_panel(
    axes[1], cy_all, 'center_y',
    ylabel='Coordenada Y  (px)',
    title='Center Y — rango de variación a lo largo de la serie',
    img_dim=IMG_HEIGHT
)

# ------------------------------------------------------------------
# Panel 3: desplazamiento acumulado (drift)
# ------------------------------------------------------------------
ax3 = axes[2]

dx = cx_all - cx_all[0]
dy = cy_all - cy_all[0]

ax3.plot(frames_all, dx, 'o-', color=COLOR_OK,  lw=1.5, ms=4,
         label='Desplazamiento acumulado X')
ax3.plot(frames_all, dy, 's-', color='#1D9E75', lw=1.5, ms=4,
         label='Desplazamiento acumulado Y')
ax3.fill_between(frames_all, dx, 0, alpha=0.08, color=COLOR_OK)
ax3.fill_between(frames_all, dy, 0, alpha=0.08, color='#1D9E75')
ax3.axhline(0, color='k', lw=0.6, ls='--', alpha=0.4)

# Anotaciones de drift máximo
idx_max_x = np.argmax(np.abs(dx))
idx_max_y = np.argmax(np.abs(dy))
for idx, arr, color, offset in [
    (idx_max_x, dx, COLOR_OK,  ( 10,  6)),
    (idx_max_y, dy, '#1D9E75', ( 10, -16)),
]:
    ax3.annotate(
        f'Drift máx: {arr[idx]:+.0f} px',
        xy=(frames_all[idx], arr[idx]),
        xytext=offset, textcoords='offset points',
        fontsize=8, color=color,
        arrowprops=dict(arrowstyle='->', color=color, lw=0.8)
    )

ax3.set_xlim(frames_all[0] - 0.5, frames_all[-1] + 0.5)
ax3.set_xlabel('Número de frame', fontsize=10)
ax3.set_ylabel('Desplazamiento respecto\nal frame inicial (px)', fontsize=9)
ax3.set_title(
    'Desplazamiento acumulado del Target Area (drift) a lo largo de la serie',
    fontsize=10
)
ax3.legend(fontsize=8, loc='upper left', framealpha=0.9)
ax3.grid(axis='both', alpha=0.2, linewidth=0.6)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.tick_params(labelsize=9)

# ------------------------------------------------------------------
plt.tight_layout()
plt.savefig(str(OUT_PNG), dpi=180, bbox_inches='tight', facecolor='white')
print(f"\nFigura guardada: {OUT_PNG}")
plt.show()