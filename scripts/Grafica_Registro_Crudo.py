"""
grafica_diff_recorte.py — Evolucion del cambio entre frames: imagen
completa contra el recorte del Target Area.

POR QUE ESTA FIGURA
-------------------
afm_diagnostics.py grafica |f_n - f_{n-1}| sobre la imagen completa de
256x128 px. Pero el modelo nunca ve esa imagen: recibe recortes de
80x80 px centrados en el Target Area, es decir menos del 20 % de los
pixeles originales.

La tesis sostiene que C2_diff senala las zonas de switching activo y que
por eso es un canal de entrada util. Ese argumento se sostiene sobre el
recorte, no sobre la imagen completa. Esta figura contrasta ambas series:

  - Si los picos coinciden -> el switching ocurre dentro del Target Area
    y el canal de diff aporta la senal que se le atribuye.
  - Si el recorte se aplana donde la imagen completa tiene picos -> esos
    eventos ocurrieron fuera de la zona de interes.
  - Si aparecen picos nuevos en el recorte -> hay eventos locales que el
    promedio global diluia.

NORMALIZACION
-------------
npy_multicanal.py solo recorta: hace slicing del array, sin renormalizar.
Los valores del recorte son por tanto los mismos que en la imagen completa,
solo que de una subregion, y las dos series SI son comparables en valor
absoluto. El CSV guarda esas magnitudes crudas.

Aun asi la figura normaliza cada serie a su propio maximo. La razon no es
que las escalas difieran, sino que la pregunta relevante es si los PICOS
COINCIDEN EN EL TIEMPO: normalizar iguala la altura de ambas curvas y deja
ver el solapamiento de los perfiles, que de otro modo quedaria oculto tras
la diferencia de amplitud entre una region y la superficie entera.

USO
---
    python grafica_diff_recorte.py

Salida (junto al script):
    diff_completa_vs_recorte.png
    diff_completa_vs_recorte.csv
"""

from pathlib import Path
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =================================================================
# CONFIGURACION — ajustar rutas
# =================================================================

# Cada repo guarda como entrada la salida del anterior, no lee del repo
# vecino. Por eso ambas rutas apuntan a carpetas 'data/' distintas.
#
# Diffs de la imagen completa (256x128) — salida de afm_preprocess.py
# copiada como entrada de AFM_ROI_Toolkit
# =================================================================
# RAIZ DEL PROYECTO
# =================================================================
# Unica ruta que hay que cambiar al mover el proyecto o al usarlo en
# otra maquina. Todo lo demas se deriva de aqui.
BASE_DIR = Path(r'C:\Users\migue\Desktop\training_afm')

DIR_COMPLETA = BASE_DIR / 'data' / 'diff' / 'npy'

# Diffs recortados 80x80 al Target Area — salida de npy_multicanal.py
# copiada como entrada de AFM_PredictiveModel
DIR_RECORTE  = BASE_DIR / 'Resultados' / 'numpy_recortes' / 'diff'

OUT_DIR = BASE_DIR / 'Resultados' / 'registro'
OUT_PNG = OUT_DIR / 'diff_completa_vs_recorte.png'
OUT_CSV = OUT_DIR / 'diff_completa_vs_recorte.csv'

COL_COMPLETA = '#888780'
COL_RECORTE  = '#D85A30'

# =================================================================


def num_frame(path):
    """Extrae el numero de frame del nombre del archivo."""
    m = re.search(r'_(\d+)_Canal', path.stem)
    return int(m.group(1)) if m else None


def serie(directorio, etiqueta):
    """Devuelve {frame: (media, maximo)} de todos los NPY del directorio."""
    if not directorio.exists():
        print(f"  AVISO: no existe {etiqueta}: {directorio}")
        return {}
    datos = {}
    for f in sorted(directorio.glob('*.npy')):
        fn = num_frame(f)
        if fn is None:
            continue
        arr = np.load(str(f)).astype(np.float64)
        datos[fn] = (float(arr.mean()), float(arr.max()))
    print(f"  {etiqueta:<18}: {len(datos)} frames"
          + (f"  ({min(datos)}-{max(datos)})" if datos else ""))
    return datos


def norm_max(vals):
    """Escala a [0,1] dividiendo por el maximo. Preserva el perfil."""
    mx = max(vals) if vals else 0.0
    return [v / mx for v in vals] if mx > 1e-12 else list(vals)


def main():
    print("Leyendo diffs...")
    comp = serie(DIR_COMPLETA, 'imagen completa')
    rec  = serie(DIR_RECORTE,  'recorte 80x80')

    if not comp or not rec:
        print("\nFaltan datos. Revisar DIR_COMPLETA y DIR_RECORTE.")
        sys.exit(1)

    frames = sorted(set(comp) & set(rec))
    solo_c = sorted(set(comp) - set(rec))
    solo_r = sorted(set(rec) - set(comp))
    if solo_c: print(f"  Solo en imagen completa: {solo_c}")
    if solo_r: print(f"  Solo en recorte        : {solo_r}")
    print(f"  Frames comunes         : {len(frames)}")

    max_c = [comp[f][1] for f in frames]
    max_r = [rec[f][1]  for f in frames]
    med_c = [comp[f][0] for f in frames]
    med_r = [rec[f][0]  for f in frames]

    nmax_c, nmax_r = norm_max(max_c), norm_max(max_r)
    nmed_c, nmed_r = norm_max(med_c), norm_max(med_r)

    # Correlacion de Pearson entre ambos perfiles de maximo
    r = float(np.corrcoef(nmax_c, nmax_r)[0, 1]) if len(frames) > 2 else float('nan')

    # ---------------- Figura ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        'Evolucion del cambio entre frames — Canal 2\n'
        'Imagen completa (256x128) vs. recorte del Target Area (80x80)',
        fontsize=12, fontweight='bold')

    ax = axes[0]
    ax.plot(frames, nmax_c, 'o-', ms=4, lw=1.4, color=COL_COMPLETA,
            label='Imagen completa 256x128')
    ax.plot(frames, nmax_r, 's-', ms=4, lw=1.8, color=COL_RECORTE,
            label='Recorte Target Area 80x80')
    ax.fill_between(frames, nmax_c, nmax_r, alpha=0.12, color=COL_RECORTE)
    ax.set_ylabel('|delta| maximo\n(normalizado a su maximo)', fontsize=9)
    ax.set_title(f'Cambio maximo por frame   ·   correlacion de perfiles r = {r:.3f}',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    ax = axes[1]
    ancho = 0.4
    ax.bar([f - ancho/2 for f in frames], nmed_c, width=ancho,
           color=COL_COMPLETA, alpha=0.75, label='Imagen completa')
    ax.bar([f + ancho/2 for f in frames], nmed_r, width=ancho,
           color=COL_RECORTE,  alpha=0.75, label='Recorte Target Area')
    ax.set_ylabel('|delta| medio\n(normalizado a su maximo)', fontsize=9)
    ax.set_xlabel('Frame (n)', fontsize=10)
    ax.set_title('Cambio medio por frame', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(OUT_PNG), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()

    # ---------------- CSV con magnitudes crudas ----------------
    with open(str(OUT_CSV), 'w', encoding='utf-8') as f:
        f.write('frame,max_completa,max_recorte,media_completa,media_recorte,'
                'max_completa_norm,max_recorte_norm\n')
        for i, fr in enumerate(frames):
            f.write(f'{fr},{max_c[i]:.6f},{max_r[i]:.6f},'
                    f'{med_c[i]:.6f},{med_r[i]:.6f},'
                    f'{nmax_c[i]:.6f},{nmax_r[i]:.6f}\n')

    # ---------------- Lectura del resultado ----------------
    print(f"\nFigura : {OUT_PNG}")
    print(f"CSV    : {OUT_CSV}")
    print(f"\nCorrelacion entre perfiles (maximo): r = {r:.3f}")
    if r > 0.7:
        print("  Los picos coinciden: el switching ocurre dentro del Target Area.")
        print("  Respalda el uso de C2_diff como senal de switching en el recorte.")
    elif r > 0.4:
        print("  Coincidencia parcial: parte del switching ocurre fuera del recorte.")
        print("  Conviene matizar la afirmacion en el Cap. 4.")
    else:
        print("  Los perfiles difieren: los picos de la imagen completa NO se")
        print("  reflejan en el recorte. La grafica global no sirve como evidencia")
        print("  de switching en la zona de interes.")

    top_c = sorted(frames, key=lambda f: comp[f][1], reverse=True)[:3]
    top_r = sorted(frames, key=lambda f: rec[f][1],  reverse=True)[:3]
    print(f"\nFrames con mayor cambio")
    print(f"  imagen completa : {top_c}")
    print(f"  recorte         : {top_r}")
    coinciden = sorted(set(top_c) & set(top_r))
    if coinciden:
        print(f"  coinciden       : {coinciden}")

    # Las dos series estan en la misma escala (el recorte no renormaliza),
    # asi que comparar sus magnitudes dice si el Target Area concentra mas
    # o menos actividad que el promedio de la superficie escaneada.
    prom_c = sum(med_c) / len(med_c)
    prom_r = sum(med_r) / len(med_r)
    print(f"\nCambio medio promedio de la serie (magnitud cruda)")
    print(f"  imagen completa : {prom_c:.4f}")
    print(f"  recorte         : {prom_r:.4f}")
    if prom_r > prom_c:
        print(f"  El recorte concentra {(prom_r/prom_c - 1)*100:.0f} % mas cambio que")
        print("  el promedio de la superficie: el Target Area no solo conserva la")
        print("  senal de switching, la concentra.")
    else:
        print(f"  El recorte registra {(1 - prom_r/prom_c)*100:.0f} % menos cambio que")
        print("  el promedio de la superficie.")


if __name__ == '__main__':
    main()