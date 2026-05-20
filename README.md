# AFM_ROI_Toolkit

Detección automática de ROI y recorte multicanal alineado (PNG + NPY) para series de imágenes AFM de BiFeO₃.

Se inserta entre el módulo de preprocesamiento (`AFM_Preprocessing`) y el entrenamiento de los modelos predictivos (`modelo_predictivo`).

**Tesis:** *Estudio de la Evolución de los Dominios Ferroeléctricos al Switching
Usando Deep Learning para Aplicación de Memorias de Estado Sólido*
**Instituto Tecnológico de Querétaro — Maestría en Ciencia de Datos**
**Autor:** Miguel Angel Castro Medina

---

## Descripción general

AFM_ROI_Toolkit resuelve el problema de alineación espacial entre frames de una serie temporal AFM. Dado que el microscopio introduce deriva mecánica entre ciclos de medición, el Target Area (la zona ferroeléctrica de interés) se desplaza ligeramente de un frame al siguiente. Este toolkit:

1. Entrena un modelo YOLO para detectar el Target Area en imágenes del Canal 1 preprocesado
2. Ejecuta YOLO frame a frame para obtener las coordenadas del centro del Target Area
3. Corrige outliers de detección para garantizar continuidad espacial
4. Genera recortes PNG y NPY alineados para los 5 tipos de datos (C1, C2, C3, diff, mask)

Los recortes resultantes son el input directo de los modelos predictivos.

---

## Contenido del proyecto

```
AFM_ROI_Toolkit/
├── scripts/
│   ├── Calcular_Registro.py     # Etapa 1 — detección YOLO + corrección de outliers → CSV
│   ├── batch_multicanal.py      # Etapa 2a — recortes PNG desde CSV
│   ├── npy_multicanal.py        # Etapa 2b — recortes NPY desde CSV
│   ├── training_yolo.py         # Entrenamiento del modelo YOLO
│   └── data.yaml                # Configuración del dataset YOLO
├── data/
│   ├── images/
│   │   ├── Train/               # PNGs Canal 1 prep para entrenamiento YOLO
│   │   ├── Val/                 # PNGs Canal 1 prep para validación YOLO
│   │   └── Test/                # PNGs Canal 1, 2, 3 prep para inferencia
│   ├── labels/                  # Anotaciones YOLO (.txt) del Canal 1
│   ├── numpy_arrays/            # NPYs preprocesados por canal
│   ├── diff/                    # NPYs y PNGs de diferencia entre frames (C2_diff)
│   └── mask/                    # NPYs y PNGs de máscaras Otsu (C3_mask)
├── Resultados/
│   ├── registro/
│   │   └── coordenadas_registro.csv   # Salida de Calcular_Registro.py
│   ├── 3_canales/               # Recortes PNG por canal
│   │   ├── canal_1/
│   │   ├── canal_2/
│   │   ├── canal_3/
│   │   ├── diff/
│   │   └── mask/
│   └── numpy_recortes/          # Recortes NPY por canal
│       ├── canal_1/
│       ├── canal_2/
│       ├── canal_3/
│       ├── diff/
│       └── mask/
├── runs/detect/                 # Salidas de entrenamiento YOLO
│   └── Target_Area/weights/best.pt
├── yolo11n.pt                   # Pesos base YOLO (preentrenados)
├── requirements.txt
└── README.md
```

---

## Instalación

**Python recomendado:** 3.10.11

```powershell
# 1. Crear y activar entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. Instalar PyTorch (elegir según CUDA disponible)
# CPU (sin GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8 (RTX 3050):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar resto de dependencias
pip install -r requirements.txt

# 4. Verificar
python -c "import torch; print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

> Si no conoces tu versión de CUDA, ejecuta `nvidia-smi` en PowerShell.

---

## Flujo de trabajo completo

```
AFM_Preprocessing
  png_procesado/prep/    ──► data/images/Test/     (Canal 1, 2, 3 prep)
  npy_procesado/prep/    ──► data/numpy_arrays/    (Canal 1, 2, 3 prep)
  npy_procesado/diff/    ──► data/diff/            (C2_diff)
  npy_procesado/mask/    ──► data/mask/            (C3_mask)
          │
          ▼
  [Una sola vez] training_yolo.py
          │  Etiqueta Canal 1 prep con LabelImg → data/labels/
          │  Entrena YOLO → runs/detect/Target_Area/weights/best.pt
          │
          ▼
  Calcular_Registro.py
          │  YOLO detecta Target Area frame a frame
          │  Corrige outliers de detección
          │  → Resultados/registro/coordenadas_registro.csv
          │
          ├──► batch_multicanal.py  → Resultados/3_canales/  (PNG recortados)
          └──► npy_multicanal.py    → Resultados/numpy_recortes/  (NPY recortados)
                    │
                    ▼
            modelo_predictivo/data/
              canal_2/   ← C2_prep recortado
              diff/      ← C2_diff recortado
              canal_3/   ← C3_prep recortado
              mask/      ← C3_mask recortada
```

---

## Etapa 0 — Entrenamiento YOLO (una sola vez)

El modelo YOLO solo necesita entrenarse una vez. Si ya existe `runs/detect/Target_Area/weights/best.pt`, omitir esta etapa.

### Preparación del dataset

1. Copiar los PNGs del Canal 1 preprocesado a `data/images/Train/` y `data/images/Val/`
2. Etiquetar con **LabelImg** — clase única: `Target_Area`
3. Guardar anotaciones en `data/labels/` en formato YOLO (`.txt`)

> Solo se etiqueta el Canal 1 porque es el canal de topografía, más estable visualmente y con menor ruido que los canales PFM.

### Configurar `data.yaml`

```yaml
path: ./data
train: images/Train
val:   images/Val

nc: 1
names: ['Target_Area']
```

### Entrenar

```powershell
python scripts/training_yolo.py
```

El modelo entrenado quedará en `runs/detect/Target_Area/weights/best.pt`.

---

## Etapa 1 — Calcular_Registro.py

Ejecuta YOLO sobre todos los frames del Canal 1 preprocesado, extrae las coordenadas del centro del Target Area y corrige outliers de detección para garantizar continuidad espacial entre frames.

```powershell
python scripts/Calcular_Registro.py
```

### Parámetros principales

```python
MODEL_PATH         = 'runs/detect/Target_Area/weights/best.pt'
TEST_DIR           = 'data/images/Test'   # PNGs Canal 1 prep
OUTPUT_CSV         = 'Resultados/registro/coordenadas_registro.csv'
CONFIDENCE_THRESHOLD = 0.85
IMG_WIDTH          = 256
IMG_HEIGHT         = 128

# Corrección de outliers
SMOOTHING_MODE       = 'outlier'    # 'outlier' | 'moving_avg'
OUTLIER_THRESHOLD_X  = 0.5         # desviación máxima en px antes de corregir
OUTLIER_THRESHOLD_Y  = 0.5
OUTLIER_WINDOW       = 4           # frames vecinos usados para interpolar
```

**Modos de suavizado:**
- `'outlier'` — preserva las detecciones YOLO válidas y solo corrige las anómalas interpolando desde los frames vecinos. Recomendado para series con buena detección.
- `'moving_avg'` — suaviza toda la serie con una media móvil. Útil si la detección es ruidosa en general.

### Salida: `coordenadas_registro.csv`

```
frame, center_x, center_y, fuente
21,    128,      64,        yolo
22,    129,      63,        yolo
23,    131,      65,        interpolado   ← outlier corregido
...
```

---

## Etapa 2a — batch_multicanal.py

Lee `coordenadas_registro.csv` y genera recortes PNG alineados para los 5 tipos de datos.

```powershell
python scripts/batch_multicanal.py
```

### Parámetros principales

```python
IMAGE_DIR    = 'data/images/Test'      # PNGs Canal 1, 2, 3 prep
DIFF_PNG_DIR = 'data/diff/png'         # PNGs C2_diff
MASK_PNG_DIR = 'data/mask/png'         # PNGs C3_mask
INPUT_CSV    = 'Resultados/registro/coordenadas_registro.csv'

# Modo de recorte
CROP_MODE    = 'cuadrado'    # 'cuadrado' | 'rectangular'
CROP_SIZE    = 80            # usado en modo cuadrado
CROP_WIDTH   = 80            # usado en modo rectangular (divisible por 32)
CROP_HEIGHT  = 64            # usado en modo rectangular (divisible por 32)
```

### Salida: `Resultados/3_canales/`

```
3_canales/
  canal_1/   bifeo_training_N_Canal_1_prep_recorte_80px.png
  canal_2/   bifeo_training_N_Canal_2_prep_recorte_80px.png
  canal_3/   bifeo_training_N_Canal_3_prep_recorte_80px.png
  diff/      bifeo_training_N_Canal_2_diff_recorte_80px.png
  mask/      bifeo_training_N_Canal_3_mask_recorte_80px.png
```

---

## Etapa 2b — npy_multicanal.py

Usa las mismas coordenadas de `coordenadas_registro.csv` para generar recortes NPY alineados.

```powershell
python scripts/npy_multicanal.py
```

### Parámetros principales

Idénticos a `batch_multicanal.py` pero apuntando a las carpetas NPY:

```python
NUMPY_DIR    = 'data/numpy_arrays/'    # NPYs Canal 1, 2, 3 prep
DIFF_NPY_DIR = 'data/diff/npy'         # NPYs C2_diff
MASK_NPY_DIR = 'data/mask/npy'         # NPYs C3_mask
```

### Salida: `Resultados/numpy_recortes/`

```
numpy_recortes/
  canal_1/   bifeo_training_N_Canal_1_prep_recorte_80px.npy
  canal_2/   bifeo_training_N_Canal_2_prep_recorte_80px.npy
  canal_3/   bifeo_training_N_Canal_3_prep_recorte_80px.npy
  diff/      bifeo_training_N_Canal_2_diff_recorte_80px.npy
  mask/      bifeo_training_N_Canal_3_mask_recorte_80px.npy
```

---

## Recortes cuadrados vs. rectangulares

Ambos scripts soportan dos modos de recorte configurables con `CROP_MODE`:

| Modo | Parámetros | Requisito |
|---|---|---|
| `'cuadrado'` | `CROP_SIZE = 80` | Cualquier valor |
| `'rectangular'` | `CROP_WIDTH = 80, CROP_HEIGHT = 64` | Ambos divisibles por 32 |

> **Por qué divisible por 32:** la arquitectura U-Net con encoder EfficientNet-B0 realiza 5 niveles de downsampling (×32 total). Dimensiones no divisibles por 32 causarán error en los modelos predictivos.

El sufijo del nombre de archivo cambia automáticamente según el modo:
- Cuadrado → `_recorte_80px`
- Rectangular → `_recorte_80x64px`

---

## Indexación posicional

Los scripts de recorte generan archivos ordenados **alfabéticamente** por nombre. Los modelos predictivos acceden a ellos por **posición** en esa lista ordenada (base 1), independientemente del número en el nombre del archivo.

Esto significa que el orden cronológico de los frames queda preservado siempre que los nombres de archivo sigan el patrón `bifeo_training_N_...` donde N es el número de frame consecutivo.

---

## Notas técnicas

**Ruta del modelo YOLO:** verificar que `MODEL_PATH` en `Calcular_Registro.py` apunta al `best.pt` correcto antes de ejecutar.

**Confianza de detección:** `CONFIDENCE_THRESHOLD = 0.85` es el valor usado en producción. Si YOLO falla en muchos frames, reducir a 0.75 y revisar el diagnóstico de detección.

**Frames sin diff:** el frame 21 (primero de la serie) no tiene `_diff` porque no existe un frame anterior. Los scripts lo omiten sin generar error — esto es comportamiento esperado.

**Compatibilidad de nombres:** los PNGs y NPYs deben seguir el patrón `bifeo_training_N_Canal_C_prep` para que los scripts puedan emparejarlos correctamente entre canales.

---

## Licencia

Proyecto académico — Instituto Tecnológico de Querétaro. Todos los derechos reservados.