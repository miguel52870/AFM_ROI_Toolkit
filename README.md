# AFM_ROI_Toolkit

Detección automática de ROI y recorte multicanal alineado (PNG + NPY) para series de imágenes AFM de BiFeO₃.

Se inserta entre el módulo de preprocesamiento (`AFM_Preprocessing`) y el entrenamiento de los modelos predictivos (`AFM_PredictiveModel`).

**Tesis:** *Estudio de la Evolución de los Dominios Ferroeléctricos al Switching
Usando Deep Learning para Aplicación de Memorias de Estado Sólido*
**Instituto Tecnológico de Querétaro — Maestría en Ciencias en Ingeniería**
**Autor:** Miguel Angel Castro Medina

---

## Qué resuelve este módulo

El microscopio introduce deriva mecánica entre ciclos de medición, de modo que el Target Area —la zona ferroeléctrica de interés— se desplaza ligeramente de un frame al siguiente. Si esa deriva no se corrige, dos recortes consecutivos cubrirían regiones distintas del material y el canal `C2_diff` mezclaría switching ferroeléctrico con desplazamiento del ROI.

El toolkit:

1. Entrena un modelo YOLO para detectar el Target Area en el Canal 1 preprocesado
2. Ejecuta YOLO frame a frame y obtiene las coordenadas del centro
3. Suaviza la variación de detección para que los recortes queden alineados entre sí
4. Genera recortes PNG y NPY para los 5 tipos de datos (C1, C2, C3, diff, mask)
5. Produce figuras y un reporte PDF que documentan la calidad del registro

Los recortes resultantes son el input directo de los modelos predictivos.

### Sobre el paso 3

YOLO detecta el Target Area en los 40 frames con confianza entre 0.889 y 0.947: no falla en ninguno. Lo que varía es la posición exacta del bounding box, que se mueve de un frame a otro aunque la estructura no lo haga.

El filtro de `Calcular_Registro.py` suprime esa variación conservando la deriva real del escáner. Sobre la serie de la tesis reduce la variación media entre frames consecutivos de 1.12 px a 0.08 px en X, y de 0.46 px a 0.08 px en Y.

Esto **no** es corrección de detecciones fallidas. Es alineación entre recortes consecutivos, que es la condición para que `C2_diff` mida switching y no movimiento del ROI.

---

## Contenido del proyecto

```
AFM_ROI_Toolkit/
├── scripts/
│   ├── training_yolo.py            # Etapa 0 — entrenamiento YOLO (una sola vez)
│   ├── Calcular_Registro.py        # Etapa 1 — detección + suavizado → CSV
│   ├── batch_multicanal.py         # Etapa 2a — recortes PNG desde el CSV
│   ├── npy_multicanal.py           # Etapa 2b — recortes NPY desde el CSV
│   ├── grafica_coordenadas.py      # Verificación — drift acumulado y contexto
│   ├── grafica_registro_crudo.py   # Verificación — cruda vs. suavizada, trayectoria
│   ├── grafica_diff_recorte.py     # Verificación — diff completa vs. recorte
│   ├── reporte_registro.py         # Reporte PDF consolidado de todo lo anterior
│   └── data.yaml                   # Configuración del dataset YOLO
├── data/
│   ├── images/
│   │   ├── Train/                  # PNGs Canal 1 prep para entrenar YOLO
│   │   ├── Val/                    # PNGs Canal 1 prep para validar YOLO
│   │   └── Test/                   # PNGs Canal 1, 2, 3 prep para inferencia
│   ├── labels/                     # Anotaciones YOLO (.txt) del Canal 1
│   ├── numpy_arrays/               # NPYs preprocesados por canal
│   ├── diff/{npy,png}/             # Diferencia entre frames (C2_diff)
│   └── mask/{npy,png}/             # Máscaras Otsu (C3_mask)
├── Resultados/
│   ├── registro/
│   │   ├── coordenadas_registro.csv
│   │   └── *.png                   # Figuras de verificación sueltas
│   ├── reporte_registro/
│   │   ├── reporte_registro.pdf    # Reporte consolidado
│   │   └── figuras/                # Las mismas figuras, una por archivo
│   ├── 3_canales/                  # Recortes PNG (canal_1..3, diff, mask)
│   └── numpy_recortes/             # Recortes NPY (canal_1..3, diff, mask)
├── runs/detect/
│   └── Target_Area_prep/weights/best.pt
├── yolo11n.pt                      # Pesos base YOLO
├── requirements.txt
└── README.md
```

---

## Instalación

**Python recomendado:** 3.10.11

```powershell
# 1. Entorno virtual
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2. PyTorch (elegir según CUDA disponible)
# CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# CUDA 11.8 (RTX 3050):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Resto de dependencias
pip install -r requirements.txt

# 4. Verificar
python -c "import torch; print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

> Si no conoces tu versión de CUDA, ejecuta `nvidia-smi`.

> **`requirements.txt` está codificado en UTF-16.** Si `pip install -r` falla con un error de codificación, reguárdalo en UTF-8 desde el editor.

---

## Flujo de trabajo

```
AFM_Preprocessing
  png_procesado/prep/  ──►  data/images/Test/     (Canal 1, 2, 3 prep)
  npy_procesado/prep/  ──►  data/numpy_arrays/    (Canal 1, 2, 3 prep)
  npy_procesado/diff/  ──►  data/diff/npy/        (C2_diff)
  npy_procesado/mask/  ──►  data/mask/npy/        (C3_mask)
        │
        ▼
  [Una sola vez]  training_yolo.py
        │   Etiquetar Canal 1 prep con LabelImg → data/labels/
        │   → runs/detect/Target_Area_prep/weights/best.pt
        ▼
  Calcular_Registro.py
        │   → Resultados/registro/coordenadas_registro.csv
        │
        ├──► batch_multicanal.py  → Resultados/3_canales/
        └──► npy_multicanal.py    → Resultados/numpy_recortes/
                  │
                  ├──► reporte_registro.py  → Resultados/reporte_registro/
                  │      (verificación; opcional para el pipeline)
                  ▼
          AFM_PredictiveModel/data/
            canal_2/  canal_3/  diff/  mask/
```

Cada repositorio guarda como entrada su propia copia de la salida del anterior. No se leen carpetas del repo vecino: eso mantiene los módulos independientes, pero exige copiar los archivos entre etapas.

---

## Etapa 0 — Entrenamiento YOLO

Solo hace falta una vez. Si ya existe `runs/detect/Target_Area_prep/weights/best.pt`, sáltala.

1. Copiar PNGs del Canal 1 preprocesado a `data/images/Train/` y `data/images/Val/`
2. Etiquetar con **LabelImg**, clase única `Target_Area`
3. Guardar las anotaciones en `data/labels/` en formato YOLO

> Solo se etiqueta el Canal 1: es topografía, visualmente más estable y con menos ruido que los canales PFM.

`data.yaml`:

```yaml
path: ./data
train: images/Train
val:   images/Val
nc: 1
names: ['Target_Area']
```

```powershell
python scripts/training_yolo.py
```

---

## Etapa 1 — Calcular_Registro.py

```powershell
python scripts/Calcular_Registro.py
```

### Parámetros

```python
MODEL_PATH  = 'runs/detect/Target_Area_prep/weights/best.pt'
IMAGE_DIR   = '.../data/images/Test'          # PNGs Canal 1 prep
OUTPUT_DIR  = '.../Resultados/registro'
FRAME_START = 21
FRAME_END   = 60
FILE_PREFIX = 'bifeo_training'
CONFIDENCE  = 0.85
IMG_WIDTH   = 256
IMG_HEIGHT  = 128

SMOOTHING_MODE      = 'outlier'   # 'outlier' | 'moving_avg'
WINDOW              = 4           # vecinos a cada lado
OUTLIER_THRESHOLD_X = 0.25        # desviación en px antes de corregir
OUTLIER_THRESHOLD_Y = 0.25
```

**Modos de suavizado**

`'outlier'` corrige solo los frames que se apartan del promedio de sus vecinos más allá del umbral. Con umbrales por debajo de 1 px sobre coordenadas enteras, el efecto es suprimir la variación de detección conservando la tendencia.

`'moving_avg'` reemplaza todos los frames por la media móvil de la ventana. Suaviza más, pero puede atenuar deriva real.

> Subir el umbral a 1.0 px o más reduce la corrección y el desplazamiento entre recortes vuelve a ser visible. El valor de 0.25 px es el usado en la tesis.

### Salida: `coordenadas_registro.csv`

```
frame,center_x,center_y,center_x_raw,center_y_raw,conf,status
21,126,61,126.43,61.28,0.9365,suavizado
22,126,61,126.71,61.55,0.9426,suavizado
28,127,62,127.12,62.34,0.9257,ok
```

| Columna | Contenido |
|---|---|
| `center_x`, `center_y` | Coordenadas corregidas, enteras. **Son las que consumen los scripts de recorte.** |
| `center_x_raw`, `center_y_raw` | Float original de YOLO, sin truncar ni corregir. Solo registro, para las figuras de verificación. |
| `conf` | Confianza de la detección |
| `status` | `ok` (sin corregir) · `suavizado` · `fallo_deteccion` · `sin_png` |

---

## Etapa 2a — batch_multicanal.py

```powershell
python scripts/batch_multicanal.py
```

```python
IMAGE_DIR    = '.../data/images/Test'
DIFF_PNG_DIR = '.../data/diff/png'
MASK_PNG_DIR = '.../data/mask/png'
INPUT_CSV    = '.../Resultados/registro/coordenadas_registro.csv'

CROP_MODE    = 'cuadrado'   # 'cuadrado' | 'rectangular'
CROP_SIZE    = 80
CROP_WIDTH   = 80           # solo en modo rectangular
CROP_HEIGHT  = 64           # solo en modo rectangular
```

Salida en `Resultados/3_canales/{canal_1,canal_2,canal_3,diff,mask}/`, con nombres `bifeo_training_N_Canal_C_prep_recorte_80px.png`.

---

## Etapa 2b — npy_multicanal.py

```powershell
python scripts/npy_multicanal.py
```

Idéntico al anterior pero sobre las carpetas NPY:

```python
NUMPY_DIR    = '.../data/numpy_arrays/'
DIFF_NPY_DIR = '.../data/diff/npy'
MASK_NPY_DIR = '.../data/mask/npy'
```

Salida en `Resultados/numpy_recortes/`, con la misma estructura.

> El recorte hace únicamente slicing del array. No renormaliza, así que los valores del recorte son los mismos de la imagen completa en esa subregión.

---

## Figuras de verificación

Cuatro scripts independientes, todos opcionales para el pipeline pero útiles para documentar la calidad del registro. Se corren después de las etapas anteriores.

Si solo quieres una pieza citable para la tesis, corre directamente `reporte_registro.py`: reúne el contenido de los tres primeros y añade la verificación visual del recorte.

### grafica_coordenadas.py

```powershell
python scripts/grafica_coordenadas.py
```

Tres paneles: coordenada X, coordenada Y y **drift acumulado** respecto al frame inicial. Cada panel de coordenada incluye un inset que sitúa el recorte de 80 px dentro de la imagen completa.

Su aporte propio es el drift acumulado y los insets de contexto. Los dos primeros paneles quedan cubiertos por el script siguiente.

Salida: `Resultados/registro/coordenadas_registro.png`

### grafica_registro_crudo.py

```powershell
python scripts/grafica_registro_crudo.py
```

Tres figuras independientes:

- `coordenadas_crudo_vs_suavizado_x.png` — detección cruda, truncada y suavizada del eje X
- `coordenadas_crudo_vs_suavizado_y.png` — lo mismo para el eje Y
- `trayectoria_centro_xy.png` — recorrido del centro en el plano, con los frames unidos en orden temporal

Cuantifica la variación media entre frames consecutivos de cada serie y marca el frame con mayor desviación.

Constantes ajustables si el texto de las anotaciones cae sobre las curvas:

```python
ANOTACION_OFFSET = {'x': (-70, 40), 'y': (-120, 0)}   # (dx, dy) en puntos
PLANO_XLIM = (120, 131)   # None = ajustar a los datos
PLANO_YLIM = (60, 66)
```

### grafica_diff_recorte.py

```powershell
python scripts/grafica_diff_recorte.py
```

Compara `|f_n − f_{n−1}|` del Canal 2 sobre la imagen completa contra el recorte del Target Area. Responde si el switching ocurre dentro de la región seleccionada.

Requiere que `npy_multicanal.py` haya corrido antes.

Salida: `diff_completa_vs_recorte.png` y `.csv` con las magnitudes crudas.

### reporte_registro.py

```powershell
python scripts/reporte_registro.py
```

Consolida todo lo anterior en un PDF de nueve páginas y añade la **verificación visual del recorte**, que ninguna figura suelta cubre:

| Página | Contenido |
|---|---|
| 1 | Portada con los parámetros del registro y las tres métricas principales |
| 2 | Confianza de detección por frame y estado de cada uno |
| 3–4 | Coordenadas X e Y: cruda, truncada y suavizada |
| 5 | Drift acumulado y trayectoria del centro en el plano |
| 6 | **ROI dibujado sobre la imagen completa** en frames seleccionados |
| 7 | **Tira temporal del mismo recorte** a lo largo de la serie |
| 8 | Cambio del Canal 2: imagen completa frente al recorte |
| 9 | Tabla resumen con las conclusiones redactadas |

Las páginas 6 y 7 son la evidencia más directa de que el registro funciona. En la 6 se superponen el centro usado (cruz naranja) y la detección cruda de YOLO (equis azul): donde el filtro corrigió, ambas marcas se separan. En la 7, si el registro es correcto, la estructura permanece en la misma posición del recuadro a lo largo de toda la serie.

Cada figura se guarda además suelta en `Resultados/reporte_registro/figuras/`, para insertarla individualmente en la tesis sin extraerla del PDF.

Constantes ajustables:

```python
FRAMES_VERIFICACION = None      # None = primero, último y el de mayor desviación
TRAY_XLIM = (120, 131)          # límites del panel de trayectoria
TRAY_YLIM = (60, 66)
PARAMS = {...}                  # parámetros que se imprimen en la portada
```

> `PARAMS` es informativo: el reporte no lee los valores de `Calcular_Registro.py`. Si cambias los umbrales del filtro, actualízalo a mano.

Las secciones 6, 7 y 8 requieren que hayan corrido `batch_multicanal.py` y `npy_multicanal.py`. Si falta alguna carpeta, esa sección se omite con un aviso y el reporte se genera igual con lo demás.

### Resultados sobre la serie de la tesis (frames 21–60)

| Verificación | Resultado |
|---|---|
| Variación entre frames, eje X | 1.12 px cruda → 0.08 px suavizada (93 % menos) |
| Variación entre frames, eje Y | 0.46 px cruda → 0.08 px suavizada (83 % menos) |
| Deriva total del Target Area | 2 px en X (0.8 % de 256), 3 px en Y (2.3 % de 128) |
| Correlación diff completa vs. recorte, media | r = 0.985 |
| Correlación diff completa vs. recorte, máximo | r = 0.770 |
| Frames de mayor cambio | 36–37 y 46–47, coincidentes en ambas series |
| Cambio medio promedio | 0.133 completa · 0.136 recorte |

Los picos de switching aparecen en ambas series, lo que confirma que la actividad ocurre dentro del Target Area y que el canal `C2_diff` recortado conserva la señal que se le atribuye.

---

## Recortes cuadrados vs. rectangulares

| Modo | Parámetros | Requisito |
|---|---|---|
| `'cuadrado'` | `CROP_SIZE = 80` | Cualquier valor |
| `'rectangular'` | `CROP_WIDTH = 80`, `CROP_HEIGHT = 64` | Ambos divisibles por 32 |

> **Por qué divisible por 32:** la U-Net con encoder EfficientNet-B0 hace 5 niveles de downsampling (×32). Dimensiones que no lo cumplan fallarán en los modelos predictivos.

El sufijo del archivo cambia solo: `_recorte_80px` o `_recorte_80x64px`.

---

## Indexación posicional

Los modelos predictivos acceden a los recortes por **posición** en la lista ordenada alfabéticamente (base 1), no por el número del nombre.

Esto preserva el orden cronológico siempre que los nombres sigan el patrón `bifeo_training_N_...` con N consecutivo. Pero atención al emparejar carpetas con distinto número de archivos: `diff/` tiene un archivo menos que `canal_2/` porque el primer frame no tiene diferencia previa, de modo que la misma posición apunta a frames distintos en cada carpeta.

---

## Notas técnicas

**Ruta del modelo.** Verificar que `MODEL_PATH` apunta al `best.pt` correcto. La carpeta usada en la tesis es `Target_Area_prep`, no `Target_Area`.

**Confianza.** `CONFIDENCE = 0.85` es el valor de producción. Si YOLO falla en muchos frames, bajar a 0.75 y revisar las figuras de verificación.

**Frame sin diff.** El primer frame de la serie no tiene `_diff` porque no existe uno anterior. Los scripts lo omiten sin error.

**Compatibilidad de nombres.** PNGs y NPYs deben seguir el patrón `bifeo_training_N_Canal_C_prep` para que los scripts emparejen los canales.

**Rutas absolutas.** Los scripts usan rutas absolutas de Windows en su sección de configuración. Ajustarlas antes de ejecutar en otra máquina.

---

## Licencia

Proyecto académico — Instituto Tecnológico de Querétaro. Todos los derechos reservados.