# AFM-ROI-Toolkit

Detección y recorte multicanal alineado (PNG + NPY)

Resumen

AFM-ROI-Toolkit procesa datos AFM de tres canales (Canal 1, Canal 2 y Canal 3). El flujo principal incluye la preparación y etiquetado de datos, el entrenamiento de un detector YOLO sobre las imágenes del de superficie, y el procesamiento/inferencia multicanal que genera recortes alineados en PNG y NumPy.

## Setup: entorno virtual e instalación de dependencias

**Versión de Python usada en este proyecto:** `3.10.11` (recomendada). El entorno `env` incluido fue creado con Python 3.10.11. Si instalas un nuevo entorno, usa Python 3.10.x para máxima compatibilidad.

Sigue estos pasos (PowerShell) desde la raíz del proyecto para crear un entorno virtual e instalar las dependencias. Ajusta la instalación de PyTorch según la versión de CUDA de tu sistema (o usa la variante CPU si no tienes GPU compatible).

1. Crear y activar el entorno virtual

```powershell
python -m venv env
.\env\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

2. Instalar PyTorch (elige la opción adecuada antes de continuar con `requirements.txt`)

- CPU-only (recomendado si no tienes GPU o para pruebas rápidas):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

- CUDA 11.8 (ejemplo):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- CUDA 12.1 (ejemplo):

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Si no estás seguro de la versión de CUDA, consulta la salida de `nvidia-smi` en tu sistema o usa la variante CPU.

3. Instalar el resto de dependencias desde `requirements.txt`

```powershell
pip install -r requirements.txt
```

4. Verificar la instalación de PyTorch y si detecta GPU

```powershell
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Notas:
- Si prefieres `conda`, crea un entorno con `conda create -n afm python=3.11` y activa con `conda activate afm`, luego instala PyTorch según la guía oficial de PyTorch para `conda`.
- Asegúrate de ejecutar los comandos dentro del entorno virtual activo para que los paquetes se instalen en `env`.

## Flujo de Trabajo: Entrenamiento

### 1. Preparación y Etiquetado de Datos
Esta fase se centró en organizar los datos de origen y crear el conjunto de entrenamiento necesario para el modelo de detección.

#### Conversión de Datos
- **Origen de Datos:** Se trabajó con datos de tres canales AFM: Canal 1, Canal 2 y Canal 3.
- **Formato de Almacenamiento:** Los datos brutos de cada canal se convirtieron y almacenaron en dos formatos paralelos:
	1. Arrays NumPy (`.npy`): los datos crudos, listos para el análisis cuantitativo.
	2. Imágenes PNG: visualizaciones de los datos, utilizadas para entrenamiento y detección visual.

#### Etiquetado Simplificado
- **Selección de Fuente:** Solo se utilizaron las imágenes PNG del Canal 1 para el etiquetado.
- **Clase Única:** La etiqueta utilizada es `Target_Area`.
- **Resultado:** Conjunto de imágenes PNG del Canal 1 etiquetadas con archivos YOLO `.txt` (ej. generados con `labelimg`).

### 2. Entrenamiento del Modelo
- **Framework:** YOLO (You Only Look Once).
- **Entrada:** PNG del Canal 1 con coordenadas de `Target_Area`.
- **Objetivo:** Entrenar para detectar con alta confianza el centro de la `Target_Area` en imágenes PNG del Canal 1.
- **Salida:** Modelo entrenado (`best.pt`).

### 3. Procesamiento y Recorte Multi-Canal (Inferencia)
Esta fase genera recortes alineados en dos formatos (PNG y `.npy`) usando la detección en Canal 1 como referencia.

#### Detección de Coordenadas
- **Fuente de Detección:** Imágenes PNG del Canal 1 en la carpeta `data/images/Test` (u otras carpetas de imágenes).
- **Inferencia:** Se ejecuta el modelo `best.pt` para obtener la detección de la `Target_Area`.
- **Extracción:** Se extraen `center_x, center_y` y se define la ventana de recorte.

#### Alineación y Recorte
- La ventana de recorte detectada se aplica a los tres canales, tanto a las imágenes PNG como a los arrays NumPy, garantizando correspondencia espacial.
- **Scripts a utilizar:**
  - `scripts/batch_multicanal.py`: ejecuta YOLO sobre las imágenes PNG del Canal 1, extrae `center_x, center_y` y guarda los recortes PNG correspondientes para Canal 1, Canal 2 y Canal 3 en `Resultados/3_canales`.
  - `scripts/npy_multicanal.py`: utiliza las mismas coordenadas detectadas por el modelo en Canal 1 para recortar los arrays NumPy (`.npy`) de los tres canales, generando versiones recortadas alineadas para análisis cuantitativo en `Resultados/numpy_recortes`. 

  - **Salidas:** 
	1. Arrays NumPy recortados (`.npy`) para análisis cuantitativo.
	2. Imágenes PNG recortadas (`.png`) para verificación visual.

---

## Notas de Implementación y Rutas Clave
- `data/numpy_arrays/` — arrays `.npy` por canal (`canal_1/`, `canal_2/`, `canal_3/`).
- `data/images/` — imágenes PNG usadas para entrenamiento y test (`Train/`, `Val/`, `Test/`).
- `runs/detect/Target_Area/weights/best.pt` — modelo entrenado (ruta típica).
- `scripts/batch_multicanal.py` — ejecuta YOLO sobre PNG del Canal 1 y guarda recortes PNG para canales 1, 2 y 3.
- `scripts/npy_multicanal.py` — toma las coordenadas detectadas en el Canal 1 y recorta los arrays `.npy` correspondientes de los tres canales.
- Salidas: `Resultados/3_canales/canal_1`, `.../canal_2`, `.../canal_3` y carpetas para `.npy` recortados: `Resultados/numpy_recortes/canal_1`, `.../canal_2`, `.../canal_3`

---

Notas adicionales:
- Revisa `MODEL_PATH` en `scripts/batch_multicanal.py` y `scripts/npy_multicanal.py` para apuntar al modelo correcto (`best.pt`).
- Asegúrate de que las imágenes PNG y los `.npy` están indexados/nombrados de forma que el script pueda emparejarlos (p. ej. mismo prefijo de archivo con sufijos `_Canal_1.png`, `_Canal_2.png`, `_Canal_3.png` y nombres `.npy` compatibles).

---

