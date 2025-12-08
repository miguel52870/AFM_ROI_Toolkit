from ultralytics import YOLO
import os


if __name__ == '__main__':

# =================================================================
# 1. Configuración de los parámetros de entrenamiento
# =================================================================

    #MODEL_PATH = os.path.join('runs', 'detect', 'Target_Area', 'weights', 'best.pt')
    # (modificar con el path de un modelo ya entrenado para modelos futuros)
    # os.path.join('runs', 'detect', 'carpeta de entrenamiento', 'weights', 'best.pt')

    args = {
        'data': 'scripts/data.yaml',     # Archivo de configuración del dataset
        'epochs': 100,                   # Número de épocas
        'imgsz': (256, 128),             # Tamaño de las imágenes
        'name': 'Target_Area',           # Nombre de la carpeta de resultados (modificar para entrenamientos con modelos ya entrenados)
        'project': 'runs/detect',        # CREACIÓN local EN ./runs/detect/
        'device': '0',                   # Usar CPU (cambiar por '0' si tienes GPU)
        'workers': 10,                   # Número de hilos
        'flipud': 0.0,                   # Deshabilita volteo vertical
        'fliplr': 0.0                    # Deshabilita volteo horizontal
    }
    # =================================================================
    # 2. Iniciar el entrenamiento
    # =================================================================

    print("Iniciando entrenamiento de YOLOv8...")

    model = YOLO('yolo11n.pt')           # modificar por MODEL_PATH para continuar entrenando un modelo ya entrenado
    results = model.train(**args)
    modelo_final_path = os.path.join(
        'runs', 'detect', args['name'], 'weights', 'best.pt'
    )
    print("\nEntrenamiento completado.")
    print(f"Modelo final guardado en: ./{modelo_final_path}")
