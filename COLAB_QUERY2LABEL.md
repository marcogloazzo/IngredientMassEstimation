# Colab: variante ConvNeXt Small + Query2Label

Notebook listo para abrir en Colab: `COLAB_QUERY2LABEL.ipynb`. Incluye los comandos
de esta guía; entrenamiento, comparación BCE, evaluación e inferencia quedan
desactivados hasta habilitar explícitamente sus variables. Completar la URL del
repositorio antes de ejecutar la celda de Git.

Para probar una época local manualmente, desde `IngredientMassEstimation`:

```powershell
$pythonExe = "..\OmniFood8K\Code\.venv\Scripts\python.exe"
& $pythonExe scripts/prepare_nutrition5k.py --config config/query2label_100.json --output-dir data/processed_100
& $pythonExe scripts/train_image_model.py --check-data
& $pythonExe -u scripts/train_image_model.py --config config/query2label_100.json --device cpu --workers 0 --epochs 1 --output-dir artifacts/query2label_100_asl
```

La última línea sí entrena; ejecutar solo cuando se decida comenzar. Para continuar
en Colab, subir el contenido completo de `artifacts/query2label_100_asl/` a
`MyDrive/IngredientMassEstimation/runs/query2label_100_asl/`, incluyendo `last.pt`,
`best.pt`, `history.csv`, `config.json` y `latest_validation.json`. En el notebook
usar `MODE = "resume"` y habilitar `RUN_TRAINING`. `TOTAL_EPOCHS = 30` es el límite
total: después de una época local comienza en la 2. Mantener el resto del config,
incluidos batch, acumulación y `amp`; AMP se desactiva automáticamente en CPU.
El script admite el estado AMP vacío de un checkpoint CPU al continuar en CUDA.

Subir a Git `src/`, `scripts/`, `config/`, `tests/`, `requirements.txt`,
`pyproject.toml`, `.gitignore`, los documentos Markdown y el notebook. Los datos
procesados se regeneran en Colab. Subir a Drive el ZIP del dataset y, si ya se
entrenó localmente, la carpeta de la corrida. La foto del celular es opcional.

Ejecutar estas celdas después de subir el código al repositorio correcto.
El repositorio local de `ProyectoNutricional` todavía no tiene commits ni remote;
no se ha publicado automáticamente. No subir dataset, pesos, outputs ni `.venv`.
Si se publica solo `IngredientMassEstimation`, conservar la estructura interna.

No se necesita profundidad, OmniFood ni otro dataset. Las 100 clases provienen de
las anotaciones actuales de Nutrition5k. La atención/cabezales necesitan entrenarse.

## Preparar los datos una sola vez en la PC

Crear un ZIP de los RGB ya descargados, metadata y splits (sin `.venv`, pesos ni
carpetas del proyecto). Desde `IngredientMassEstimation` en PowerShell:

```powershell
Compress-Archive -LiteralPath ".\data\raw\nutrition5k_dataset" -DestinationPath ".\data\nutrition5k_rgb.zip"
```

Si el ZIP ya existe, reutilizarlo. Subirlo una vez a
`MyDrive/IngredientMassEstimation/nutrition5k_rgb.zip`.
El conjunto RGB local ocupa aproximadamente 1.45 GB; no descargar el dataset
completo de videos. El ZIP debe contener `nutrition5k_dataset/metadata`,
`nutrition5k_dataset/dish_ids` y `nutrition5k_dataset/imagery/realsense_overhead`.
Cada sesión Colab nueva necesitará copiar/extractar el ZIP, pero no volver a
transferirlo desde la PC. Drive es persistente; `/content` y paquetes del runtime no.

## 1. GPU y Drive

Elegir un runtime GPU en Colab antes de empezar.

```python
import torch
assert torch.cuda.is_available(), "Este runtime no tiene GPU CUDA. Cambiá el entorno de ejecución."
print(torch.cuda.get_device_name(0))
from google.colab import drive
drive.mount("/content/drive")
```

## 2. Obtener el código

Para repositorio privado, guardar `GITHUB_TOKEN` como secreto Colab con lectura
del repositorio y habilitar su acceso al notebook. No pegar el token en el código.
Reemplazar la URL siguiente por el repositorio al que efectivamente subiste estos archivos:

```python
from google.colab import userdata
from pathlib import Path
import base64, os, subprocess

REPO = Path("/content/nutrition-code")
URL = "https://github.com/TU_USUARIO/TU_REPOSITORIO.git"
token = userdata.get("GITHUB_TOKEN")
authorization = base64.b64encode(f"x-access-token:{token}".encode()).decode()
environment = os.environ.copy()
environment.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.extraHeader",
                    "GIT_CONFIG_VALUE_0": f"Authorization: Basic {authorization}"})
command = ["git", "-C", str(REPO), "pull", "--ff-only"] if (REPO / ".git").exists() else ["git", "clone", URL, str(REPO)]
result = subprocess.run(command, env=environment, capture_output=True, text=True)
if result.returncode:
    raise RuntimeError("Git falló. Verificá URL, permisos del token y que no haya cambios locales; no se imprime la credencial.")
del token, authorization, environment
candidates = [REPO / "IngredientMassEstimation", REPO]
PROJECT = next((p for p in candidates if (p / "config/query2label_100.json").is_file()), None)
assert PROJECT is not None, "El repositorio no contiene la nueva variante."
os.chdir(PROJECT)
print("Project:", PROJECT)
```

## 3. Dependencias mínimas

Colab normalmente ya proporciona PyTorch con CUDA. No instalar encima el torch CPU
de Windows ni copiar `.venv`; verificar antes de modificar paquetes.

```python
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "Pillow>=10,<13", "tqdm>=4.66,<5"], check=True)
import torch, torchvision, numpy
assert hasattr(torch.amp, "GradScaler"), "Hace falta PyTorch >= 2.3."
assert torch.cuda.is_available()
print("Torch:", torch.__version__, "Torchvision:", torchvision.__version__, "NumPy:", numpy.__version__)
```

No se fuerza NumPy 1.x; el código admite NumPy 2.x. No hacen falta Open3D,
Transformers, xFormers ni los paquetes de OmniFood. Las sesiones nuevas siguen
necesitando dependencias, pero acá no se instalan los paquetes pesados ya presentes.

## 4. Copiar los datos a almacenamiento local de Colab

```python
import shutil, zipfile
DRIVE = Path("/content/drive/MyDrive/IngredientMassEstimation")
DATA = Path("/content/nutrition5k/nutrition5k_dataset")
if not (DATA / "metadata").is_dir():
    archive = Path("/content/nutrition5k_rgb.zip")
    shutil.copy2(DRIVE / "nutrition5k_rgb.zip", archive)
    destination = DATA.parent
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            resolved = (destination / name).resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError("Ruta insegura en el ZIP")
        source.extractall(destination)
assert (DATA / "metadata").is_dir(), "Revisá la estructura interna del ZIP."
```

Se entrena leyendo RGB de `/content`, no abriendo miles de archivos directamente
en Drive. Mantener espacio para ZIP, extracción, pesos y checkpoints completos.

## 5. Preparar las etiquetas y verificar sin entrenar

```python
import sys
subprocess.run([sys.executable, "scripts/prepare_nutrition5k.py",
                "--config", "config/query2label_100.json", "--dataset-root", str(DATA),
                "--output-dir", "data/processed_100"], check=True)
subprocess.run([sys.executable, "scripts/train_image_model.py", "--check-data",
                "--dataset-root", str(DATA)], check=True)
```

Esperado si el ZIP tiene los mismos datos locales: 100 clases, 2332 train,
423 validación, 507 test. Si cambian los datos disponibles, cambian registros/splits.
No ejecutar extracción del cache de vectores para este modelo.

## 6. Entrenar ASL

```python
RUN = DRIVE / "runs/query2label_100_asl"
arguments = [sys.executable, "-u", "scripts/train_image_model.py",
             "--dataset-root", str(DATA), "--output-dir", str(RUN),
             "--torch-cache", str(DRIVE / "torch_cache"), "--device", "cuda", "--workers", "2"]
subprocess.run(arguments, check=True)
```

Se descargan una vez los pesos ImageNet ConvNeXt Small (~192 MB) en el cache de
Drive. El modelo anterior de OmniFood no interviene. Los checkpoints completos
incluyen el encoder: `best.pt` y `last.pt` pueden ocupar cientos de MB cada uno.
Guardar en Drive permite recuperar el cierre de época tras perder el runtime;
esa escritura añade algo de tiempo. No se puede reanudar una época a mitad.
Si el proceso se interrumpe, después de ejecutar preparación nuevamente:

```python
subprocess.run(arguments + ["--resume", str(RUN / "last.pt")], check=True)
```

Observar tiempos reales por época y consumo de unidades de Colab antes de
continuar las 30 épocas. No se garantiza completar el entrenamiento con 100 unidades.
Si falta memoria, usar batch 4 y acumulación 8 en una copia del config para mantener
batch efectivo 32; no cambiar la configuración de una corrida que se reanuda.
No entrenar una corrida ASL encima de una BCE ni viceversa.

## 7. Comparar contra BCE (si el presupuesto lo permite)

```python
RUN_BCE = DRIVE / "runs/query2label_100_bce"
subprocess.run([sys.executable, "-u", "scripts/train_image_model.py",
                "--dataset-root", str(DATA), "--output-dir", str(RUN_BCE),
                "--torch-cache", str(DRIVE / "torch_cache"), "--device", "cuda",
                "--workers", "2", "--presence-loss", "bce"], check=True)
```

Mismo seed, datos y arquitectura; solo cambia la pérdida. La selección del modelo
y la elección ASL/BCE se hacen con validación. No elegir entre variantes mirando
reiteradamente el test. Dos entrenamientos completos cuestan aproximadamente el
doble que uno; primero comprobar viabilidad de una corrida.

## 8. Calibrar presencia con validación y evaluar test

```python
VALIDATION = DRIVE / "reports/query2label_asl_validation"
subprocess.run([sys.executable, "scripts/evaluate_image_model.py",
                "--checkpoint", str(RUN / "best.pt"), "--dataset-root", str(DATA),
                "--split", "validation", "--calibrate", "--output-dir", str(VALIDATION),
                "--device", "cuda"], check=True)
CHECKPOINT = VALIDATION / "calibrated.pt"
subprocess.run([sys.executable, "scripts/evaluate_image_model.py",
                "--checkpoint", str(CHECKPOINT), "--dataset-root", str(DATA),
                "--split", "test", "--output-dir", str(DRIVE / "reports/query2label_asl_test"),
                "--device", "cuda"], check=True)
```

El umbral cambia reconocimiento/reportes, no recorta las proporciones ni cambia
los nutrientes. Calibración de ASL puede ser importante porque las probabilidades
no necesariamente están calibradas. Las métricas mantienen MAE y PMAE del proyecto
y agregan gramos falsos/error de masa por plato.

## 9. Una foto del celular y resultados intermedios

Subir una foto a `MyDrive/IngredientMassEstimation/photos/plato.png` y ejecutar:

```python
subprocess.run([sys.executable, "scripts/infer_image_model.py",
                "--checkpoint", str(CHECKPOINT), "--image", str(DRIVE / "photos/plato.png"),
                "--output-dir", str(DRIVE / "predictions"), "--device", "cuda",
                "--save-attention", "--top-k", "10"], check=True)
```

Se guardan JSON, crop RGB, tensores y superposiciones de atención en
`predictions/plato/`. La inferencia no descarga nuevos pesos: el checkpoint contiene
todo el modelo. Los mapas no son máscaras de segmentación ni prueban presencia.
Un plato fuera de Nutrition5k puede tener errores mayores por cambio de dominio.

En la PC usar los mismos scripts con `--device cpu --workers 0`, pero el entrenamiento
completo sería lento. Copiar `best.pt`/`calibrated.pt` para inferencia local no requiere
copiar el runtime ni el optimizer de `last.pt`.
