# Colab: IngredientMassEstimation V2

Abrir [COLAB_QUERY2LABEL_V2.ipynb](COLAB_QUERY2LABEL_V2.ipynb).
Se creó la rama local **v2-multiscale-depth**. No se hizo commit ni push.
El notebook original V1 se conservó sin modificar.

## Qué subir a Git

Subir código/configuración/documentación/notebook/pruebas de V2, no los datos.
La siguiente selección explícita evita incluir el notebook V1 con salidas o
archivos ajenos que ya estaban sin guardar. Ejecutar desde IngredientMassEstimation:

```powershell
git branch --show-current
# Debe mostrar v2-multiscale-depth.
$v2Files = @(
  ".gitignore", "README.md", "pyproject.toml", "requirements-v2.txt",
  "config/query2label_100_v2.json",
  "src/ingredient_mass/spatial_model.py",
  "src/ingredient_mass/multiscale_model.py",
  "src/ingredient_mass/depth.py",
  "src/ingredient_mass/image_training.py",
  "scripts/train_image_model.py",
  "scripts/evaluate_image_model.py",
  "scripts/infer_image_model.py",
  "scripts/inspect_image_model.py",
  "scripts/prepare_depth.py",
  "tests/test_multiscale_depth.py",
  "tests/test_colab_v2.py",
  "ARQUITECTURA_QUERY2LABEL_V2.md",
  "COLAB_QUERY2LABEL_V2.md",
  "COLAB_QUERY2LABEL_V2.ipynb"
)
git add -- $v2Files
git diff --cached --stat
git commit -m "Add V2 multiscale RGB and frozen Depth Anything pipeline"
git push -u origin v2-multiscale-depth
```

También se puede hacer Stage/Commit/Publish Branch desde Source Control de VS Code,
seleccionando los mismos archivos. La carpeta ya es el repositorio: no se sube como
un archivo/carpeta a Source Control.

Opcionales: Cambios_2do_modelo.md y photos/annotations.json, que documentan
las decisiones y las masas reales informadas para IMG_9983/IMG_9993.
Las anotaciones son referencias, no etiquetas agregadas a entrenamiento.
No incluir imágenes personales, dataset, cache, pesos, runs, reports,
predictions, outputs ni .venv; quedan ignorados.
Los archivos que ya estaban versionados no dejan de estarlo por agregar gitignore.

## Qué guardar en Drive

Reutilizar la estructura V1 y crear una rama de carpetas V2:

```text
MyDrive/IngredientMassEstimation/
├── nutrition5k_rgb.zip          # ya subido: reutilizar, no extraer en Drive
├── torch_cache/                # cache ImageNet existente: reutilizar
├── photos/
│   ├── IMG_9983.png             # opcional, para comparar
│   └── IMG_9993.png             # opcional, para comparar
├── runs/ y reports/             # conservar V1
└── v2/
    ├── weights/
    │   └── depth_anything_v2_vits.pth
    ├── depth_v2_vits.zip
    ├── runs/query2label_100_asl/
    │   ├── best.pt
    │   ├── last.pt
    │   ├── history.csv
    │   ├── config.json
    │   └── latest_validation.json
    ├── reports/
    │   ├── query2label_asl_validation/
    │   │   ├── calibrated.pt
    │   │   ├── metrics.json
    │   │   └── evaluation_details.json
    │   └── query2label_asl_test/
    │       ├── metrics.json
    │       ├── evaluation_details.json
    │       └── predictions.npz
    └── predictions/<foto>/
        ├── prediction.json
        └── intermediates/
```

**Desde la PC no necesitás subir pesos nuevos ni cache de profundidad.**
El notebook descarga Small oficial a v2/weights, calcula el cache cuando lo
habilites y guarda el ZIP en v2. Entrenamiento/evaluación/inferencia escriben sus
resultados directamente en esas carpetas.

No copiar checkpoints V1 dentro de la corrida V2. Conservar best.pt y last.pt:
el primero sirve para seleccionar/evaluar; el segundo incluye optimizer/scaler/RNG
para reanudar. Para inferencia nueva también hacen falta los pesos Small.
Los mapas sin comprimir ocupan unos 327 MB para 3262 RGB; los checkpoints
completos requieren espacio adicional. /content es temporal y Drive persistente.

## Primera sesión Colab

1. Publicar la nueva rama y abrir el notebook V2. Elegir GPU.
2. Montar Drive y clonar esa rama. Para repo privado, secreto GITHUB_TOKEN
   de lectura con acceso habilitado. Para público, PRIVATE_REPO=False.
3. Instalar dependencias mínimas sin reemplazar PyTorch CUDA.
4. Copiar y extraer el ZIP RGB a /content. Soporta nombres Windows con
   barras invertidas y valida rutas; detecta la raíz que contiene metadata,
   dish_ids e imagery/realsense_overhead.
5. Preparar etiquetas con el config V2. Mantiene el split actual.
6. Descargar código/pesos oficiales de Depth Anything Small.
7. Habilitar RUN_DEPTH_PREPARATION=True una vez; ejecutar y guardar el cache.
   Ejecutar los controles --check-cache y --check-data. No entrenan.
8. Habilitar RUN_TRAINING=True, MODE="new", TOTAL_EPOCHS=100.
   Esto **sí entrena**, con progreso visible en la celda.
9. Al terminar, RUN_CALIBRATION=True; luego RUN_TEST=True para evaluación final.
10. RUN_INFERENCE=True para una foto propia o un plato validation/test.
    Las figuras se muestran dentro de Colab.

Los flags arrancan en False: no se inicia entrenamiento al abrir el notebook.
El preprocesamiento de profundidad es inferencia de un modelo congelado,
no entrenamiento del estimador de masa.

## Reanudar tras perder el runtime

Ejecutar de nuevo las celdas 1–7. Se recupera el cache desde su ZIP de Drive,
sin repetir Depth Anything si está completo. Usar RUN_DEPTH_PREPARATION=False.
En el punto 8: MODE="resume" y RUN_TRAINING=True.
TOTAL_EPOCHS es el techo total; desde last.pt época 40 continúa en 41.
No cambiar batch, pérdidas, encoder depth, etc. al reanudar una misma corrida.
Para comparar otra configuración, usar otra carpeta.

No se reanuda a mitad de una época: se recupera el último cierre guardado.
El entrenamiento puede terminar antes de 100 por early stopping.
Si falta memoria con batch 4, crear una configuración/corrida nueva con batch 2
y acumulación 16; no alterar una corrida existente sin dejar registro.

## Comandos equivalentes

Estas líneas son para un runtime Colab ya preparado; no se ejecutaron localmente:

```bash
python -u scripts/prepare_depth.py \
  --config config/query2label_100_v2.json \
  --dataset-root /content/nutrition5k/nutrition5k_dataset \
  --processed-dir data/processed_100 \
  --depth-cache /content/depth_v2_vits \
  --depth-anything-repo /content/depth-anything-v2 \
  --depth-checkpoint /content/drive/MyDrive/IngredientMassEstimation/v2/weights/depth_anything_v2_vits.pth \
  --device cuda

python -u scripts/train_image_model.py \
  --config config/query2label_100_v2.json \
  --dataset-root /content/nutrition5k/nutrition5k_dataset \
  --processed-dir data/processed_100 \
  --depth-cache /content/depth_v2_vits \
  --output-dir /content/drive/MyDrive/IngredientMassEstimation/v2/runs/query2label_100_asl \
  --torch-cache /content/drive/MyDrive/IngredientMassEstimation/torch_cache \
  --device cuda --workers 2 --epochs 100
```

Para reanudar, agregar --resume con el last.pt **de V2**.
Para control sin entrenamiento, agregar --check-data al comando de train.
La ruta DATA detectada por el notebook puede diferir si el ZIP tiene un prefijo
extra: usar siempre esa variable, no adivinar una ruta.

BCE sigue configurable con --presence-loss bce, pero debe tener otra carpeta
de corrida/reportes. No se ejecuta un segundo entrenamiento automáticamente.

## Verificar localmente sin entrenar

```powershell
$pythonExe = "..\OmniFood8K\Code\.venv\Scripts\python.exe"
& $pythonExe -m unittest discover -s tests -v
& $pythonExe tests/test_core.py
& $pythonExe scripts/inspect_image_model.py --config config/query2label_100_v2.json
```

La inspección usa ConvNeXt real sin pesos y tensores sintéticos.
No descarga modelos ni usa Nutrition5k para entrenar.
La validación real del pipeline Depth Anything y del costo GPU se hará en Colab.
