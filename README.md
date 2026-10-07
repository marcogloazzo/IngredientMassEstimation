# Estimación de ingredientes y masa con bajo cómputo

## V5: MLP compartido y masa total auxiliar

Experimento aislado pendiente: mantiene el MLP compartido de V3 y activa el cabezal
auxiliar de masa total y consistencia. No combina los MLP separados de V4.
Los gramos finales siguen siendo `p × c`; se reutilizan datos/cache V3 sin cambios.
Ver [arquitectura V5](ARQUITECTURA_QUERY2LABEL_V5.md),
[guía Git/Drive/Colab](COLAB_QUERY2LABEL_V5.md) y
[notebook V5](COLAB_QUERY2LABEL_V5.ipynb), con entrenamiento y test desactivados.

## V4: cabezales especializados y diagnóstico p × c

Nueva variante aislada, sin modificar checkpoints/configs/notebooks V2/V3.
Ver [diagnóstico real de V3](DIAGNOSTICO_GATE_V3.md),
[arquitectura V4](ARQUITECTURA_QUERY2LABEL_V4.md) y
[Colab V4](COLAB_QUERY2LABEL_V4.md).
El notebook `COLAB_QUERY2LABEL_V4.ipynb` deja entrenamiento y test desactivados.

## V3: gramos condicionales y split agrupado provisional

Variante experimental separada de V2: BCE ponderada, cantidades supervisadas
en positivos, penalización de gramos falsos, RGB completo 320×320, fusión
residual de profundidad y scheduler warmup/cosine. La masa total deriva de
sumar ingredientes. Ya se entrenó; el diagnóstico de validación de época 39
está documentado en [DIAGNOSTICO_GATE_V3.md](DIAGNOSTICO_GATE_V3.md). No atribuir
comparaciones históricas con V2 solo a arquitectura: cambiaron split y vocabulario.

- [Notebook Colab V3](COLAB_QUERY2LABEL_V3.ipynb), conservar su copia ejecutada.
- [Guía Colab, Git, Drive y preservación de V2](COLAB_QUERY2LABEL_V3.md).
- [Arquitectura y pérdidas V3](ARQUITECTURA_QUERY2LABEL_V3.md).

## V2: multiescala RGB y profundidad estimada

La nueva variante mantiene una sola imagen RGB como entrada y agrega Depth Anything
V2 Small congelado, una CNN de profundidad entrenable, fusión de etapas 3/4 de
ConvNeXt Small y dos capas de atención cruzada. Tras tres épocas de warmup se
ajustan etapas 2–4; solo stem/etapa 1 permanecen congelados. Las pérdidas y los
cabezales 768 → 384 → 192 se conservan.

- [Arquitectura V2](ARQUITECTURA_QUERY2LABEL_V2.md).
- [Guía Colab, Git y Drive](COLAB_QUERY2LABEL_V2.md).
- [Notebook Colab V2](COLAB_QUERY2LABEL_V2.ipynb).
- Configuración: `config/query2label_100_v2.json`.

La V2 requiere una corrida nueva, no reanudar desde V1. No se ha entrenado ni
medido todavía; el notebook deja desactivados entrenamiento/evaluación/inferencia.
Las variantes de referencia y su documentación histórica continúan debajo.

Este proyecto es una segunda etapa independiente de `OmniFood8K`. Recibe una sola
imagen RGB y produce:

- ingredientes probables;
- proporción y gramos estimados de cada ingrediente;
- calorías, grasa, carbohidratos y proteína calculados a partir de esos gramos.

La prioridad es que sea entrenable en CPU o en una GPU gratuita. Por eso no se
entrena una red visual completa: se extraen características una sola vez con un
EfficientNet-B0 preentrenado y congelado, y después se entrenan solamente cabezales
pequeños.

Los resultados comparativos de las variantes entrenadas están en
[`EXPERIMENTS.md`](EXPERIMENTS.md).

## Variante con atención espacial y ajuste parcial (decisiones aprobadas)

Configuración: `config/query2label_100.json`. ConvNeXt Small + adaptación compacta
Query2Label, 99 ingredientes y `other`, MLP 768 -> 384 -> 192, ASL/BCE configurable,
supervisión directa en gramos y cabezal de masa 192 -> 128 -> 1.
Se ajusta solo la última etapa visual tras tres épocas de warmup.
Se mantiene la conversión nutricional con tabla fija.

- [Arquitectura y explicación de la adaptación](ARQUITECTURA_QUERY2LABEL.md).
- [Celdas y pasos para Google Colab](COLAB_QUERY2LABEL.md).
- Scripts nuevos: `train_image_model.py`, `evaluate_image_model.py`, `infer_image_model.py`.

Esta variante trabaja con imágenes, no con el cache de vectores de los scripts
anteriores. No se ha entrenado todavía y no tiene nuevas métricas de test.

## Nueva variante: ConvNeXt Small y 100 clases

Se incorpora `config/convnext_small_100.json`: 99 ingredientes + `other`.
La arquitectura de los cabezales, pérdidas y tabla nutricional no cambia aún.
ConvNeXt Small usa los pesos ImageNet-1K de torchvision y genera 768 características.
Estos pesos no son los checkpoints de OmniFood (no se cargan automáticamente).
Por ahora se mantiene congelado; el ajuste parcial del encoder está pendiente de decisión.
Las configuraciones y resultados anteriores se conservan.

Para esta variante usar siempre carpetas distintas y el mismo `--config`:

```powershell
$py = "..\OmniFood8K\Code\.venv\Scripts\python.exe"
& $py scripts/prepare_nutrition5k.py --config config/convnext_small_100.json --output-dir data/processed_100
& $py scripts/extract_features.py --config config/convnext_small_100.json --processed-dir data/processed_100 --cache-dir data/cache_convnext_small_100 --device cuda --batch-size 16
& $py scripts/train_heads.py --config config/convnext_small_100.json --vocabulary data/processed_100/vocabulary.json --cache-dir data/cache_convnext_small_100 --output-dir artifacts/convnext_small_100
& $py scripts/evaluate.py --checkpoint artifacts/convnext_small_100/best_heads.pt --features data/cache_convnext_small_100/test_features.npz --output-dir outputs/evaluation_convnext_small_100
& $py scripts/infer_single.py --checkpoint artifacts/convnext_small_100/best_heads.pt --image "RUTA_A_TU_IMAGEN" --output-dir outputs/single_convnext_small_100
```

Estos comandos son la ruta congelada de referencia, no un entrenamiento ya ejecutado.
La extracción con `--device cuda` requiere NVIDIA/CUDA (Colab, no Iris Xe).
En Colab reemplazar `$py` por `python` y usar las rutas del dataset montado/copiado.
No reutilizar el cache de EfficientNet ni los cabezales de 50 clases para esta variante.
La extracción descargará los pesos solo si faltan en `cache/torch/`.

## Qué hay que descargar

1. **Nutrition5k**, pero no el archivo completo de 181.4 GB. Solo hacen falta:
   `metadata/`, `dish_ids/` e imágenes RGB cenitales de
   `imagery/realsense_overhead/`. El script incluido omite las imágenes de
   profundidad.
2. **Google Cloud CLI**, para hacer la descarga selectiva con `gcloud storage`.
3. **Pesos ImageNet de EfficientNet-B0**. No hay que descargarlos manualmente:
   `torchvision` los descarga la primera vez en `cache/torch/`, dentro de este
   proyecto y por lo tanto en el disco D.

No hace falta descargar Recipe1M, FoodSeg103 ni otro checkpoint para la primera
versión. Los pesos de OmniFood y Depth Anything que ya tenés tampoco se duplican.

Fuentes oficiales: [repositorio Nutrition5k](https://github.com/google-research-datasets/Nutrition5k)
y [Google Cloud CLI](https://cloud.google.com/sdk/docs/install-sdk).

### Descargar el subconjunto de Nutrition5k

Desde PowerShell, parado en esta carpeta:

```powershell
cd "D:\IB\7mo SEMESTRE\Deep Learning\ProyectoNutricional\IngredientMassEstimation"
powershell -ExecutionPolicy Bypass -File ".\scripts\download_nutrition5k.ps1"
```

El script invoca `gcloud.cmd` para evitar bloqueos de la política de ejecución de
PowerShell.

El resultado esperado es:

```text
data/raw/nutrition5k_dataset/
├── metadata/
├── dish_ids/
│   └── splits/
└── imagery/
    └── realsense_overhead/
        └── dish_.../
            └── rgb.png
```

## Entorno Python

Se reutiliza el `.venv` de OmniFood para no instalar otra copia pesada de PyTorch.
No es necesario activarlo y así se evita el error de políticas de PowerShell:

```powershell
cd "D:\IB\7mo SEMESTRE\Deep Learning\ProyectoNutricional\IngredientMassEstimation"
uv pip install --python "..\OmniFood8K\Code\.venv\Scripts\python.exe" -r requirements.txt
```

Comprobación de dependencias, dataset y checkpoints, sin cargar modelos:

```powershell
& "..\OmniFood8K\Code\.venv\Scripts\python.exe" ".\scripts\verify_setup.py"
```

## Flujo completo

Definí una abreviatura para no repetir la ruta del intérprete:

```powershell
$py = "..\OmniFood8K\Code\.venv\Scripts\python.exe"
```

### 1. Preparar etiquetas y splits

Lee las masas por ingrediente de Nutrition5k, conserva los 49 ingredientes más
frecuentes y agrupa el resto como `other`. Usa el split de test oficial y obtiene
validación únicamente desde train, para no contaminar la evaluación.

```powershell
& $py ".\scripts\prepare_nutrition5k.py"
```

Los nombres elegidos quedan documentados en
`data/processed/vocabulary.json`. Se puede cambiar la cantidad con `top_k` en
`config/default.json` y volver a ejecutar desde este paso.

### 2. Extraer características visuales una sola vez

```powershell
& $py ".\scripts\extract_features.py" --device cpu --batch-size 16 --workers 0
```

Este es el paso más lento en CPU, pero se ejecuta una sola vez. Guarda tres archivos
`.npz` en `data/cache/`. Después, las épocas de entrenamiento ya no vuelven a leer
ni procesar las imágenes.

Si más adelante hay una GPU CUDA disponible, se puede reemplazar `--device cpu` por
`--device cuda` sin cambiar el resto.

### 3. Entrenar los cabezales pequeños

```powershell
& $py ".\scripts\train_heads.py" --device cpu
```

Se guarda el mejor modelo en `artifacts/ingredient_heads/best_heads.pt`, su historial
en `history.csv` y se aplica *early stopping*.

La variante regularizada reduce el cabezal de 384 a 192 neuronas y aumenta dropout y
weight decay. Además guarda checkpoints separados según F1, error por ingrediente y
error de masa total:

```powershell
& $py ".\scripts\train_heads.py" `
  --config ".\config\regularized.json" `
  --output-dir ".\artifacts\ingredient_heads_regularized" `
  --device cpu
```

### 4. Evaluar

```powershell
& $py ".\scripts\calibrate_threshold.py" --device cpu

& $py ".\scripts\evaluate.py" `
  --checkpoint ".\artifacts\ingredient_heads\best_heads_calibrated.pt" `
  --output-dir ".\outputs\evaluation_calibrated" `
  --device cpu
```

El primer comando elige el umbral de presencia usando solamente validación; no
reentrena el modelo. La evaluación genera
`outputs/evaluation_calibrated/metrics.json` y `predictions.csv`. Informa por separado:

- reconocimiento de ingredientes: precision, recall y F1;
- gramos de ingredientes y masa total con el cabezal liviano;
- composición usando la masa total verdadera, para aislar el error de ingredientes.

### 5. Inferir una imagen propia

Versión completamente liviana, usando también la masa total predicha por los nuevos
cabezales:

```powershell
& $py ".\scripts\infer_single.py" `
  --image "D:\ruta\a\mi_foto.jpg" `
  --device cpu
```

Versión recomendada, reutilizando la masa que ya predice OmniFood:

```powershell
& $py ".\scripts\infer_single.py" `
  --image "D:\ruta\a\mi_foto.jpg" `
  --omnifood-json "..\OmniFood8K\Code\inference_outputs\mi_foto\prediction.json" `
  --device cpu
```

La salida queda en `outputs/single/<nombre_imagen>/prediction.json` y
`ingredients.csv`.

## Qué se reutiliza y qué se entrena

| Módulo | Origen | ¿Se entrena? | Función |
|---|---|---:|---|
| EfficientNet-B0 | Preentrenamiento ImageNet de `torchvision` | No, queda congelado | Convertir RGB en 1280 características |
| Cabezal de presencia | Nuevo | Sí | Detectar varios ingredientes |
| Cabezal de proporciones | Nuevo | Sí | Repartir la masa entre ingredientes |
| Cabezal de masa total liviano | Nuevo | Sí | Alternativa rápida cuando no se usa OmniFood |
| OmniFood8K `ckpt_best.pth` | Entrenamiento de los autores | No | Dar una estimación de masa total más alineada con el proyecto previo |
| Tabla nutricional | Etiquetas Nutrition5k | No | Convertir gramos a nutrientes de manera determinista |

Con `top_k=50` se entrenan aproximadamente 0.53 millones de parámetros; el encoder
visual, que contiene la mayor parte del costo, nunca recibe gradientes. Esto hace
viable entrenar los cabezales en CPU y evita volver a generar RGB-D para Nutrition5k.

## Organización

```text
IngredientMassEstimation/
├── config/default.json          # hiperparámetros
├── data/raw/                    # descarga Nutrition5k, ignorada por Git
├── data/processed/              # registros y vocabulario generados
├── data/cache/                  # características congeladas
├── artifacts/                   # checkpoints propios
├── outputs/                     # métricas y predicciones
├── scripts/                     # comandos del flujo
├── src/ingredient_mass/         # dataset, arquitectura y métricas
├── tests/test_core.py           # pruebas sintéticas
└── requirements.txt
```

## Alcance y limitaciones

Este es un baseline experimental, no una calculadora clínica. Nutrition5k contiene
platos de cafeterías de California y muchas imágenes cenitales tomadas en condiciones
controladas; una foto de celular puede tener un cambio de dominio importante. Además,
ingredientes ocultos como aceite o salsas no se observan directamente: el modelo solo
puede inferirlos por correlaciones del entrenamiento. La clase `other` usa una densidad
nutricional promedio y es menos interpretable que las clases explícitas.

Para mejorar luego sin multiplicar el cómputo, el siguiente experimento razonable es
descongelar únicamente el último bloque de EfficientNet durante pocas épocas. Primero
conviene medir este baseline congelado y conservarlo como comparación.
