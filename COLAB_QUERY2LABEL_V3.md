# Preparar V3 para Colab, preservando V2

Notebook nuevo: `COLAB_QUERY2LABEL_V3.ipynb`. El notebook V2 y sus resultados
locales no se modifican. Esta entrega NO ejecuta entrenamiento real ni genera
depth de todo el dataset. Todas las operaciones costosas están desactivadas
en el notebook V3; habilitarlas manualmente cuando corresponda.

## Qué subir a Git

Crear/publicar una rama separada, por ejemplo `v3-conditional-mass`. El notebook
espera ese nombre en BRANCH; se puede adaptar a la rama realmente publicada.
No se creó ninguna rama, commit ni push automáticamente.

Desde IngredientMassEstimation, incluir:

- `COLAB_QUERY2LABEL_V3.ipynb` (sin salidas), este documento y
  `ARQUITECTURA_QUERY2LABEL_V3.md` y `README.md` actualizado.
- `config/query2label_100_v3.json`, `config/query2label_100_v2_bce_grouped.json`,
  `requirements-v3.txt`, `.gitignore`.
- `src/ingredient_mass/conditional_model.py` y los cambios compatibles en
  `spatial_model.py`, `multiscale_model.py`, `depth.py`, `image_training.py`.
- Los cambios de `train_image_model.py`, `evaluate_image_model.py`,
  `infer_image_model.py` y `inspect_image_model.py`, `prepare_depth.py`.
- Nuevos scripts `package_grouped_data.py`, `archive_experiment.py` y
  `compare_image_experiments.py`.
- `tests/test_v3.py` y ajustes de pruebas de notebooks V2 que ahora admiten
  salidas históricas ejecutadas sin exigir su eliminación.
- Si todavía no están publicados: scripts/módulo/pruebas de agrupamiento,
  `AUDITORIA_SPLITS_AGRUPADOS.md`, diagnóstico V2 y el import perezoso de
  `src/ingredient_mass/__init__.py` de la etapa anterior.

NO subir los cambios locales de los notebooks V1/V2 con resultados como parte
de esta V3 por accidente. Conservarlos localmente/Drive como evidencia. No
subir dataset, ZIP de etiquetas, cache, `.venv-v3`, pesos, runs/reports/predictions.
Revisar Source Control y seleccionar archivos explícitos, no todos a ciegas.

## Qué subir o conservar en Drive

```text
IngredientMassEstimation/
├── nutrition5k_rgb.zip                  existente, se reutiliza
├── photos/                             existente, fotos + annotations.json
├── v2/                                 CONSERVAR COMPLETA
│   ├── runs/                           best.pt, last.pt, history.csv, config
│   ├── reports/                        metrics, calibrated.pt, predictions
│   ├── predictions/
│   ├── weights/
│   └── depth_v2_vits.zip
├── backups/
│   └── v2_before_v3.zip                respaldo opcional, no sobrescribir
└── v3/
    ├── data/
    │   └── processed_100_grouped_provisional.zip   SUBIR desde esta PC
    ├── weights/                        puede copiar vits desde v2/weights/
    ├── depth_v3_320.zip                se genera en Colab una sola vez
    ├── runs/query2label_100_v3_bce/
    ├── reports/run_01/validation/ y test/
    ├── predictions/run_01/
    ├── comparisons/
    └── baselines/v2_bce_grouped/        opcional, referencia controlada
```

El ZIP listo para subir es `data/processed_100_grouped_provisional.zip`. Solo
contiene etiquetas/perfiles/IDs/manifiestos; no duplica RGB. Su packaging no
cambia el split existente: 2355 train, 400 validation y 507 test.

Si se regenera el paquete, elegir un archivo NUEVO:

```powershell
py -3.12 scripts/package_grouped_data.py --output data/processed_100_grouped_provisional_new.zip
```

No hace falta subir a Drive las 28 páginas de revisión para usar el provisional.
Guardar el manifiesto agrupado incluido en el ZIP para reproducibilidad.
No se pueden usar mapas 224 de V2 como cache de V3: geometría 320 diferente.
Los pesos congelados de Depth Anything vits sí se pueden reutilizar.

Si la carpeta está compartida, usar un acceso directo en Mi unidad de la cuenta
conectada a Colab o adaptar DRIVE. La extensión de VS Code debe seleccionar
el kernel Colab, no el entorno local base.

## Qué ejecutar en Colab

1. Abrir el notebook V3, conectar GPU y montar Drive.
2. Clonar la rama publicada; instalar dependencias y definir rutas propias.
3. Restaurar RGB y el ZIP de etiquetas del punto 4. **No ejecutar la preparación
   individual de Nutrition5k sobre esos datos**, ni recalcularlos con otro split.
4. Opcional: RUN_BACKUP_V2=True en punto 5, solo cuando V2 NO esté entrenando.
   El respaldo es un ZIP nuevo con hashes; no mueve ni borra el original.
5. Preparar código/pesos depth. Primera vez RUN_DEPTH_PREPARATION=True en 7.
   Luego se reutiliza depth_v3_320.zip con la bandera False.
6. Revisar controles de datos/config/cache. RUN_SHAPE_CHECK opcional usa un
   modelo sintético sin pesos ImageNet ni entrenamiento.
7. Para comenzar cuando lo decidas: RUN_TRAINING=True, MODE="new", máximo 100.
   Para smoke test se puede elegir 5 y luego MODE="resume" hasta 100.
8. Al finalizar, punto 9: RUN_CALIBRATION=True para validation y después
   RUN_TEST=True para test. REPORT_TAG debe ser nuevo si se reevalúa.
9. Punto 10: inferencia y tarjetas real/estimado; punto 11: diagnóstico guardado.

`last.pt` de V3 es el único checkpoint para reanudar V3. No reanudar V2 ni
inicializar V3 con su checkpoint: ya vio parte de la nueva validación y cambia
la arquitectura/vocabulario. El encoder se inicializa con ImageNet, como antes;
los cabezales nuevos se inicializan solo con estadísticas del NUEVO train.

El scheduler conserva su horizonte original 100 al reanudar un chequeo corto.
Si se extiende más allá de 100, seguirá en su LR mínimo; no cambiar parámetros
del scheduler a mitad de resume. La configuración salvo epochs debe coincidir.

## Preservación y comparación de V2

V3 tiene model_type/config/carpetas propios. Se rechaza escribir train/eval/cache/
predicciones V3 dentro de una ruta llamada v2; los reportes V3 existentes no
se sobrescriben. Los notebooks anteriores y los archivos preparados originales
se conservaron byte a byte en esta implementación.

La comparación histórica V2 vs V3 cambia también split y seis clases/perfiles:
no demuestra por sí sola qué cambio arquitectónico ayudó. Para referencia más
controlada, el punto 12 entrena OPCIONALMENTE V2+BCE desde cero con el mismo ZIP
agrupado; usa el cache 224 existente y guarda todo en v3/baselines/.

Para evaluar ese baseline, después de entrenarlo:

```python
BASELINE_VALIDATION = V3 / "baselines/v2_bce_grouped/reports/run_01/validation"
BASELINE_TEST = V3 / "baselines/v2_bce_grouped/reports/run_01/test"
shared_baseline = ["--dataset-root", DATA, "--processed-dir", PROCESSED,
                   "--depth-cache", BASELINE_DEPTH, "--device", DEVICE, "--batch-size", "4", "--workers", "2"]
run_stream([sys.executable, "-u", "scripts/evaluate_image_model.py", "--checkpoint", BASELINE_RUN / "best.pt",
            "--split", "validation", "--calibrate", "--output-dir", BASELINE_VALIDATION, *shared_baseline])
run_stream([sys.executable, "-u", "scripts/evaluate_image_model.py", "--checkpoint", BASELINE_VALIDATION / "calibrated.pt",
            "--split", "test", "--output-dir", BASELINE_TEST, *shared_baseline])
```

No ejecutarlo mientras el kernel esté ocupado entrenando. En punto 13 usar
LEFT_REPORT=BASELINE_TEST para comparar con V3. El comparador verifica
fingerprints y rechaza mezclar validation con test. La igualdad de datos hace
la comparación más controlada, pero cambiar varios componentes simultáneamente
no permite atribuir ganancias a uno solo. Usar validación para seleccionar;
test se reserva para evaluación final.

## Verificación local (sin entrenar el dataset)

```powershell
.venv-v3/Scripts/python.exe -m pytest tests -q
```

Se comprueban formas, conservación de masa, gradientes de cantidad con gates
bajos, congelamiento, geometría RGB/depth, checkpoints, scheduler/resume,
huellas de datos, backups y seguridad del notebook. No se validó entrenamiento
real ni rendimiento de GPU/Colab; esas comprobaciones quedan para la corrida.
