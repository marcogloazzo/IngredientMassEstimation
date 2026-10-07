# Colab V5, Git y Drive

Abrir `COLAB_QUERY2LABEL_V5.ipynb`. No se entrenó V5 ni se modificaron los notebooks
ejecutados V3/V4. Todos los controles RUN están en False.

V5 prueba **MLP compartido + cabezal auxiliar de masa total**, no ambos cambios
combinados de V4. Ver [arquitectura V5](ARQUITECTURA_QUERY2LABEL_V5.md).

## Git

Crear/publicar una rama nueva, sugerida `v5-auxiliary-mass`, desde la versión
actual V4. No hubo cambio de rama, commit ni push automático.

Subir solamente los archivos de esta implementación:

- `config/query2label_100_v5.json`.
- `COLAB_QUERY2LABEL_V5.ipynb`, limpio, y documentos V5.
- `tests/test_v5.py` y README.
- Cambios en `src/ingredient_mass/spatial_model.py`, `specialized_model.py`,
  `image_training.py` y scripts `train_image_model.py`, `evaluate_image_model.py`,
  `infer_image_model.py` para reconocer V5, registrar pérdidas y proteger resultados.
- `src/ingredient_mass/notebook_reports.py`: visualización nativa, sin HTML.
- `tests/test_notebook_reports.py`: pruebas de tablas/gráficos y lectura sin cambios.

Revisar Source Control: **no incluir tus notebooks V3/V4 ejecutados** en el commit
V5. No subir NPZ, PT/PTH, dataset, mapas, imágenes ni outputs; ya están ignorados.
No se agregaron dependencias de entrenamiento: reutilizar `requirements-v3.txt`.
Para mostrar resultados, el notebook instala además Matplotlib (no afecta train).

## Drive

Conservar `v2/`, `v3/`, `v4/` completas. Solo abrir/guardar el notebook nuevo V5.
No hace falta volver a subir RGB, etiquetas, pesos ni mapas.

```text
IngredientMassEstimation/
  nutrition5k_rgb.zip
  v3/data/processed_100_grouped_provisional.zip
  v3/depth_v3_320.zip
  v3/reports/run_01/validation/                # control
  v4/reports/split_only_run_01_eval_01/validation/ # control
  v5/                                        # resultados nuevos
    runs/query2label_100_v5_aux_run_01/
    reports/run_01_eval_01/validation/
    reports/run_01_eval_01/test/               # solo cuando se decida evaluar test
    diagnostics/
    comparisons/
```

Si la carpeta es compartida, mantener el mismo acceso directo de Drive en Mi
unidad o adaptar DRIVE en punto 3. El runtime puede reutilizar cache/data de V3;
no escribe resultados en las versiones anteriores.

## Orden

1. Puntos 1–4: Drive, rama publicada, dependencias y etiquetas originales.
2. Punto 5: diagnóstico V3 **opcional**. Ya lo hicimos; no es requisito para train.
3. Punto 6: restaurar RGB/cache y habilitar RUN_DATA_CHECK=True. No ejecuta modelo.
4. **Cuando decidas entrenar**, punto 7: RUN_TRAINING=True, MODE="new".
   TOTAL_EPOCHS=100 es máximo, early stopping paciencia 8. No ejecutar resume desde
   V3/V4. GPU recomendada para la corrida prevista: verificar DEVICE="cuda".
5. Tras entrenar, punto 8: RUN_CALIBRATION=True, mantener RUN_TEST=False mientras
   se selecciona arquitectura. La calibración no cambia gramos.
6. Punto 9: RUN_V5_DIAGNOSTIC=True y RUN_COMPARISON=True para tablas y gráficos del diagnóstico
   y comparaciones contra V3 y V4 split_only, usando validación guardada.

Cada evaluación/diagnóstico/comparación debe usar carpetas nuevas. Cambiar
EXPERIMENT_TAG para otro entrenamiento, REPORT_TAG para otra evaluación o
DIAGNOSTIC_TAG para repetir diagnóstico V3; reejecutar punto 3 para actualizar rutas.
Para continuar **esta misma V5**, MODE="resume" con su propio last.pt.

## Resultados relevantes

`history.csv` agrega pérdidas aux_mass/mass_consistency activas y columnas
auxiliary_total_mass_mae/mass_consistency_gap_mae. Estas deben existir en V5,
a diferencia de split_only de V4. `metrics.json` informa el auxiliar aparte.

La mejora debe verse en gramos finales y nutrientes, no solo en la masa auxiliar.
No se cambió el selector de checkpoint para favorecer otra métrica a posteriori.

## Visualización sin HTML

Los puntos 5, 8 y 9 leen los JSON guardados y muestran tablas/gráficos de Matplotlib
directamente en la celda. No usan `display(HTML(...))` ni abren páginas HTML.
Los archivos HTML que generen los scripts pueden conservarse, pero no se muestran.
Incluye métricas finales/auxiliares, contrafactuales, los 15 ingredientes de mayor
error, grupos TP/FN/FP/TN y comparaciones V3/V4 contra V5.
Si falta un reporte, se informa su ruta en texto; no se ejecuta el modelo para crearlo.
Verificación de esta actualización: 14 pruebas aprobadas (6 de visualización y
8 de V5), con revisión visual de tablas renderizadas. La celda de training se
conservó idéntica.

## Verificación local

8 pruebas nuevas V5 aprobadas. Suite de regresión: 90 aprobadas, excluyendo dos
pruebas históricas que exigen los notebooks V3/V4 sin ejecuciones/outputs.
Se preservaron esas copias ejecutadas y sus SHA-256 quedaron idénticos.
No se entrenó con Nutrition5k, no se descargaron pesos ni se regeneraron mapas.
