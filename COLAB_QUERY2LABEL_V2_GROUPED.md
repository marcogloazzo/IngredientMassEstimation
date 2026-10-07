# Reentrenar V2 original con split agrupado, sin perder resultados

Notebook nuevo: `COLAB_QUERY2LABEL_V2_GROUPED.ipynb`.
Configuración nueva: `config/query2label_100_v2_asl_grouped.json`.
No se entrenó ni se generaron mapas/pesos. Todos los controles RUN están False.
La V2 histórica y los notebooks V2–V6 se preservan.

## Qué se conserva

Se confirmó la [configuración V2 publicada](https://github.com/marcogloazzo/IngredientMassEstimation/blob/3106056b4a64c68d9358b35753f1a2d2af1f01dd/config/query2label_100_v2.json).
Se copia sin cambiar arquitectura, pérdidas, hiperparámetros o geometría:

- ConvNeXt Small, stem/etapa 1 congelados; ajuste de etapas 2–4 tras 3 épocas.
- RGB resize230/crop224, Depth Anything Small congelado y CNN depth entrenable.
- Fusión RGB/depth por concatenación+Conv, no compuertas residuales V3/V6.
- 245 tokens (196+49), dos capas Cross-Attention y MLP 768→384→192.
- Cabezales de presencia/proporciones y masa total independiente de V2.
- **ASL**, gamma_neg4, gamma_pos1, clip0,05; no se sustituye por BCE.
- Batch4×accum8, LR heads3e-4/encoder1e-5, 100 épocas máximo, paciencia8 por L1.
- No se agrega scheduler cosine, CLIP, escalas por clase ni recorte de masa.

Solo se agregan flags de aceptación del split nuevo, namespace de protección y
referencia al commit original. La infraestructura actual conserva comportamiento
V2 y añade verificaciones/fingerprints; no se está entrenando V6 con nombre V2.

## Datos nuevos

Se reutiliza `v3/data/processed_100_grouped_provisional.zip`, sin regenerar
splits ni revisar imágenes manualmente. Train/validation agrupados, test oficial
preservado. El agrupamiento sigue siendo provisional, no independencia certificada.
Ese paquete cambia seis claves y perfiles nutricionales frente a la V2 histórica.
Es correcto para comparar contra V3–V6 con los mismos datos; no comparar diferencias
históricas como si únicamente hubiera cambiado una arquitectura.

La escala de pérdida de masa total se recalcula desde train (216,89 g en ese
paquete), no se fija a 219,11 g. Las pérdidas de ingredientes conservan 300 g.

**Nueva corrida desde ImageNet, no resume/warm-start desde V2 histórica.**
El vocabulario nuevo no es intercambiable con sus cabezales/checkpoints.

## Drive: separación completa

```text
IngredientMassEstimation/
  nutrition5k_rgb.zip                         # reutilizar
  torch_cache/                               # reutilizar ImageNet si existe
  v2/                                        # histórico: SOLO LECTURA
    weights/depth_anything_v2_vits.pth
    depth_v2_vits.zip                         # puede copiarse, no modificarse
    runs/ reports/ predictions/               # conservar completamente
  v3/data/processed_100_grouped_provisional.zip
  v3/ v4/ v5/ v6/                            # no tocar
  v2_grouped_asl/                             # NUEVO experimento
    weights/depth_anything_v2_vits.pth        # copia/descarga por sección 5
    depth_v2_grouped_224.zip                  # copia verificada/generación por sección 6
    runs/query2label_100_v2_asl_run_01/
    reports/run_01_eval_01/validation/
    reports/run_01_eval_01/test/
    comparisons/
    predictions/
```

No hace falta volver a subir datos. Abrir/subir el nuevo notebook en Colab.
Conservar el acceso directo a la carpeta compartida o adaptar DRIVE.

Los mapas de depth no dependen de las etiquetas ni de train/val. Se puede copiar
el ZIP histórico V2 **224×224** a un cache local nuevo y reutilizarlo solo tras
verificar todos los IDs, hashes RGB, geometría y pesos. Si faltan mapas, preparar
los faltantes en esa copia. Se guarda otro ZIP en v2_grouped_asl/, nunca sobre v2/.
Si la copia está corrupta o tiene otra configuración, usar un cache local nuevo.

**No usar los mapas320 de V3–V6 ni redimensionarlos a224:** parten de otra región
RGB/letterbox y no reproducen la profundidad estimada desde el crop de V2.

## Git

Crear/publicar rama sugerida `v2-grouped-asl` desde el código actual compatible.
No se cambió de rama ni se hizo commit/push automáticamente. La rama original
`v2-multiscale-depth` sigue intacta.

Subir:

- `COLAB_QUERY2LABEL_V2_GROUPED.ipynb` limpio, este documento y README.
- `config/query2label_100_v2_asl_grouped.json`.
- `tests/test_v2_grouped.py`.
- Cambios compatibles de protección/UI en `src/ingredient_mass/image_training.py`,
  `notebook_reports.py`, `scripts/train_image_model.py`, `evaluate_image_model.py`
  e `infer_image_model.py`.

No incluir el notebook V6 ejecutado ni otros notebooks históricos, datasets,
checkpoints, NPZ, cache, fotografías o resultados. `.gitignore` ya los excluye.
Reutilizar `requirements-v2.txt`; el notebook instala Matplotlib para visualización.
No necesita dependencias/pesos CLIP de V6.

## Orden en Colab

1. Nueva sesión con GPU, no actualizar paquetes mientras otra corrida entrena.
2. Secciones 1–4: Drive, rama publicada, dependencias, RGB y paquete agrupado V3.
3. Sección 5: código/pesos frozen Depth Anything (copia en carpeta nueva).
4. Sección 6: intenta restaurar copia de cache224. Si falta o es incompleto,
   `RUN_DEPTH_PREPARATION=True` lo genera/completa; no entrena el modelo de masa.
   Habilitar `RUN_DATA_CHECK=True` y verificar antes de entrenar.
5. **Cuando quieras entrenar**, sección 7: `RUN_TRAINING=True`, `MODE="new"`.
   TOTAL_EPOCHS=100 es techo; early stopping puede terminar antes.
6. Después, sección 8: `RUN_CALIBRATION=True`; dejar `RUN_TEST=False` durante
   selección de alternativas. El umbral de presencia V2 no cambia gramos.
7. Sección 9: errores por ingrediente y comparaciones con referencias disponibles
   V3–V6 en validación. Misma huella de records/vocabulario = comparación controlada.
8. Sección 10 opcional: inferencia con tabla real/estimado y atención, sin HTML.

Cambiar EXPERIMENT_TAG para otra corrida, REPORT_TAG para otra evaluación e
INFERENCE_TAG para nuevas predicciones; reejecutar sección 3 para actualizar rutas.
Para resume usar **solo** last.pt del nuevo control v2_grouped_asl, no el histórico.

## Protecciones

El notebook rechaza salidas fuera de v2_grouped_asl y carpetas no vacías para
corridas/evaluaciones/comparaciones/inferencias nuevas. Los scripts de train,
eval e infer también verifican el namespace del config: no permiten escribir
en v2/v3/v4/v5/v6 aunque se pase una ruta equivocada.
No se borra, mueve ni reemplaza ningún resultado previo.

## Verificación local

7 pruebas nuevas aprobadas: configuración V2 original conservada, geometría224,
estado de parámetros y forward idénticos con la misma seed en backbone simulado,
245 tokens, namespaces protegidos, rechazo de sobrescritura, notebook seguro y
etiquetas correctas en tablas nativas.
Suite de regresión: **115 aprobadas**, omitiendo 5 comprobaciones que exigen los
notebooks históricos V3–V6 sin ejecuciones/outputs y controles apagados.
Se preservaron esas copias ejecutadas: SHA-256 de notebooks V2–V6 y config V2
original idénticos antes/después. No se entrenó ni descargó/generó ningún peso/mapa.
