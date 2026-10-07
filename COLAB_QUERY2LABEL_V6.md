# V6 lista para Colab: Git, Drive y orden de ejecución

Notebook nuevo: `COLAB_QUERY2LABEL_V6.ipynb`. Todos los controles RUN están False.
No se inició entrenamiento, inferencia de alimentos, descarga de pesos ni CLIP
real en la PC. Las verificaciones usan tensores/cache sintéticos.
Ver [arquitectura V6](ARQUITECTURA_QUERY2LABEL_V6.md).

## Qué cambia

ASL sin corrección BCE; solo etapa 4 ajustable (sin LoRA); escalas condicionales
por clase; dos compuertas espaciales 2D; consultas inicializadas con CLIP; filtro
de masa por probabilidad al evaluar/inferir y calibrado en validación.
Se mantiene MLP compartido y auxiliar V5, RGB como único input, los datos
agrupados y los mapas depth de V3. No se regeneran etiquetas ni se revisan imágenes.

Los resultados se muestran con tablas/gráficos nativos, **sin HTML**, incluso en
comparaciones y diagnóstico. La sección 10 muestra atención y compuertas.

## Git: crear/publicar una rama nueva

Nombre sugerido: `v6-asl-spatial-clip`, desde la versión actual. No se hizo cambio
de rama, commit ni push automático. El punto 2 debe apuntar a una rama publicada.

Subir los archivos NUEVOS:

- `COLAB_QUERY2LABEL_V6.ipynb`, limpio/sin salidas.
- `config/query2label_100_v6.json` y `requirements-v6.txt`.
- `src/ingredient_mass/robust_model.py`, `clip_queries.py`, `mass_filtering.py`.
- `scripts/prepare_clip_queries.py` y `tests/test_v6.py`.
- Documentos V6 y README.

Subir también los cambios compatibles en:

- `src/ingredient_mass/spatial_model.py`, `multiscale_model.py`, `image_training.py`.
- `src/ingredient_mass/gate_diagnostics.py` y `notebook_reports.py`.
- `scripts/train_image_model.py`, `evaluate_image_model.py`, `infer_image_model.py`.

No seleccionar todos los cambios sin revisar: tus notebooks V3/V4/V5 ejecutados
y figuras V5 pendientes no forman parte de este commit. Se conservaron.
No subir datasets, pesos, caches, NPZ, PT/PTH, fotos ni reportes; `.gitignore`
ya los excluye. El cache CLIP es un NPZ, no código para Git.

## Drive: no volver a subir datos existentes

Conservar completamente v2/v3/v4/v5. Reutilizar:

```text
IngredientMassEstimation/
  nutrition5k_rgb.zip
  torch_cache/                                # ImageNet, si ya existe
  v3/data/processed_100_grouped_provisional.zip
  v3/depth_v3_320.zip
  v3/weights/depth_anything_v2_vits.pth         # solo para inferir fotos RGB
  v3/reports/run_01/validation/                # comparación
  v4/reports/split_only_run_01_eval_01/validation/
  v5/reports/run_01_eval_01/validation/         # comparación, si disponible
  v6/                                        # se crea con los resultados nuevos
    weights/clip/ViT-B-32.pt                   # descarga una vez, por celda 5
    weights/clip_queries.npz                  # consultas, por celda 5
    runs/query2label_100_v6_asl_run_01/
    reports/run_01_eval_01/validation/
    reports/run_01_eval_01/test/
    diagnostics/
    comparisons/
    predictions/
```

Solo abrir/subir el nuevo ipynb a Colab/Drive. Los pesos/cache de CLIP se generan
automáticamente **cuando habilites la sección 5**; no hay que prepararlos en la PC
ni requiere API key. Si cambiás vocabulario/modelo CLIP, usar otro cache nuevo,
no sobreescribir el anterior. El cache se verifica también al reanudar training.

Si la carpeta es compartida, mantener el acceso directo en Mi unidad y la misma
cuenta con permisos, o adaptar DRIVE en sección 3. No cambia este flujo.

## Orden de ejecución

1. Sesión Colab nueva con GPU. No actualizar paquetes mientras otra corrida corre.
2. Secciones 1–3: Drive, repo V6 publicado y dependencias. El helper de gráficos
   se carga desde archivo para evitar el error de imports editables del kernel.
3. Sección 4: restaurar SOLO el paquete de etiquetas V3 y verificar hashes.
4. Sección 5: primera vez, `RUN_CLIP_PREPARATION=True`. Descarga CLIP congelado
   y codifica nombres; **no entrena IngredientMassEstimation**. Luego puede volver
   a False: el NPZ existente se valida sin volver a cargar CLIP.
5. Sección 6: restaurar RGB y depth V3; `RUN_DATA_CHECK=True`. Verifica datos,
   grupos/cache/CLIP sin descargar ImageNet ni ejecutar el modelo.
6. **Cuando decidas entrenar**, sección 7: `RUN_TRAINING=True`, `MODE="new"`.
   Máximo 100 épocas, early stopping 8 por L1 filtrado en validación. Nunca cargar
   last.pt de V5. Para continuar V6, `MODE="resume"` con su propio last.pt.
7. Sección 8: después de entrenar, `RUN_CALIBRATION=True`. Calibra presencia y
   filtro de masa por separado. Mantener `RUN_TEST=False` al elegir arquitectura.
8. Sección 9: `RUN_V6_DIAGNOSTIC=True` y `RUN_COMPARISON=True` para informes
   guardados, comparados con referencias disponibles V3/V4/V5. No entrena baselines.
9. Sección 10: `RUN_INFERENCE=True` solo si querés inferir una foto; por defecto
   elige validación. No necesita CLIP en runtime; sí los pesos/código depth V3.

Cambiar EXPERIMENT_TAG para una corrida nueva, REPORT_TAG para otra evaluación
y INFERENCE_TAG para otra foto/corrida. Reejecutar sección 3 para actualizar rutas.
Informes y diagnósticos deben usar carpetas nuevas para no perder resultados.

## Qué mirar

No comparar el valor de ASL con BCE como si fueran la misma pérdida.
Evaluar L1/MAE presente, falsos gramos, masa total, nutrientes y F1 en el mismo split.
V6 guarda métricas **con y sin filtro**, masas eliminadas, platos vacíos, escalas,
promedios de compuertas, pérdidas separadas y MAE de la masa auxiliar.
Un buen auxiliar solo no garantiza buenos gramos. Un corte que reduce falsos
puede empeorar ingredientes reales: la calibración incluye guardas de masa/calorías.
La tolerancia predeterminada de +5% es respecto al modelo V6 sin recorte, no a V5.

El umbral de masa puede quedar en 0 si recortar perjudica. Esto no es un fallo de
implementación: es el resultado de selección en validación. Nunca ajustar con test.

## Verificación realizada

15 pruebas nuevas V6 aprobadas: escalas solo de train, ASL sin offsets BCE,
congelamiento/eval de E1–E3, gradientes de E4/consultas/compuertas, máscaras 2D,
masas exactamente cero, caché CLIP validada con encoder simulado, roundtrip de
checkpoint sin CLIP, calibración/guardas solo en validación y notebook seguro.

Suite de regresión: **109 aprobadas**, omitiendo 4 comprobaciones que exigen
notebooks históricos V3/V4/V5 sin ejecuciones/outputs y controles apagados.
Esas copias ya estaban ejecutadas por el usuario y se preservaron. Sus SHA-256
y los de las configuraciones V3/V4/V5 quedaron idénticos.

No se entrenó con Nutrition5k ni se descargaron pesos CLIP/ImageNet/depth en la
PC. El primer uso real de CLIP y el entrenamiento GPU se ejecutarán cuando
habilites las respectivas celdas en Colab.

Fuentes de implementación: [OpenAI CLIP](https://github.com/openai/CLIP),
[ASL](https://github.com/Alibaba-MIIL/ASL). CLIP se instala desde una revisión fija,
no desde una dependencia cambiante sin identificar.
