# Colab V4: diagnosticar V3 y probar cabezales especializados

Abrir `COLAB_QUERY2LABEL_V4.ipynb`. V3 y su notebook ejecutado se conservaron;
no se inició entrenamiento/evaluación real de V4.

## Diagnóstico ya realizado en la PC

Se analizaron los archivos que enviaste (validación, época 39), sin cargar el
modelo. Resumen en [DIAGNOSTICO_GATE_V3.md](DIAGNOSTICO_GATE_V3.md).
Informe completo local: `outputs/v3_gate_diagnostic_epoch39_final/diagnostic.html` y JSON.

La cantidad condicional también subestima; quitar el gate no soluciona todo.
Recortar probabilidades mejora levemente L1, pero puede empeorar masa/calorías.
Por eso V4 conserva la regla `p × c` y cambia únicamente las dos propuestas
aprobadas: MLP separados y supervisión auxiliar de masa total.

## Git

Crear una rama nueva, por ejemplo `v4-specialized-heads`, desde la versión actual.
No se creó una rama ni se hicieron commits/push automáticamente.

Subir:

- `src/ingredient_mass/specialized_model.py` y `gate_diagnostics.py`.
- Cambios compatibles en `spatial_model.py`, `depth.py`, `image_training.py`.
- `scripts/diagnose_mass_gate.py` y cambios en train/eval/infer/inspect.
- Las tres configuraciones `config/query2label_100_v4*.json`.
- `tests/test_v4.py`, README, documentos V4 y diagnóstico V3.
- `COLAB_QUERY2LABEL_V4.ipynb`, limpio, sin resultados.

No incluir NPZ/PT/PTH, dataset, cachés, imágenes ni outputs. El `.gitignore`
existente los excluye. `COLAB_QUERY2LABEL_V3.ipynb` ya tenía cambios/salidas
locales del usuario: **no se editó ni limpió**; no incluirlo en el commit V4 si
querés conservar esa copia ejecutada solo localmente. No seleccionar todos los
cambios sin revisar Source Control.

El notebook apunta a esa rama: publicarla antes de clonar en Colab, o adaptar
BRANCH a la rama publicada. Se reutiliza `requirements-v3.txt`: no se agregaron
dependencias de entrenamiento.

## Drive

Conservar intactas las carpetas `v2/` y `v3/` y reutilizar:

```text
IngredientMassEstimation/
  nutrition5k_rgb.zip
  torch_cache/                          # si ya existe
  v2/                                   # histórico, no tocar
  v3/
    data/processed_100_grouped_provisional.zip
    depth_v3_320.zip
    weights/depth_anything_v2_vits.pth    # para inferencia de fotos, no para train con cache
    runs/query2label_100_v3_bce/          # histórico, no tocar
    reports/run_01/validation/
      predictions.npz
      evaluation_details.json
      metrics.json
  v4/                                   # se crea al generar resultados
    diagnostics/
    runs/split_only_run_01/
    runs/aux_only_run_01/
    runs/combined_run_01/
    reports/
    comparisons/
```

**No hace falta subir nuevamente datos o pesos existentes.** Podés abrir el
notebook V4 desde la PC en Colab o guardarlo en Drive. Opcionalmente subir el
diagnóstico local completo a una carpeta nueva `v4/diagnostics/local_epoch39/`.

## Orden de ejecución

1. Puntos 1–4: montar, clonar código publicado, dependencias y etiquetas originales.
2. Punto 5: `RUN_GATE_DIAGNOSTIC=True`, mostrar informe HTML. Solo lee archivos
   guardados; no requiere GPU, RGB ni cache. Si repetís, usar DIAGNOSTIC_TAG nuevo.
3. Punto 6: restaurar RGB y cache V3 existente. `RUN_DATA_CHECK=True` verifica
   hashes/grupos; no ejecuta el modelo. No regenerar etiquetas/mapas de V2.
4. Elegir VARIANT en punto 3: primero `split_only`. Reejecutar punto 3 cuando
   cambies variante/tags para actualizar rutas. Cada variante tiene carpeta propia.
5. **Solo cuando decidas entrenar**, punto 7: `RUN_TRAINING=True`, `MODE="new"`.
   Máximo 100 épocas y early stopping 8; puedes elegir un piloto menor y después
   `resume`, manteniendo el mismo scheduler/config. No resume desde V3.
6. Punto 8: `RUN_CALIBRATION=True` evalúa validación; no modifica gramos.
   Dejar `RUN_TEST=False` durante selección de variantes.
7. Punto 9: habilitar diagnóstico V4/comparación con V3 sobre validación.
   Los hashes deben coincidir para llamar controlada a la comparación.
8. Repetir con `aux_only`; luego `combined` si corresponde. Test solo para la
   opción elegida, sin ajustar parámetros según test.

En variantes con auxiliar, revisar si mejoran **las masas finales**, no solo
`auxiliary_total_mass.mae_grams`. La masa auxiliar no se usa para reescalar ni
predecir nutrientes.

## Diagnóstico sin notebook

En una sesión Colab ya preparada y desde el directorio del repositorio:

```python
subprocess.run([sys.executable, "scripts/diagnose_mass_gate.py",
    "--predictions", str(DRIVE / "v3/reports/run_01/validation/predictions.npz"),
    "--details", str(DRIVE / "v3/reports/run_01/validation/evaluation_details.json"),
    "--vocabulary", str(PROCESSED / "vocabulary.json"),
    "--output-dir", str(DRIVE / "v4/diagnostics/v3_gate_nuevo")], check=True)
display(HTML((DRIVE / "v4/diagnostics/v3_gate_nuevo/diagnostic.html").read_text()))
```

La salida debe ser nueva/vacía. Nunca usar NPZ de test para elegir umbrales.

## Verificación local

13 pruebas nuevas V4 aprobadas; suite completa: 83 aprobadas y una falla en la
prueba histórica que exige `COLAB_QUERY2LABEL_V3.ipynb` sin outputs. Esa copia ya
estaba ejecutada por el usuario antes de este trabajo y se preservó sin cambios.
No se entrenó V4 con datos reales ni se descargaron pesos para las pruebas.
