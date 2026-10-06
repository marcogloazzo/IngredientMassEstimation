# Diagnóstico real del producto p × c — V3, época 39

Análisis offline de los archivos suministrados `predictions.npz` y
`evaluation_details.json`: validación, 400 imágenes, 100 clases, checkpoint
seleccionado en época 39. No se cargaron pesos ni se ejecutó inferencia/training.
El SHA-256 del vocabulario local coincide con el del reporte.

Checkpoint: `3d91b9a7ef460a510060466e2f8064d150ea1fd7c1c7a21e59ab703324209b81`.
Vocabulario: `61bcc0b71d4ec1deebdba2279fc0408bbb28912e5c55a6c525dec22d4116cc21`.

## Resultados

| Regla de masa | L1 ingredientes g/plato | Gramos falsos/plato | MAE masa total g | MAE calorías kcal |
| --- | ---: | ---: | ---: | ---: |
| V3 guardada: p × c | 160,26 | 36,81 | 62,15 | 85,39 |
| Oracle: quitar solamente clases ausentes | 123,45 | 0,00 | 85,76 | 104,11 |
| Oracle: presencia verdadera × c | 114,59 | 0,00 | 72,70 | 87,75 |
| Recortar p × c cuando p < 0,35 | 158,42 | 33,99 | 64,41 | 88,12 |
| Usar c cuando p ≥ 0,35, cero en el resto | 164,53 | 42,79 | 58,31 | 84,29 |
| Mejor recorte de p × c en esta validación: τ=0,75 | 155,25 | 24,85 | 76,12 | 100,18 |

Los escenarios oracle usan etiquetas reales: no son reglas de inferencia ni
límites inferiores garantizados para otras métricas. El umbral 0,75 fue elegido
en el mismo conjunto reportado; la mejora de 3,13% de L1 es exploratoria,
no una mejora independiente ni un resultado de test. Ninguna regla fue aplicada
a un checkpoint o a test.

En las 3.232 entradas de ingredientes realmente presentes:

- MAE condicional `c`: **14,18 g/ingrediente**.
- MAE final `p × c`: **15,28 g/ingrediente**.
- Sesgo condicional: **−7,57 g/ingrediente**; final: **−9,64 g/ingrediente**.
- El gate empeora el error presente en **67,70%** de esas entradas; el efecto
  neto es **8,85 g/plato** adicionales de L1 presente, manteniendo `c` fijo.
- Con el umbral calibrado 0,35, los FN tienen `c` medio de **2,45 g** y MAE
  condicional **16,82 g**: no falla solamente la multiplicación por `p`.
- Los FP reconocidos aportan aproximadamente **33,99 de 36,81 g** falsos/plato:
  eliminar únicamente probabilidades pequeñas no resuelve la mayor parte de
  la masa fantasma.

## Interpretación y decisión

Hay subestimación tanto en `c` como en el resultado final. La masa fantasma
compensa parte de esa subestimación en masa total y nutrientes: eliminarla puede
mejorar composición y empeorar métricas globales. No conviene optimizar una sola
métrica con un recorte indiscriminado.

Esto justifica probar MLP especializados y supervisión auxiliar de masa total,
pero no demuestra conflicto de gradientes ni asegura una mejora. V4 mantiene
`p × c`, BCE, encoder, fusión y profundidad como controles; agrega solo esas dos
modificaciones, con configuraciones para probarlas separadamente.

## Reproducir en CPU o Colab

```bash
python scripts/diagnose_mass_gate.py \
  --predictions /ruta/validation/predictions.npz \
  --details /ruta/validation/evaluation_details.json \
  --vocabulary /ruta/processed_100_grouped_provisional/vocabulary.json \
  --output-dir /ruta/nueva/gate_diagnostic
```

La carpeta de salida debe ser nueva/vacía. El script rechaza test, archivos sin
cantidades condicionales, claves incompatibles, fingerprints distintos y masas
que no correspondan a `p × c`. Genera `diagnostic.json` y `diagnostic.html` con
tarjetas, contrafactuales, ranking por ingrediente y errores TP/FN/FP/TN.

El informe local completo quedó en `outputs/v3_gate_diagnostic_epoch39_final/` (ignorado
por Git). El diagnóstico no importa torch ni ejecuta el modelo.
