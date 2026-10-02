# Registro de experimentos

Todos los resultados corresponden al mismo split oficial de test. Las decisiones de
checkpoint y umbral se tomaron usando solamente validación.

| Variante | Parámetros entrenables | Precision | Recall | Micro-F1 | MAE ingrediente | MAE masa total | MAE calorías |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline, umbral 0.5 | 533,349 | 0.584 | 0.709 | 0.641 | **18.13 g** | 51.58 g | 99.94 kcal |
| Baseline calibrado, umbral 0.725 | 533,349 | 0.668 | 0.605 | 0.635 | 18.13 g | 51.58 g | 99.94 kcal |
| Regularizado, seleccionado por MAE de ingrediente | 268,005 | 0.620 | 0.670 | **0.644** | 18.74 g | 47.62 g | 99.89 kcal |
| Regularizado, seleccionado por MAE de masa total | 268,005 | 0.625 | 0.657 | 0.641 | 18.72 g | **47.53 g** | **99.23 kcal** |

## Lectura

- El modelo regularizado usa 49.7% menos parámetros que el baseline.
- La variante seleccionada por ingrediente obtiene el mejor Micro-F1 y mejora la
  masa total, pero el baseline sigue siendo mejor para los gramos de cada ingrediente.
- Subir el umbral aumenta precision y exact match, pero reduce demasiado el recall en
  test. Se conserva como modo conservador, no como modelo predeterminado.
- La mejora siguiente que puede cambiar sustancialmente los gramos requiere adaptar
  características visuales, no solamente los cabezales congelados. La opción de menor
  costo es descongelar el último bloque de EfficientNet durante pocas épocas.

## Checkpoints

- Baseline principal: `artifacts/ingredient_heads/best_heads.pt`
- Baseline conservador: `artifacts/ingredient_heads/best_heads_calibrated.pt`
- Regularizado equilibrado:
  `artifacts/ingredient_heads_regularized/best_ingredient_mass_heads.pt`
- Regularizado orientado a masa total:
  `artifacts/ingredient_heads_regularized/best_total_mass_heads.pt`

