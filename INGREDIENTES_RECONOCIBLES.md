# Ingredientes reconocibles por el modelo

El modelo fue entrenado con **50 clases de salida**: 49 etiquetas específicas y una clase residual `other`. Los nombres en inglés son las etiquetas exactas almacenadas en `data/processed/vocabulary.json`.

| # | Etiqueta del modelo | Traducción |
|---:|---|---|
| 1 | `olive oil` | aceite de oliva |
| 2 | `salt` | sal |
| 3 | `garlic` | ajo |
| 4 | `pepper` | pimienta |
| 5 | `vinegar` | vinagre |
| 6 | `onions` | cebollas |
| 7 | `lemon juice` | jugo de limón |
| 8 | `arugula` | rúcula |
| 9 | `cherry tomatoes` | tomates cherry |
| 10 | `parsley` | perejil |
| 11 | `carrot` | zanahoria |
| 12 | `spinach (raw)` | espinaca cruda |
| 13 | `shallots` | chalotas |
| 14 | `broccoli` | brócoli |
| 15 | `mustard` | mostaza |
| 16 | `cucumbers` | pepinos |
| 17 | `mixed greens` | mezcla de hojas verdes |
| 18 | `thyme` | tomillo |
| 19 | `wheat berry` | grano de trigo |
| 20 | `chicken` | pollo |
| 21 | `cheese pizza` | pizza de queso |
| 22 | `white rice` | arroz blanco |
| 23 | `cauliflower` | coliflor |
| 24 | `lime` | lima |
| 25 | `bok choy` | pak choi |
| 26 | `jalapenos` | jalapeños |
| 27 | `brown rice` | arroz integral |
| 28 | `squash` | calabaza |
| 29 | `kale` | col rizada |
| 30 | `bell peppers` | pimientos morrones |
| 31 | `olives` | aceitunas |
| 32 | `chard` | acelga |
| 33 | `sweet potato` | batata |
| 34 | `beef` | carne vacuna |
| 35 | `caesar salad` | ensalada César |
| 36 | `scrambled eggs` | huevos revueltos |
| 37 | `cilantro` | cilantro |
| 38 | `bacon` | panceta / tocino |
| 39 | `fish` | pescado |
| 40 | `apple` | manzana |
| 41 | `pineapple` | ananá / piña |
| 42 | `mushroom` | hongo / champiñón |
| 43 | `millet` | mijo |
| 44 | `pork` | cerdo |
| 45 | `spinach (cooked)` | espinaca cocida |
| 46 | `egg whites` | claras de huevo |
| 47 | `lettuce` | lechuga |
| 48 | `roasted potatoes` | papas asadas |
| 49 | `berries` | frutos rojos |
| 50 | `other` | otros ingredientes agrupados |

> `other` no identifica cuál es el ingrediente: reúne todas las etiquetas de Nutrition5k que quedaron fuera de las 49 clases más frecuentes. La presencia de una clase en esta lista significa que el modelo puede producirla, no que vaya a reconocerla correctamente en cualquier fotografía.
