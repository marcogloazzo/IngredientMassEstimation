# Variante aprobada: ConvNeXt Small + Query2Label, 100 clases

Implementación: `src/ingredient_mass/spatial_model.py`, configuración:
`config/query2label_100.json`. Los scripts anteriores de características congeladas
siguen disponibles para comparar. Los checkpoints anteriores no sirven como pesos
del nuevo decoder/cabezales. La nueva variante requiere entrenamiento.

## Recorrido de una imagen

| Módulo | Salida con RGB 224 x 224, batch B | Entrenamiento |
|---|---|---|
| ConvNeXt Small, etapas iniciales | B x 384 x 14 x 14 | Congeladas y en modo evaluación |
| Último downsampling + etapa + normalización | B x 768 x 7 x 7 | Ajuste parcial después del warmup |
| Proyección espacial 1 x 1 | B x 49 x 256 | Desde inicialización aleatoria |
| Decoder de atención, 100 consultas | B x 100 x 256 | Desde inicialización aleatoria |
| Proyección a 768 canales | B x 100 x 768 | Desde inicialización aleatoria |
| MLP compartido 768 -> 384 -> 192 | B x 100 x 192 | Desde inicialización aleatoria |
| Clasificación por ingrediente + sigmoid | B x 100 probabilidades | ASL o BCE ponderada |
| Cantidad positiva, filtrada por probabilidad, normalizada | B x 100 proporciones de masa | Supervisión de proporciones y gramos |
| Promedio de representaciones por ingrediente -> 192 -> 128 -> 1 | B masas totales positivas | Error de masa total y gramos |
| Proporciones x masa total | B x 100 gramos | Incluye ingredientes ausentes |
| Gramos x tabla nutricional | B x 4 nutrientes | Tabla fija; pérdida nutricional ayuda a entrenar |

El MLP se aplica por ingrediente con los mismos pesos; no son 100 MLP separados.
Las salidas de presencia/cantidad sí tienen pesos específicos por ingrediente.
El promedio para masa total no pondera por umbral de presencia, evitando un corte
no diferenciable durante el entrenamiento. Las proporciones suman uno y los gramos
suman la masa total predicha.

## Qué significa usar Query2Label aquí

Adaptación compacta de la idea de Query2Label: consultas aprendidas por clase,
posiciones 2D, atención cruzada a mapas espaciales y salida específica por clase.
Un decoder de una capa, dimensión 256, 4 cabezas, FFN 1024; sin encoder Transformer
adicional ni self-attention entre consultas. Esta configuración reduce costo frente
a configuraciones grandes del repositorio oficial.

Se extiende a proporciones, masa y nutrientes y se añade el MLP elegido por el usuario.
No es una reproducción exacta de un experimento del paper y no carga directamente
sus checkpoints. No deben atribuirse a esta adaptación sus métricas publicadas.
No se exige una instalación del repositorio antiguo ni sus dependencias adicionales.

Referencias: [paper Query2Label](https://arxiv.org/abs/2107.10834),
[repositorio oficial](https://github.com/SlongLiu/query2labels),
[ConvNeXt Small en torchvision](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_small.html).

## Pérdidas

La configuración inicial suma:

`L = L_presencia + 5 L_proporciones + L_masa + 0.25 L_nutrición + L_gramos`

- ASL multilabel por defecto: gamma negativo 4, positivo 1, clipping 0.05;
  promedio sobre batch y clases (el repositorio original usa suma). El peso focal
  se desacopla del gradiente, configurable. No se agrega además `pos_weight` a ASL.
- Alternativa: `--presence-loss bce` usa BCE con `pos_weight`, calculado solo en
  entrenamiento, limitado al rango 1 a 20.
- Proporciones: L1 ponderada para clases presentes, como la implementación anterior.
- Masa total: MAE / masa media de los platos de **train**, calculada una sola vez.
  Se guarda como `mass_scale` en la configuración del checkpoint y se reutiliza
  para validación, evaluación y reanudación. No usa test/validación ni la media del lote.
  Checkpoints anteriores sin este campo mantienen su escala histórica de 300 g
  durante evaluación; no se reanudan silenciosamente con una pérdida diferente.
- Gramos: por cada plato, sumar `abs(proporción * masa_predicha - gramos_reales)`
  sobre las 100 clases; promediar platos y dividir por 300 g. Esta reducción evita
  que ampliar el vocabulario diluya artificialmente el error al promediar más ceros.
  Incluye explícitamente ingredientes ausentes y `other`; usa los gramos originales,
  no una reconstrucción de targets a partir de la masa total.
- Nutrientes: MAE normalizado por [500 kcal, 30 g grasa, 80 g carbohidratos, 40 g proteína].

Los pesos son iniciales, no valores óptimos demostrados. ASL/BCE tienen escalas y
balances distintos, por lo que se deben registrar las partes de pérdida y evaluar
con métricas comunes. No hay garantía de mejora por añadir capas o atención.

Referencia: [ASL, implementación oficial](https://github.com/Alibaba-MIIL/ASL).

## Entrenamiento

- Pesos iniciales ConvNeXt Small ImageNet-1K de torchvision (~192 MB); no se cargan
  automáticamente los pesos de OmniFood ni Depth Anything.
- Épocas 1-3: solo proyección, consultas, decoder y cabezales.
- A partir de época 4: también último downsampling (índice 6), última etapa (7)
  y normalización final de ConvNeXt. Las tres etapas anteriores siguen congeladas.
- AdamW: learning rate 3e-4 para módulos nuevos y 1e-5 para el bloque visual ajustado.
- Batch 8, acumulación de 4 lotes (batch efectivo habitual 32), AMP CUDA,
  clipping de gradiente 1, máximo 30 épocas, paciencia 8.
- Augmentación moderada solo en train (flip y color); geometría 230 -> crop 224 y
  normalización de los pesos torchvision. Validación/inferencia deterministas.
- Selección por menor error L1 de gramos por plato en VALIDACIÓN; también se
  registra F1, error de masa total y calorías. Test no participa en la selección.
- Sin caché de vectores finales: la atención necesita mapas y el ajuste parcial
  necesita recomputar la etapa visual entrenable. No ejecutar `extract_features.py`
  para esta variante.

El split actual se conserva para comparar: train 2332, validación 423, test 507.
Sigue pendiente auditar si escaneos relacionados de un plato quedan en diferentes
splits, lo que podría inflar métricas. No se ha establecido independencia por sesión.

## Archivos y métricas

`best.pt`: modelo completo y vocabulario, elegido por validación.
`last.pt`: además optimizer, AMP scaler, época y RNG para reanudar al final de una época.
`history.csv`: progreso resumido y tiempos; `latest_validation.json`: pérdidas y métricas.
Si Colab se interrumpe a mitad de época, se repite desde el último cierre guardado.
Con varios workers no se promete identidad bit a bit al reanudar.

Evaluación exporta `metrics.json`, `evaluation_details.json` y `predictions.npz`.
PMAE en este proyecto significa sum(error absoluto) / sum(target absoluto),
no promedio del porcentaje individual por imagen. El campo histórico
`end_to_end_lightweight_mass` se conserva por compatibilidad, aunque el nuevo
modelo visual no sea tan liviano.

Además del MAE de ingredientes presentes se reportan error L1 por plato, MAE de
todas las clases y gramos asignados a ingredientes ausentes. Comparar 50 y 100
clases requiere reagrupar las nuevas clases al vocabulario anterior, no comparar
directamente solo el MAE promedio por clase.

## Inferencia y mapas para la presentación

`infer_image_model.py --save-attention` exporta:

- `prediction.json`: las 100 clases, probabilidades, gramos y nutrientes.
- `intermediates/input_crop.png`: imagen que realmente recibió la red.
- `intermediates/tensors.npz`: entrada normalizada, mapas, representación de consultas,
  MLP, probabilidades, proporciones, masas y atención de la última capa.
- `intermediates/tensor_summary.json`: formas y estadísticas de cada salida.
- `intermediates/*_attention.png`: regiones consultadas por los ingredientes con
  más gramos predichos, promediadas entre las cabezas de atención.

Las superposiciones se alinean con el crop de entrada, no con toda la foto original.
Cada mapa se escala para visualizar su patrón espacial; colores de diferentes
ingredientes no comparan confianza. Atención NO es segmentación, una explicación
causal validada, ni evidencia de que un ingrediente oculto sea visible.

## Verificación realizada

Pruebas sintéticas sin descargar pesos: dimensiones de 100 clases, conservación
de masa, atención normalizada en evaluación, gradientes de la pérdida en gramos,
ASL/BCE, etapas congeladas, checkpoint sin descarga y exportación de mapas.
Se verifica la ruta de los 3262 registros locales. No se ejecuta entrenamiento
real, evaluación real ni benchmark GPU como parte de la implementación.

La inspección de la implementación real (sin pesos descargados y con imagen
sintética) confirma las formas de la tabla. Conteos de parámetros de esta variante:
51,098,473 totales; 1,643,785 módulos nuevos; 15,472,128 de encoder ajustables;
17,115,913 entrenables después del warmup. No se han medido GFLOPs del sistema
completo ni velocidad GPU; no confundir estos conteos con un resultado entrenado.
Se puede repetir con `python scripts/inspect_image_model.py`.
