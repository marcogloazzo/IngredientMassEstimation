# IngredientMassEstimation V2: RGB multiescala y profundidad estimada

Configuración: `config/query2label_100_v2.json`.
Implementación: `multiscale_model.py` y `depth.py`.
La V1 y sus checkpoints siguen disponibles. La V2 requiere una corrida nueva:
no se carga ni se reanuda el decoder entrenado de V1.

## Cambios implementados

| Componente | V1 | V2 |
|---|---|---|
| ConvNeXt Small | Etapas 1–3 congeladas | Solo stem y etapa 1 congelados tras warmup |
| Ajuste visual | Último downsampling + etapa 4 + norma final | Downsamplings y etapas 2, 3, 4 + norma final |
| Mapas RGB | 768 × 7 × 7 | 384 × 14 × 14 y 768 × 7 × 7 |
| Profundidad | No | Depth Anything V2 Small congelado, desde el mismo RGB |
| Rama de profundidad | No | CNN nueva, entrenable, a dos escalas |
| Memoria de atención | 49 posiciones | 245 posiciones: 196 + 49 |
| Query2Label | 1 capa | 2 capas de atención cruzada |
| Consultas / ancho / cabezas | 100 / 256 / 4 | Sin cambios |
| MLP y cabezales | 768 → 384 → 192 | Sin cambios |
| Pérdida | ASL/BCE + proporciones + masa + nutrición + gramos | Sin cambios |

No se promete una mejora antes de entrenar y medir. Más capas y profundidad
estimada son hipótesis experimentales, no garantías.

## Recorrido detallado

Entrada externa: **una sola imagen RGB**. Se corrige EXIF, se redimensiona el
lado corto a 230 con interpolación bilinear y se hace center crop 224 × 224.
Esta geometría coincide con los pesos ConvNeXt Small de torchvision.

```text
RGB 224 × 224
  ├─ ConvNeXt Small, ImageNet
  │    stem → etapa 1 [congelados]
  │    → etapa 2 → etapa 3: 384 × 14 × 14
  │                    └→ etapa 4 + norma: 768 × 7 × 7
  │
  └─ Depth Anything V2 Small [congelado, sin gradientes]
       → profundidad relativa 1 × 224 × 224
       → CNN entrenable
            → 64 × 14 × 14
            → 128 × 7 × 7

Cada escala:
  RGB → Conv 1×1 a 256
  depth → Conv 1×1 a 256
  concatenar → Conv 1×1 (512→256) + GELU
  → aplanar: 196 tokens y 49 tokens

Memoria: 245 × 256
Posición: seno/coseno 2D + embedding aprendido de escala
100 consultas aprendidas × 256
  → atención cruzada + residual + LayerNorm + FFN
  → atención cruzada + residual + LayerNorm + FFN
  → proyección 256→768
  → LayerNorm + MLP 768→384→192

Representaciones por ingrediente
  ├─ presencia: pesos por clase + sigmoid
  ├─ cantidad: pesos por clase + softplus
  │    sigmoid(presencia) × cantidad positiva → normalizar → proporciones
  └─ promedio de las 100 representaciones → MLP 192→128→1 + softplus → masa total

proporciones × masa total = gramos por ingrediente
gramos × tabla nutricional fija = calorías, grasa, carbohidratos y proteína
```

Las dos capas tienen 4 cabezas, ancho 256, FFN 1024 y dropout 0.1.
No se añade self-attention entre consultas ni un encoder Transformer adicional.
El MLP se comparte entre ingredientes; los cabezales de presencia/cantidad
mantienen parámetros por clase. Las proporciones suman uno y los gramos suman
la masa total predicha, sin cortar por el umbral de presencia.

La CNN de profundidad utiliza cinco bloques Conv 3×3, stride 2, padding 1,
GroupNorm de 8 grupos y GELU. Canales: 1→16→32→64→64→128.

## Profundidad desde RGB: entrenamiento y despliegue

Se utiliza el modelo **relativo Small/vits**, no profundidad métrica ni sensor
depth de Nutrition5k. No se cargan automáticamente los pesos Base/Large
existentes en OmniFood. El código admite esos encoders con otra configuración,
pero requieren pesos, cache y corrida propios.

Depth Anything se ejecuta con eval, inference_mode y requires_grad=False.
La salida se interpola a 224 × 224, se normaliza usando percentiles 2 y 98 de
cada imagen y se limita a [0,1]. No tiene unidades físicas ni permite obtener
volumen absoluto por sí sola.

Para no ejecutar el estimador de profundidad en cada época:

1. `scripts/prepare_depth.py` calcula un mapa por RGB con pesos congelados.
2. Guarda los mapas float16 y un manifest con configuración, revisión del código,
   SHA-256 de los pesos y de los RGB originales.
3. El DataLoader lee RGB y mapa desde almacenamiento local de Colab.
4. Flip horizontal idéntico en ambas ramas; color jitter únicamente al RGB.

No se necesitan imágenes de profundidad adicionales. Calcular mapas de test
no utiliza sus etiquetas ni ajusta ningún parámetro.
La arquitectura PyTorch recibe los tensores RGB y depth estimada ya preparados;
la CLI de inferencia recibe **solo la foto RGB** y ejecuta todo el pipeline.

El checkpoint de IngredientMassEstimation contiene ConvNeXt, CNN depth,
fusiones, atención y cabezales. **No incluye** los pesos congelados de Depth
Anything: para una nueva foto también se necesita ese archivo y su código.
Durante inferencia se libera Depth Anything antes de mover ConvNeXt a CUDA.
Se valida que los pesos de profundidad coincidan con los usados al entrenar.

Fuente oficial: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).
Revisión fijada: `a561b849ebae10a6f5ef49e26c83cbbcd36c71bf`.
Small se distribuye bajo Apache-2.0; revisar las licencias antes de cambiar
a Base/Large o redistribuir pesos.

## Pérdida: permanece igual

`L = L_presencia + 5 L_proporciones + L_masa + 0.25 L_nutrición + L_gramos`

- ASL por defecto: gamma negativo 4, positivo 1, clipping 0.05; BCE ponderada
  sigue disponible mediante `--presence-loss bce`.
- Proporciones: L1 ponderada, peso 5 para clases presentes y 1 para ausentes.
- Masa total: MAE dividido por la media de **train**, aproximadamente 219,11 g
  con el dataset actual; se calcula una vez y se guarda como mass_scale.
- Gramos: promedio de la suma de errores absolutos de las 100 clases por plato,
  dividido por **300 g**, incluyendo ingredientes ausentes y other.
- Nutrición: MAE normalizado por [500 kcal, 30 g, 80 g, 40 g].

No se agrega una pérdida depth: la rama usa profundidad estimada como
característica auxiliar. La tabla nutricional, las etiquetas y la selección
de las 100 clases no cambian.

## Entrenamiento y selección

- Épocas 1–3: ConvNeXt completo congelado; CNN depth, fusiones, consultas,
  dos capas de atención y cabezales entrenables.
- Desde época 4: también etapas 2–4, sus downsamplings y normalización final.
  Stem/etapa 1 y Depth Anything siguen congelados.
- AdamW: 3e-4 módulos nuevos, 1e-5 encoder ajustado, weight decay 0.001.
- Batch 4, acumulación 8: efectivo 32; AMP CUDA y clipping 1.
- Máximo 100 épocas, early stopping paciencia 8.
- Mejor checkpoint por **menor ingredient_mass_l1_per_plate de validación**.
  No se usa test para seleccionar.
- Se conserva el split actual para comparación directa: 2332 train,
  423 validation, 507 test con los mismos datos. La independencia de escaneos
  relacionados sigue pendiente de auditoría; este cambio no resuelve ese riesgo.
- V1 no es un checkpoint de reanudación válido para V2.

history.csv contiene train_loss, validation_loss y F1/errores **de validación**.
La calibración del umbral usa validación; el test final queda en metrics.json
en la carpeta de reportes de test. Un umbral diferente no recorta proporciones
ni altera gramos/nutrientes.

## Parámetros y costo

Inspección de ConvNeXt Small real con pesos aleatorios y entrada sintética:

| Conteo | Parámetros |
|---|---:|
| Modelo de masa V2 completo | 52.434.313 |
| Módulos nuevos, incluida CNN depth/fusión/atención | 2.979.625 |
| ConvNeXt ajustable tras warmup | 49.211.904 |
| Total entrenable tras warmup | 52.191.529 |

Depth Anything Small agrega aproximadamente 24,8 millones congelados y se
ejecuta fuera del modelo de masa. No está incluido en la tabla anterior.
El ajuste de tres etapas cuesta más que V1; batch 4 es un punto de partida,
no una garantía de memoria/tiempo en cualquier GPU.

No se midió rendimiento GPU ni se entrenó con Nutrition5k durante esta
implementación. La verificación incluye shapes, conservación de masa,
gradientes, congelamiento, alineación RGB/depth, checkpoints y exportación
de mapas. La integración Depth Anything se prueba con un modelo simulado;
sus pesos reales se descargarán y comprobarán al preparar el cache en Colab.

## Exportaciones visuales

Inferencia con `--save-attention` exporta RGB usado, mapa de profundidad,
tensores y atención de la segunda capa:

- attention_tokens: 100 × 245;
- attention_stage3: 100 × 14 × 14;
- attention_stage4: 100 × 7 × 7;
- atención combinada sobre 14 × 14, conservando la suma de pesos.

Se guardan overlays combinados y overlays nativos por escala en subcarpetas.
La atención promedia las 4 cabezas; **no es segmentación ni prueba de presencia**.
El notebook muestra RGB, profundidad, gramos, nutrientes y estos overlays.
