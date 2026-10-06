# V3: presencia + gramos condicionales

Implementación: `src/ingredient_mass/conditional_model.py`.
Configuración: `config/query2label_100_v3.json`.
Esta variante todavía NO se entrenó; no hay resultados V3 ni mejora demostrada.

## Cambios frente a V2

| Componente | V2 histórica | V3 |
|---|---|---|
| Datos | Split individual | Split agrupado provisional, aceptado explícitamente |
| Entrada | Resize 230 + crop central 224 | Imagen completa con letterbox 320×320 |
| Features | Etapas 3 y 4: 14×14 / 7×7 | Etapas 3 y 4: 20×20 / 10×10 |
| Tokens | 245 | 500 |
| RGB/depth | Concatenación + convolución + GELU | RGB normalizado + puerta aprendida × depth normalizado |
| Presencia | ASL en la corrida histórica | BCEWithLogitsLoss con pesos positivos de train |
| Cantidad | Factor relativo, multiplicado por presencia y normalizado | Gramos condicionales, con supervisión directa en positivos |
| Masa total | Cabezal separado sobre promedio de consultas | Suma de gramos finales de ingredientes |
| Pérdidas | Presencia, shares, masa, nutrientes, L1 gramos | Presencia, cantidad condicional, gramos falsos, L1 gramos, masa, nutrientes |
| LR | Constante | Warmup + cosine, horizonte fijo de 100 épocas |

Se conservan ConvNeXt Small, stem/etapa 1 congelados, ajuste de etapas 2–4
después de 3 épocas, 100 clases, dos capas Query2Label con 4 heads y dimensión
256, y MLP compartido 768→384→192. No se agranda el encoder.

## Salidas y unidades

Cada consulta produce un logit de presencia `z_i` y un logit de cantidad `a_i`:

- `conditional_grams_i = 300 × softplus(a_i)`: cantidad si está presente.
- `p_i = sigmoid(z_i − log(pos_weight_i))`: gate con corrección del sesgo de
  BCE ponderada. En BCE aislada esta corrección corresponde al posterior sin
  pesos; con entrenamiento multitarea NO garantiza probabilidades calibradas.
- `ingredient_masses_i = p_i × conditional_grams_i + 1e-8`.
- `total_mass = sum(ingredient_masses)`.
- `shares = ingredient_masses / total_mass`.
- Nutrientes: gramos finales × matriz nutricional del nuevo train.

Se conserva la API de shares/masa para evaluación e inferencia. Los
checkpoints tienen un `model_type` diferente, por lo que V2 y V3 no se mezclan.
La calibración posterior elige solamente el umbral binario de presencia sobre
validación: NO cambia los gramos ni certifica probabilidades bien calibradas.

## Supervisión de cantidades

La cantidad condicional tiene una pérdida positiva independiente del gate:
un ingrediente real recibe gradiente de cantidad incluso si su presencia
predicha es casi cero. Se usa Smooth L1 sobre gramos/300, beta=0,1 (transición
de 30 g), ponderando cada positivo con un balanceo moderado derivado de train.

Los pesos de cantidad se calculan como raíz inversa de frecuencia, normalizados
y limitados a [0,25; 3]. Clases sin positivos de train no reciben supervisión
condicional ficticia. Los pesos positivos de BCE siguen limitados a [1; 20].

Además se conserva el L1 end-to-end sobre TODAS las clases y se añade una
penalización de gramos asignados a ausentes. Pesos iniciales:

- Presencia: 1.
- Cantidad condicional: 1.
- Gramos falsos: 0,5.
- L1 ingredientes: 1 (escala 300 g).
- Masa total: 1 (media fija del nuevo train; provisional actual 216,89 g).
- Nutrición: 0,25; escalas [500 kcal, 30 g grasa, 80 g carbohidratos, 40 g proteína].

Son hipótesis de partida, NO pesos optimizados ni evidencia de superioridad.
No se conserva la pérdida de proporciones de V2: la composición deriva de
gramos supervisados directamente. La inicialización de cantidades usa medias
condicionales de TRAIN solamente, y los gates usan prevalencias de train.

## Geometría RGB y profundidad

Depth Anything V2 Small/vits permanece congelado y recibe el RGB con la misma
geometría que ConvNeXt. Sigue siendo profundidad RELATIVA estimada de RGB,
no centímetros, volumen ni profundidad del sensor.

El letterbox preserva relación de aspecto y contenido completo. Los márgenes
usan color próximo a la media ImageNet; los percentiles de depth se calculan
solo en la región válida y el padding de depth se pone a cero.

La augmentación comparte flip y rotación entre RGB y depth. Ambos se rotan
como tensores con la MISMA grilla, expandiendo antes de redimensionar para no
cortar alimentos. Color jitter solo afecta RGB. El cache es determinista,
anterior a augmentations, float16 y 320×320, separado de V2.

La fusión tiene dos puertas escalares entrenables, inicializadas en 0,1, más
normalización por rama y por token. Esto preserva un camino RGB directo y
permite reducir el uso de depth; no prueba que la rama ayude en este dataset.

## Optimización y reanudación

Batch físico 2, acumulación 16 (efectivo 32). AdamW, LR inicial nominal heads
3e-4 y encoder 1e-5, warmup de LR 3 épocas, descenso coseno hasta 1% del LR.
El scheduler tiene horizonte fijo de 100 aunque se haga un chequeo de 5 épocas.
`last.pt` conserva modelo, optimizador, scaler AMP, scheduler y todos los RNG.

Best checkpoint por menor L1 de ingredientes de validación; paciencia 8.
`history.csv` V3 añade macro-F1, gramos falsos, learning rates y componentes de
pérdida train/validación. Los NPZ de evaluación añaden `conditional_grams`.
Los fingerprints de records/vocabulario y los hashes de RGB/depth previenen
reanudación o evaluación con datos diferentes.

## Comparaciones

La V2 histórica usa otro split y seis clases distintas del top 100 recalculado.
Su comparación con V3 es descriptiva, no una atribución causal de mejoras.
`config/query2label_100_v2_bce_grouped.json` permite un baseline NUEVO V2+BCE
con los mismos records y perfiles que V3. Se guarda en v3/baselines/, no en V2.

La independencia de TODOS los escaneos no está certificada: los grupos del
split provisional son heurísticos, pueden unir platos diferentes u omitir
relaciones. No se pide revisión manual para usar esta modalidad aceptada.
