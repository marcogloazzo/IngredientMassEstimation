# V4 — especialización de tareas y masa total auxiliar

Estado: implementada y verificada con pruebas sintéticas; **no entrenada**.
Motivación empírica y límites: [diagnóstico V3](DIAGNOSTICO_GATE_V3.md).

## Qué cambia y qué permanece

Se conserva RGB completo 320×320, ConvNeXt Small etapas 3/4, Depth Anything V2
Small congelado, CNN de profundidad, dos compuertas residuales escalares, 500
tokens, 100 consultas y dos capas de Cross-Attention sin Self-Attention.
Se sigue congelando stem/etapa 1 y ajustando etapas 2–4 tras tres épocas de
warmup. No se modifica el split agrupado provisional ni el vocabulario.

La salida de cada consulta se proyecta de 256 a 768 dimensiones. En la variante
especializada, el MLP compartido se sustituye por dos MLP independientes:

```text
RGB → ConvNeXt etapas 3/4 ──┐
                          ├→ memoria multiescala → Query2Label → proyección 768
RGB → depth relativo → CNN┘                                      ├→ MLP presencia 768→384→192 → logits → p
                                                                └→ MLP cantidad 768→384→192 → softplus ×300 → c
                                   p × c → gramos → suma → masa total desplegada
                                           └→ matriz nutricional → nutrientes

Memoria multiescala → media espacial → LayerNorm → MLP 256→128→1
                                         └→ softplus × media train → masa auxiliar
```

Los dos MLP tienen LayerNorm de entrada, GELU y dropout como V3. Se comparten
encoder, fusión, atención y proyección; no se afirma eliminar toda interacción
entre gradientes. La rama de cantidad mantiene supervisión directa en positivos.

El cabezal auxiliar recibe la media de la memoria visual fusionada, no los
gramos/presencias estimados ni etiquetas. Se inicializa para predecir la media
de masa total de **train solamente**; esa escala queda en el checkpoint.
La media se recalcula con los registros reales (216,89 g en este split V3),
no se fija a 219,11 g de la partición histórica.

## Regla de inferencia y pérdidas

Se mantiene:

```text
p_i = sigmoid(logit_i - log(pos_weight_i))
c_i = softplus(amount_logit_i) × 300 g
m_i = p_i × c_i
M = sum_i m_i
nutrientes = m @ perfiles_train
```

La corrección de logits es una corrección de prior para BCE ponderada, no una
calibración garantizada. Los 300 g no limitan el máximo. El umbral de F1 no corta
gramos y no se incorporó el umbral exploratorio 0,75 al modelo.

Se conservan las seis pérdidas de V3 y sus pesos. Con el cabezal auxiliar activo:

```text
L_aux = mean(|M_aux - M_real|) / media_train
L_consistency = mean(|M_aux - sum_i m_i|) / media_train
L_V4 = L_V3 + 0.25 × L_aux + 0.10 × L_consistency
```

Los pesos nuevos son puntos de partida experimentales, no valores óptimos
demostrados. Consistencia transmite gradientes a ambas ramas, sin detach.
La consistencia no identifica qué ingrediente necesita corregirse; vigilar L1,
MAE presente, gramos falsos y nutrientes, no solamente MAE de masa total.

**M_aux no reemplaza M ni reescala m en inferencia.** Se registra por separado
para comprobar su utilidad. Los nutrientes y la masa desplegada siempre usan
los mismos gramos y mantienen conservación por suma.

## Experimentos aislados

| Configuración | MLP separados | Masa auxiliar / consistencia |
| --- | --- | --- |
| `query2label_100_v4_split_only.json` | Sí | No, pesos 0 |
| `query2label_100_v4_aux_only.json` | No; compartido como V3 | Sí |
| `query2label_100_v4.json` | Sí | Sí |

Entrenar primero `split_only` y `aux_only` con datos, seed y presupuesto
comparables; después comparar la combinación. No utilizar test para decidir
arquitectura/pesos/umbrales. Un solo seed no permite asegurar significancia.

## Compatibilidad y preservación

- Tipo distinto: `convnext_small_query2label_specialized_mass_v4`.
- V4 inicia una corrida nueva; `--resume` rechaza V3 y otras configuraciones.
- Código de train/eval/inferencia admite V1/V2/V3 sin cambiar sus arquitecturas.
- Caché V3 320×320 compatible: mismo depth config, mapas y fingerprints.
- Resultados V4 separados; se bloquea escribir train/eval/inferencia en namespaces
  `v2/` o `v3/`. El cache puede reutilizarse allí en lectura.
- Se guardan `conditional_grams`, masas explícitas y, cuando corresponde,
  `auxiliary_total_mass` en NPZ. V4 guarda además predicciones y detalles de la
  mejor validación durante training, sin volver a ejecutar el modelo.
- `history.csv` registra pérdidas nuevas y, con auxiliar activo, MAE auxiliar y
  discrepancia con la suma de ingredientes.

## Pruebas

`tests/test_v4.py`: formas/500 tokens, conservación de masa, MLP independientes,
ablaciones, escalas derivadas solo de train, gradientes/loss weights, checkpoints,
cache compatible y diagnósticos NPZ. Son tensores sintéticos; no entrenamientos
con Nutrition5k ni descargas de pesos.
