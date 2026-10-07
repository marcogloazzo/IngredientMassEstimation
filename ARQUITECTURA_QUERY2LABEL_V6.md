# V6 — ASL, cantidades por clase, depth espacial y consultas CLIP

Estado: implementación nueva, no entrenada. No se afirma que supere versiones
anteriores. Implementa juntos los cambios solicitados; una comparación del paquete
no permite atribuir mejoras a un único módulo sin ablaciones posteriores.

## Cambios

| Componente | V6 |
| --- | --- |
| Encoder RGB | ConvNeXt Small ImageNet; solo transición E3→E4, etapa 4 y norm final ajustables tras warmup |
| Presencia | ASL, gamma_neg=4, gamma_pos=1, clipping=0,05; gate sigmoid sin log(pos_weight) |
| Cantidad | c_i = s_i × softplus(a_i), escalas robustas exclusivamente de train |
| Fusión depth | Dos compuertas espaciales aprendidas B×1×20×20 y B×1×10×10 |
| Consultas | 99 nombres explícitos inicializados con texto CLIP; other aleatorio entrenable |
| Masa desplegada | p_i×c_i recortado por probabilidad; umbral de masa distinto del de presencia |

Se mantiene RGB completo 320×320, split agrupado provisional, 100 clases,
CNN depth entrenable, dos capas Cross-Attention sin Self-Attention,
MLP **compartido** 768→384→192 y cabezal auxiliar de masa total de V5.
No se usa LoRA: se eligió ajustar solo la etapa 4, una de las alternativas solicitadas.

## Encoder y profundidad

Stem y etapas 1–3 (`features[:6]`) quedan con requires_grad=False y en eval.
Solo `features[6:]` y el norm final participan del grupo de parámetros del encoder.
Durante las primeras 3 épocas también quedan congelados; luego ajustan con LR 1e-5.
Las proyecciones multiescala/atención/cabezales siguen entrenables con LR 3e-4.

Depth Anything V2 Small sigue congelado, externo al módulo entrenable. Los mismos
mapas estimados desde RGB de V3 se reutilizan: no hay depth de sensor como input
ni regeneración de cache. Sigue siendo profundidad relativa, no volumen métrico.

Por escala s:

```text
R_s = GroupNorm(Conv1x1(RGB_s))
D_s = GroupNorm(Conv1x1(depth_s))
G_s = sigmoid(Conv1x1(GELU(Conv1x1(concat(R_s,D_s)))))
F_s = R_s + G_s * D_s
```

G_s tiene un canal, varía espacialmente y usa 32 canales intermedios. Su última
convolución se inicializa en cero con bias logit(0,1), conservando al inicio un
residual depth de 0,1. No es una máscara de segmentación ni prueba de presencia.
Flatten+LayerNorm produce 400+100 tokens, más posición seno/coseno y embedding
de escala para atención. Inferencia puede guardar las dos compuertas como PNG.

## CLIP: solo inicialización textual

La preparación usa OpenAI CLIP ViT-B/32, revisión fijada
`d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, con texto
`a photo of {name}.` y sin imágenes de entrenamiento ni labels de validación/test.
El vocabulario original se construyó de train; se verifica su fingerprint.

El archivo contiene vectores unitarios de 512 dimensiones. Los explícitos
inicializan los embeddings de consultas; `other` conserva un vector aleatorio.
Las consultas y la proyección 512→256 son entrenables. La proyección inicia
ortogonal con generador separado seed=42 y se normaliza la consulta proyectada
a norma sqrt(256). Los espacios CLIP y ConvNeXt no quedan alineados mágicamente:
esa adaptación se aprende con las tareas.

El encoder completo CLIP queda congelado durante preparación y no se incorpora
a ConvNeXt/Q2L. Los pesos de consultas/proyección quedan en el checkpoint: no
hace falta instalar, descargar ni ejecutar CLIP para inferir desde un checkpoint.

## ASL coherente

Presencia usa `p_i=sigmoid(z_i)` y el bias inicial `logit(prevalencia_train_i)`.
Se elimina tanto `−log(pos_weight)` del gate como el offset `+log(pos_weight)`
de inicialización. No se pondera ASL con pos_weight de BCE.
ASL trata de forma asimétrica positivos/negativos fáciles, pero no elimina
automáticamente masa fantasma; se conserva la pérdida explícita sobre ausentes.

## Escalas condicionales

Para cada clase, s_i es el percentil 90 de **cantidades positivas de train** si
hay al menos 10 registros positivos. Con menos soporte se usa el mismo percentil
de cantidades positivas de todas las clases de train. Se limita la escala a
[5,1000] g para estabilidad; **eso no limita la cantidad predicha**.

Se guardan soportes, escalas y provenance en `class_mass_scales.json`; las escalas
también son buffers persistentes del modelo. No se necesitan registros de train
para inferencia. Los 10 registros son escaneos, no necesariamente platos
físicamente independientes: el agrupamiento provisional sigue siendo provisional.

La inicialización de c conserva la media condicional train de cada ingrediente.
La regresión positiva usa SmoothL1(c_i/s_i, gt_i/s_i), beta=0,1, con el balance
por clase existente. El L1 final sigue en gramos; 300 g permanece **solo como
normalizador** de pérdidas globales de ingredientes/falsos, no como escala del
cabezal condicional. Masa total usa media train; nutrientes conservan sus escalas.

## Filtro de masa y pérdidas

Entrenamiento y cálculo de pérdidas usan siempre `m_dense=p*c`, sin corte duro:
esto mantiene gradientes y supervisión directa de cantidad aunque p sea baja.
Evaluación/inferencia usan:

```text
m_i = p_i*c_i si p_i >= tau_mass, y 0 en el resto
M = sum_i(m_i)
N = m @ matriz_nutricional_train
```

No hay epsilon agregado a masas ausentes. Si no sobrevive ninguna clase, M=0,
shares=0 y nutrientes=0, todo finito. La composición oracle con masa real se
marca no disponible o excluye esas muestras, no presenta un falso reparto válido.

Se conservan los pesos: presencia=1, condicional=1, falsos=0,5, L1=1, masa=1,
nutrientes=0,25, auxiliar=0,25, consistencia=0,10. Auxiliar/consistencia supervisan
masa real y suma **dense**, respectivamente; la masa auxiliar no reescala gramos.
Por esto las pérdidas validation_mass/ingredient_mass no se deben reconciliar
con métricas filtradas: el CSV guarda también las métricas SIN filtro.

## Calibración solo en validación

Tau_mass inicia 0,05. Tau_presence inicia 0,5 y se calibra por F1 global.
El corte de masa se elige por menor L1 entre candidatos que no empeoren MAE total
ni calorías más de 5% relativo respecto a no filtrar, tolerancias configurables.
El candidato 0 siempre está disponible: si recortar perjudica, puede elegirse
no recortar. No se promete eliminar todos los ingredientes falsos confidentes.

Esta búsqueda es exploratoria sobre la validación usada también para seleccionar
checkpoint, no evidencia independiente en test. El checkpoint calibrado conserva
ambos umbrales. A diferencia de V1–V5, calibrar masa en V6 **sí cambia los gramos**.

## Reproducibilidad y preservación

Tipo distinto: `convnext_small_query2label_spatial_clip_asl_v6`.
Nueva corrida desde ImageNet, no resume desde V5. Resume exige mismo modelo,
config salvo total epochs, datos, manifest depth y hash de caché CLIP.
Train/eval/inferencia bloquean escribir dentro de v2/v3/v4/v5.
Historial, mejor/último checkpoint, NPZ y métricas quedan en v6/.
Diagnóstico distingue efectos de sigmoid p y recorte, y compara con/sin filtro.
