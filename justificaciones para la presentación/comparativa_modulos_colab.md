# Comparativa de mejoras por módulo: entrenamiento con GPU de Colab

Fecha de revisión: 1 de octubre de 2026. Objetivo: mejorar el sistema RGB → ingredientes → gramos → nutrientes con un presupuesto inicial de 100 unidades de Colab. Este documento corresponde a la fase de comparación; las propuestas todavía no fueron implementadas ni entrenadas.

## 1. Decisión recomendada

Priorizaría **ampliar de 50 a 100 clases, supervisar directamente los gramos y ajustar el encoder con GPU**, conservando la conversión nutricional explícita. Compararía EfficientNet-B0 con ConvNeXt-Tiny. Probaría un cabezal de dos capas como ablación económica, sin asumir que más parámetros dan mejores resultados.

La alternativa más económica es conservar B0 y ajustar sus últimos bloques. La alternativa con mayor capacidad es ConvNeXt-Tiny con ajuste de su última etapa y, si validación lo justifica y queda presupuesto, ajuste completo. Ninguna requiere un dataset nuevo.

La elección entre las dos se debe hacer por validación, usando error en gramos y nutrientes además de F1. Los números de ImageNet o COCO sirven como antecedentes, no como predicciones de mejora en Nutrition5k.

## 2. Qué hace el sistema actual

Fuentes locales revisadas: [features.py](../src/ingredient_mass/features.py), [model.py](../src/ingredient_mass/model.py), [data.py](../src/ingredient_mass/data.py), [metrics.py](../src/ingredient_mass/metrics.py), [configuración regularizada](../config/regularized.json) y [experimentos previos](../EXPERIMENTS.md).

```text
RGB → resize / center crop / normalización
    → EfficientNet-B0 congelada → pooling global → vector de 1280 valores
    → LayerNorm → Linear(1280, 192) → GELU → Dropout
        ├─ Linear(192, K) → sigmoid → presencia de ingredientes
        ├─ Linear(192, K) → softplus → cantidades positivas
        │       proporciones = normalizar(presencia × cantidades)
        └─ Linear(192, 1) → softplus → masa total
    → gramos por ingrediente = proporción × masa total
    → nutrientes = gramos × tabla de nutrientes por gramo
```

K = 50: 49 etiquetas específicas y `other`. El baseline anterior utiliza 384 unidades ocultas; la variante regularizada utiliza 192.

| Módulo actual regularizado | Función | Parámetros entrenables | MACs por imagen, partes lineales |
|---|---|---:|---:|
| EfficientNet-B0 sin clasificador | Obtener características visuales | 0, está congelada; 4.007.548 parámetros totales | Aproximadamente 0,39 G en la referencia publicada, incluido su clasificador original |
| Pooling global promedio | Resumir el mapa espacial | 0 | Reducciones, costo pequeño respecto del encoder |
| LayerNorm + proyección 1280 → 192 | Compartir información entre tareas | 248.512 | 245.760 |
| Presencia 192 → 50 | Reconocimiento multietiqueta | 9.650 | 9.600 |
| Cantidad 192 → 50 | Composición relativa | 9.650 | 9.600 |
| Masa total 192 → 1 | Tamaño de la porción en gramos | 193 | 192 |
| Multiplicaciones y tabla nutricional | Convertir composición a gramos y nutrientes | 0 | 200 MACs para la tabla, más operaciones elementales |

El cabezal completo tiene **268.005 parámetros** y 265.152 MACs lineales por imagen. B0 sin el clasificador original más ese cabezal suma **4.275.553 parámetros**, aunque actualmente solamente se entrenan los del cabezal. Los 5,3 M citados en la comparación anterior pertenecen al clasificador original de ImageNet.

Datos preparados disponibles: 2.332 imágenes de entrenamiento, 423 de validación y 507 de test. Total: 3.262 registros. Hay 197 ingredientes distintos en entrenamiento antes de agruparlos. Estos conteos provienen de los archivos locales, no de una descarga nueva.

Resultados de test ya obtenidos:

| Variante | Parámetros del cabezal | Micro-F1 | MAE ingrediente presente | MAE masa total | MAE calorías |
|---|---:|---:|---:|---:|---:|
| Baseline 384 | 533.349 | 0,641 | 18,13 g | 51,58 g | 99,94 kcal |
| Regularizada 192, seleccionada por ingrediente | 268.005 | 0,644 | 18,74 g | 47,62 g | 99,89 kcal |
| Regularizada 192, seleccionada por masa total | 268.005 | 0,641 | 18,72 g | 47,53 g | 99,23 kcal |

## 3. Cómo interpretar costo y métricas

**Parámetros** describen tamaño y capacidad; los **parámetros entrenables** determinan además cuánto estado del optimizador se necesita. **Operaciones** describen trabajo aritmético; no equivalen directamente a segundos, memoria de activaciones ni unidades de Colab.

Los encoders se muestran con los GFLOPs nominales publicados por sus fuentes. Para los diseños pequeños calculados aquí se reportan tanto GMACs como GFLOPs con la convención explícita **1 MAC = 2 FLOPs**. No sumar cifras de distintas convenciones sin convertirlas. Los valores de cabezales omiten normalización, activaciones y operaciones elementales; son estimaciones analíticas, no perfiles de ejecución.

No se entrenó ni ejecutó inferencia de ningún candidato durante esta comparación. Por eso sus futuros F1, MAE, segundos por época y unidades consumidas están pendientes. Afirmar una mejora numérica en nuestra tarea antes de medirla sería inventarla.

## 4. Entrada, aumentos y representación espacial

La transformación actual es la de los pesos de torchvision: resize, recorte central y normalización de ImageNet. Las características guardadas se extraen de una única transformación fija.

| Opción | Qué cambia | Costo adicional / parámetros | Métricas decisivas | Recomendación |
|---|---|---|---|---|
| Actual: recorte central | Mantiene la transformación del preentrenamiento | Sin red adicional | MAE y F1 de referencia | Conservar como control |
| Resize completo / padding que conserve el plato | Evita quitar comida situada en los bordes | 0 parámetros; procesamiento O(HW), generalmente fuera del conteo de GFLOPs del modelo | MAE masa e ingredientes; ejemplos con bordes | Comparar con el recorte actual |
| Flips, rotaciones moderadas y variación suave de color | Aumenta diversidad sin cambiar las etiquetas de masa | 0 parámetros; mismo forward por imagen | Diferencia train/val, Macro-F1 y MAE | Recomendado al ajustar el encoder |
| Recortes fuertes, CutMix o borrado de ingredientes | Cambia cuánto alimento queda visible | La red puede costar lo mismo, pero las etiquetas de gramos dejan de corresponder a lo observado | Errores de masa y falsos negativos | No utilizar por defecto |

El recorte central es un posible problema, no una causa demostrada del error. Las variantes se deben comparar manteniendo resolución y entrenamiento controlados. Los aumentos de imagen requieren imágenes o nuevos pases del encoder; un vector ya guardado no reproduce el efecto de rotar la imagen.

Para el módulo que resume las características:

| Opción de pooling | Ventaja | Costo frente a GAP | Limitación |
|---|---|---|---|
| GAP actual | Barato, representación compacta | Base; 0 parámetros | Pierde localización |
| GAP + máximo global | Añade evidencia de regiones pequeñas | 0 parámetros en pooling; duplica la entrada del MLP | Necesita volver a obtener mapas/características |
| GeM | Interpola entre promedio y máximo | Uno o pocos exponentes entrenables; O(CHW), sin convoluciones extra | Requiere tratar correctamente valores negativos del mapa |
| Atención por ingrediente | Cada etiqueta busca regiones útiles | Se cuantifica en la sección de cabezales espaciales | Más implementación y memoria |

GeM procede de [Fine-tuning CNN Image Retrieval with No Human Annotation](https://arxiv.org/abs/1711.02512), con [código oficial](https://github.com/filipradenovic/cnnimageretrieval-pytorch). Su evidencia original es recuperación de imágenes; no prueba por sí sola una mejora de gramos. Como antecedente de máximo global en multietiqueta, el [repositorio ML-GCN](https://github.com/megvii-research/ML-GCN) reporta mAP en COCO de 77,3 con GAP y 81,9 con GMP para su ResNet-101. Es otro benchmark y otro encoder.

## 5. Encoder y estrategia de ajuste

La comparación más amplia permanece en [comparacion_encoders.md](comparacion_encoders.md). Para 100 unidades seleccionaría estos candidatos:

| Opción | Parámetros originales aproximados | Costo nominal por imagen | Referencia de precisión | Uso recomendado |
|---|---:|---:|---|---|
| [B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html) | 5,29 M | 0,39 GFLOPs a 224 | Top-1 ImageNet 77,69 % | Ajustar últimos bloques; primera prueba GPU |
| [B2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html) | 9,11 M | 1,09 GFLOPs a 288 | Top-1 ImageNet 80,61 % | Alternativa intermedia |
| [ConvNeXt-Tiny](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) | 28,59 M | 4,46 GFLOPs a 224 | Top-1 ImageNet 82,52 % | Candidato con mayor capacidad para la comparación principal |
| ConvNeXt-Small 22k→1k que ya tenés | Aproximadamente 50 M | 25,5 GFLOPs a 384; aproximadamente 8,7 a 224 | Top-1 85,8 % a 384; no atribuir este valor a 224 | Evita descargar otros pesos, pero cuesta más que Tiny |
| DINOv2 ViT-S/14 congelado | 21 M | Aproximadamente 6,1 GMACs a 224, estimación analítica del ViT | 81,1 % con evaluación lineal ImageNet del repositorio; otro protocolo | Experimento adicional con representación auto-supervisada |

ConvNeXt-Small y sus pesos: [paper](https://arxiv.org/abs/2201.03545) y [tabla oficial de modelos](https://github.com/facebookresearch/ConvNeXt). DINOv2: [paper](https://arxiv.org/abs/2304.07193) y [repositorio](https://github.com/facebookresearch/dinov2). La estimación de DINO usa 257 tokens, dimensión 384, 12 bloques y FFN de 1536; omite operaciones pequeñas. Sus GMACs no se deben comparar con una cifra que cuente cada MAC como dos FLOPs sin convertirla.

### Reutilización de OmniFood8K: corrección importante

En tu código local `OmniFood8K/Code/infer_single.py`, la rama **RGB es Swin** y la rama **profundidad es ConvNeXt-Small**. La primera carga `checkpoint["net"]` y la segunda `checkpoint["net2"]`. Por eso extraer `net2` y suministrarle RGB implica cambiar de dominio respecto de su ajuste nutricional.

El archivo separado `convnext_small_22k_1k_384.pth`, de aproximadamente 201 MB, sí es el punto de partida preentrenado de ImageNet. Lo podés usar para un nuevo encoder RGB. Los checkpoints completos `ckpt_best.pth` y `ckpt_best-nut.pth` ocupan aproximadamente 3,24 GB cada uno; no hacen falta para la ruta RGB liviana. Esta conclusión se basa en el cargador local y el script `train_nutrition.py`, sin cargar los checkpoints grandes.

| Entrenamiento | Trabajo | ¿Sirven los vectores cacheados? | Costo |
|---|---|---|---|
| Encoder congelado | Entrenar solo cabezales | Sí, si coinciden encoder y transformación | Muy bajo por época |
| Última etapa ajustable | Entrenar bloque final y cabezales | No los vectores finales; podrían cachearse mapas de un prefijo congelado | Intermedio; buen uso inicial de GPU |
| Encoder completo ajustable | Adaptar todas las características | No | Más memoria y operaciones de backward |

Congelar capas reduce el costo de entrenamiento, pero **no reduce los GFLOPs de inferencia**. Como aproximación, entrenamiento completo cuesta del orden de tres forwards por muestra contando backward; ajuste parcial cuesta `F_prefijo_congelado + ~3 × F_parte_entrenable`. No es una medición de tiempo.

Usaría primero warm-up de cabezales, luego ajuste con learning rate menor para el encoder, early stopping y precisión mixta. PyTorch documenta AMP en [esta receta oficial](https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html). AMP puede mejorar velocidad/memoria según hardware y batch; no cambia las operaciones teóricas del modelo ni garantiza un multiplicador fijo.

## 6. Cabezal compartido: ¿más capas y pesos ayudan?

La evidencia local favorece probar cambios moderados: reducir el cabezal a la mitad mantuvo F1 y mejoró masa total. No demuestra que toda ampliación vaya a fallar, pero tampoco respalda saltar directamente a millones de parámetros con 2.332 imágenes.

Los siguientes diseños son MLP propuestos, no resultados tomados de un paper. Todos usan LayerNorm inicial y 101 salidas totales para 50 clases: presencia, composición y masa. Se agregan GELU y Dropout entre capas.

| Diseño, dimensiones | Parámetros de todos los cabezales | GMACs lineales / imagen | GFLOPs con 2 FLOPs/MAC | Cambio de costo del cabezal |
|---|---:|---:|---:|---:|
| 1280 → 192 → salidas, regularizado actual | 268.005 | 0,000265 | 0,000530 | 1× |
| 1280 → 384 → salidas, baseline | 533.349 | 0,000530 | 0,001061 | 2× |
| 1280 → 384 → 192 → salidas | 587.877 | 0,000585 | 0,001169 | 2,20× |
| 1280 → 512 → 256 → salidas | 815.717 | 0,000812 | 0,001625 | 3,06× |
| 1280 → 1024 → 512 → salidas | 1.890.917 | 0,001887 | 0,003773 | 7,12× |
| Compartido 192 + capa privada 128 en cada tarea | 335.653 | 0,000332 | 0,000665 | 1,25× |

Aun el MLP grande cuesta poco frente al encoder, pero puede aprender correlaciones espurias y sobreajustarse. **Agregar capas al vector global no recupera los detalles espaciales que GAP ya eliminó.**

Recomendación: probar una capa privada pequeña por tarea o el MLP 384→192 como ablación sobre características guardadas; elegir por validación. Son pruebas baratas, que incluso pueden hacerse localmente para reservar GPU al encoder. Comparar con igual número de clases, pérdida y presupuesto de búsqueda. El paper [Nutrition5k](https://arxiv.org/html/2103.03375v1) también utiliza capas compartidas y privadas; sus capas de 4096 unidades y preentrenamiento JFT no constituyen evidencia de que ese tamaño sea adecuado aquí.

## 7. Reconocimiento multietiqueta de ingredientes

La cabeza actual genera K logits y los convierte a probabilidades con sigmoid. Se entrena con BCE ponderada por clase.

| Opción | Qué aporta | Parámetros y costo extra | Métricas |
|---|---|---|---|
| BCE ponderada actual | Compensa frecuencia de positivos | Base | Micro/Macro-F1, precision, recall |
| Sigmoid + Asymmetric Loss (ASL) | Reduce influencia de negativos fáciles y trata positivos/negativos de forma distinta | 0 parámetros nuevos; 0 GFLOPs extra en inferencia; entrenamiento O(BK) | Añadir mAP y F1 por frecuencia |
| Umbrales por clase regularizados | Adapta decisiones al desbalance | K constantes; O(K) comparaciones, sin red | Precision/recall por clase y calibración |
| Corrección suave por coocurrencia | Aprovecha que ciertos ingredientes aparecen juntos | Matriz fija sin parámetros entrenables; K² MACs por muestra, 0,0000025 GMACs para K=50 | Macro-F1, falsos positivos fuera del dominio |
| ML-Decoder / Query2Label reducido | Consulta regiones del mapa por etiqueta | Ejemplo cuantificado abajo; requiere mapas espaciales | mAP y F1 de ingredientes pequeños |

ASL es un candidato de bajo riesgo de cómputo: [paper ICCV 2021](https://arxiv.org/abs/2009.14119), [código oficial](https://github.com/Alibaba-MIIL/ASL). Su evidencia corresponde a multietiqueta general; la ventaja en nuestras clases todavía debe medirse.

Para coocurrencia, [ML-GCN, CVPR 2019](https://arxiv.org/abs/1904.03582) y su [repositorio](https://github.com/megvii-research/ML-GCN) modelan dependencias con grafos. La matriz fija anterior sería una adaptación sencilla, no una reproducción del GCN completo. Su mayor riesgo es inferir ingredientes invisibles por costumbre y fallar al cambiar la cocina.

### Cabezal de atención espacial

[Query2Label](https://arxiv.org/abs/2107.10834), con [repositorio oficial](https://github.com/SlongLiu/query2labels), usa consultas por etiqueta sobre el mapa visual. [ML-Decoder](https://arxiv.org/abs/2111.12933), con [repositorio oficial](https://github.com/Alibaba-MIIL/ML_Decoder), reduce el costo del decoder y permite agrupación de etiquetas.

Sus sistemas completos reportan respectivamente **91,3 % y 91,4 % mAP en COCO**. Estos resultados incluyen encoders y configuraciones de entrenamiento grandes; no son resultados nutricionales ni ganancias aisladas del cabezal.

Una adaptación chica para nuestro caso, con mapa 7×7×1280, dimensión 128, 50 consultas, FFN 512 y un bloque, tendría aproximadamente **0,37–0,44 M parámetros** y **0,019–0,023 GMACs**, equivalentes a **0,038–0,046 GFLOPs** bajo la convención de dos FLOPs por MAC. El rango refleja incluir o no self-attention. Se usarían salidas por consulta para presencia y composición, más una salida global para masa. Son estimaciones de un diseño propuesto, no el costo del ML-Decoder oficial ni una implementación existente.

Al ampliar K o resolución hay que recalcular: la atención y la FFN crecen con el número de consultas/tokens. Recomendaría esta opción después de obtener una referencia ajustada con MLP. Da mapas de atención útiles para explicar el modelo, pero estos no equivalen automáticamente a máscaras de ingredientes.

La calibración de umbrales actual solo modifica reconocimiento y presentación. En `metrics.py`, los gramos usan todas las proporciones sin umbral; subir el umbral no cambia el MAE de masas. Cualquier modificación que aplique máscara y renormalice cantidades debe evaluarse separadamente.

## 8. Composición relativa y gramos por ingrediente

Actualmente `share = normalizar(sigmoid(presencia) × softplus(cantidad))` y `gramos = share × masa_total`. La pérdida de composición es L1 ponderada; **no hay una pérdida explícita sobre los gramos de cada ingrediente**. La pérdida nutricional no identifica de forma única qué ingrediente aportó los nutrientes.

| Opción | Qué cambia | Costo extra | Ventaja / riesgo |
|---|---|---|---|
| Factorización actual | Conserva suma coherente con masa total | Base | Interpretable, pero el error de masa afecta todas las cantidades |
| Misma factorización + pérdida directa en gramos | Supervisa el objetivo final, incluyendo cantidades asignadas a ausentes | 0 parámetros; 0 GFLOPs adicionales en inferencia; pérdida O(BK) | Primera mejora recomendada |
| Distribución softmax o normalizada + pérdida de distribución | Aprende proporciones con CE/KL u otra pérdida validada | Misma cantidad de logits; operaciones O(K) | Hay que controlar ingredientes ausentes y dominantes |
| Regresión directa de K masas positivas | Cada salida predice gramos; masa total = suma | Puede conservar exactamente el número de pesos de las dos cabezas K existentes si se reemplaza cantidad y elimina la cabeza global | Evita propagación del error global; puede acumular muchos gramos falsos |
| Wasserstein/Sinkhorn | Penaliza errores según similitud funcional entre ingredientes | Matriz K×K; costo de pérdida O(BTK²), T iteraciones; 0 costo extra en inferencia | Más diseño y costo de entrenamiento; puede tolerar confusiones nutricionalmente relevantes |

PITA proporciona evidencia para composición relativa y Wasserstein: [paper completo](https://arxiv.org/html/2010.08727v1). Con igual reconocimiento, su variante CE y Wasserstein reporta EMD 154,35 y 147,29, manteniendo IoU 0,42. Es un benchmark de recetas y cantidades relativas, no gramos absolutos en Nutrition5k. No reproduzco todo PITA: implicaría representar recetas y construir distancias/subgrupos. El paper enlaza demo/datos; no se verificó un repositorio oficial listo para usar en este entorno.

Usaría primero SmoothL1/Huber o L1 escalada en gramos, con términos separados para ingredientes presentes y masa falsa en ausentes. La regresión directa sería una ablación contra la factorización. No haría MAPE por ingrediente sin tratar los ceros: es indefinida cuando el ingrediente está ausente y puede sobrerrepresentar dosis muy pequeñas.

Métricas: MAE de ingredientes presentes, MAE por clase, masa total predicha en ingredientes ausentes, error absoluto acumulado por plato y MAE de nutrientes. El MAE actual de shares promedia sobre todas las clases; puede parecer pequeño por los muchos ceros.

## 9. Masa total: RGB, profundidad y escala

La cabeza actual es una regresión escalar con softplus. Su error baseline de 51,58 g baja a aproximadamente 47,5 g con el cabezal regularizado.

| Opción | Parámetros / costo extra frente al sistema RGB | Qué exige | Recomendación |
|---|---|---|---|
| MLP privado pequeño para masa | Por ejemplo, 192→128→1 reemplaza 193 pesos por 24.833; añade unos 0,000049 GFLOPs lineales con 2 FLOPs/MAC | Mismas etiquetas | Probar junto con ajuste del encoder |
| Predecir masa normalizada o log-masa | Sin cambio sustancial de pesos; O(1) para transformación | Estadísticas de train y pérdida apropiada | Ablación barata; validar MAE en gramos al desnormalizar |
| Masa = suma de gramos individuales | Elimina la salida global; suma O(K) | Regresión directa de ingredientes | Ablación, no garantía de precisión |
| Profundidad estimada + segunda rama | Un Depth Anything V2 Small añade 24,8 M parámetros, más el encoder/fusión de profundidad | Generar mapas del mismo modo en train y en inferencia | Segunda fase si RGB no alcanza |
| Profundidad real solamente como supervisión auxiliar de entrenamiento | Cabeza auxiliar pequeña durante entrenamiento; se puede quitar en inferencia | Descargar raw depth y alinear máscaras válidas | Alternativa experimental para conservar entrada RGB |
| Masa real o referencia de escala proporcionada por el usuario | Sin cómputo de una red adicional | Una medición o dato extra | Útil como diagnóstico, pero cambia la condición “solo RGB” |

Para profundidad, [Depth Anything V2](https://arxiv.org/abs/2406.09414) y [repositorio](https://github.com/DepthAnything/Depth-Anything-V2). El costo GFLOPs de la rama completa no está homologado en la tabla consultada y depende del tamaño de entrada, encoder DPT y fusión; se debe perfilar. No es correcto atribuirle los 0,39 GFLOPs de B0. Su entrada predeterminada es 518, mucho mayor que 224.

La profundidad relativa no da metros o gramos automáticamente. Incluso una predicción métrica necesita escala/cámara, regiones de alimento y densidad para convertir volumen físico a masa. Entrenar una rama con profundidad real y luego sustituirla por estimada también puede introducir cambio de dominio.

Para supervisión privilegiada, [Unifying Distillation and Privileged Information](https://arxiv.org/abs/1511.03643), con [código de los autores](https://github.com/lopezpaz/distillation_privileged_information), fundamenta usar información adicional durante entrenamiento. Una pérdida auxiliar de profundidad para este proyecto sería una adaptación experimental, no un resultado validado por ese paper en comida.

Como evidencia cercana, el [paper Nutrition5k](https://arxiv.org/html/2103.03375v1) reporta masa MAE 40,4 g con predicción RGB directa y 29,4 g con un escalar de volumen. Usa otro protocolo, vistas y preentrenamiento; no se compara directamente con nuestro test de 507 overheads. [RGB-D + FLAVA](https://www.sciencedirect.com/science/article/pii/S0889157525016370), con [código](https://github.com/yih-f/RGBD-nutrient-estimation-by-FLAVA), reporta 14,43 % PMAE medio y aprovecha ingredientes durante entrenamiento; usa RGB-D y otros splits. No lo recomendaría como primera implementación con 100 unidades.

## 10. Vocabulario y conversión nutricional: cuello de botella medido

La tabla nutricional es determinística. La información por ingrediente conocido es específica, pero `other` mezcla perfiles muy diferentes. Un gramo de aceite no es nutricionalmente equivalente a un gramo de fruta.

Hice un diagnóstico usando los ingredientes y gramos reales del JSONL. Las clases y los perfiles nutricionales se construyen **solo con entrenamiento**, aplicando el orden frecuencia y luego masa del código. Se evalúa el error de conversión en las 423 imágenes de validación, sin ejecutar una red.

| Clases, incluida `other` | Masa agrupada en `other`, train | Masa agrupada, validación | Soporte mínimo de clase seleccionada en train | MAE kcal usando masas reales en validación |
|---|---:|---:|---:|---:|
| 50 actuales | 35,34 % | 35,80 % | 112 platos | 63,94 kcal |
| 100 propuestas | 9,51 % | 8,91 % | 33 platos | 15,21 kcal |
| 150 propuestas | 2,52 % | 2,18 % | 10 platos | 4,75 kcal |

Estos son **errores de representación con cantidades reales**, no métricas de un modelo nuevo ni un límite inferior matemático para toda predicción posible. Un modelo puede compensar el perfil incorrecto asignando masas incorrectas; eso mejora nutrientes a costa de la interpretación. La expansión reduce esa presión, pero añade clases difíciles y poco frecuentes.

En el test ya evaluado, `other` reúne 36,15 % de la masa y el diagnóstico de la tabla de 50 clases da 63,76 kcal de MAE incluso con masas reales. Las decisiones propuestas se apoyan en train/validación, no en buscar checkpoints contra esos datos de test. Los diagnósticos numéricos están en [diagnostico_representacion.json](diagnostico_representacion.json).

| Conversión/vocabulario | Costo | Evaluación |
|---|---|---|
| Mantener 50 y tabla actual | 200 MACs por muestra en la conversión | Conserva el cuello de botella observado |
| 100 clases y tabla específica | 400 MACs en conversión; cabeza B0-192 pasa de 268.005 a 287.305 pesos, +7,2 % en cabezales | Recomendación principal; no requiere dataset nuevo |
| 150 clases | 600 MACs en conversión; más logits, no nuevo encoder | Mejor cobertura, pero clases con solo 10 ejemplos |
| Selección por cobertura de masa y diversidad nutricional | Mismo costo con igual K | Puede priorizar alimentos nutritivamente relevantes sobre condimentos frecuentes; elegir solo con train |
| Nutrición directa o corrección residual aprendida | Un Linear(192,4) añade 772 parámetros y 768 MACs | Puede mejorar kcal, pero deja de ser una conversión puramente explicada por gramos |

Revisaría además la consistencia con el CSV local `ingredients_metadata.csv` y las cantidades de los platos. Para ingredientes conocidos, usar la tabla oficial evita un mapeo manual a USDA. Reemplazarla por una calculadora externa requiere distinguir cocido/crudo y preparación; no es un cambio automáticamente beneficioso. La documentación oficial del dataset describe su [tabla de ingredientes](https://github.com/google-research-datasets/Nutrition5k).

Para comparar 50 y 100 clases justamente, hay que reportar resultados completos de cada vocabulario y resultados sobre una **representación común**: colapsar las 100 predicciones a las 49 etiquetas originales más `other`. Cambiar K altera el denominador del MAE por ingrediente y las clases de Macro-F1; no comparar números sin aclararlo.

## 11. Pérdidas y entrenamiento multitarea

La pérdida actual combina presencia, proporciones, masa y nutrientes con pesos fijos 1, 5, 1 y 0,25, respectivamente. Masa se divide por 300 g; los nutrientes se escalan por constantes.

| Alternativa | Qué aporta | Parámetros / costo de inferencia | Costo de entrenamiento |
|---|---|---|---|
| Pesos fijos, ajustados en pocas pruebas | Control transparente | Sin cambio | Un backward habitual |
| ASL + gramos + pérdida nutricional | Alínea reconocimiento y cantidades con objetivos finales | Sin cambio | Operaciones O(BK) extra, pequeñas frente al encoder |
| Ponderación por incertidumbre | Aprende el balance de las tareas | Un escalar por tarea, removible en inferencia | Poco costo aritmético adicional |
| GradNorm | Equilibra magnitudes/ritmos de gradiente | Sin cambio de inferencia | Necesita gradientes por tarea sobre una parte compartida |
| PCGrad | Reduce conflictos entre tareas | Sin cambio de inferencia | Varios cálculos/proyecciones de gradientes; mayor tiempo y memoria |

Fuentes: [Kendall et al., CVPR 2018](https://arxiv.org/abs/1705.07115), [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html), [PCGrad](https://arxiv.org/abs/2001.06782) y [repositorio de sus autores](https://github.com/tianheyu927/PCGrad).

GradNorm/PCGrad no tienen un GFLOPs extra universal: dependen de qué parámetros se diferencian, número de tareas y reutilización del grafo. No elegiría PCGrad como primera prueba: puede consumir presupuesto en múltiples backwards. Mantendría pesos fijos con objetivos escalados y usaría incertidumbre como ablación barata si las tareas muestran desequilibrio.

Añadir incertidumbre de tareas no produce por sí solo intervalos fiables por imagen. Para intervalos se requieren objetivos probabilísticos o calibración adicional.

## 12. Datasets y acceso desde la nube

### Ruta recomendada: solamente Nutrition5k overhead RGB

Tu copia local contiene **3.490 RGB**, cuyo tamaño sumado es **1.454.144.607 bytes, aproximadamente 1,45 GB**. El protocolo preparado usa 3.262 de ellos. Los metadatos ocupan aproximadamente 2,33 MB y los tres caches comprimidos suman aproximadamente 15,70 MB.

El [repositorio oficial Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) ofrece los archivos por separado en Google Cloud Storage y el archivo completo de 181,4 GB. Basta recuperar selectivamente:

```text
gs://nutrition5k_dataset/nutrition5k_dataset/metadata/
gs://nutrition5k_dataset/nutrition5k_dataset/dish_ids/
gs://nutrition5k_dataset/nutrition5k_dataset/imagery/realsense_overhead/<dish_id>/rgb.png
```

Estos son los orígenes documentados por los autores y usados por tu script de descarga. No se descargaron otra vez ni se comprobó la disponibilidad individual de cada objeto en esta revisión.

No hace falta bajar videos, profundidad ni todo el dataset a tu PC. Los overheads no están disponibles para todos los IDs; el resumen local marca 1.506 faltantes para el split RGB solicitado. **No implica necesariamente una descarga fallida:** el dataset publica overheads “cuando están disponibles”. No cambiar al split depth para rellenarlos sin rehacer y documentar el protocolo.

| Dataset / recurso | Qué supervisa | ¿Gramos reales por ingrediente? | Uso con Colab / decisión |
|---|---|---|---|
| Nutrition5k RGB overhead | Ingredientes, gramos, masa y nutrientes | Sí | Necesario y ya disponible; bucket oficial → disco de Colab |
| Nutrition5k raw depth | Geometría medida | Los gramos vienen de los mismos metadatos | Opcional para supervisión de profundidad; descarga selectiva |
| FoodSeg103 | Máscaras de ingredientes: 7.118 imágenes, 103 clases + fondo | No | Opcional solo si elegimos segmentación; descarga del autor directamente a Colab |
| Food-101 | 101.000 imágenes con clase de plato | No | No necesario para mejorar los gramos; añade otra descarga/preentrenamiento |
| Recipe1M / datos PITA | Recetas, ingredientes y composición de recetas | No equivalen a gramos del plato fotografiado | No recomendado inicialmente: adquisición y preparación grandes |
| VireoFood-172 | Reconocimiento de platos/ingredientes, cocina principalmente china | No | No necesario; los enlaces institucionales consultados no pudieron verificarse |
| OmniFood8K | Estimación nutricional; habría que auditar el detalle de sus etiquetas para esta tarea | No asumirlo sin revisar anotaciones | No necesario para la primera mejora; tus RGB-D existentes no sustituyen automáticamente etiquetas por ingrediente |

FoodSeg103: [paper](https://arxiv.org/abs/2105.05409) y [repositorio oficial](https://github.com/LARC-CMU-SMU/FoodSeg103-Benchmark-v1), con enlace de descarga y modelos. Food-101: [página oficial](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/). Recipe1M requiere acceso/preparación según [Inverse Cooking y su código oficial](https://github.com/facebookresearch/inversecooking). La generación de recetas no es necesaria para nuestras tres salidas.

No encontré un bucket oficial de Google Cloud para todos esos datasets; algunos tienen descargas web o espejos de terceros. Estar alojado en Hugging Face no significa entrenamiento sin transferencia: streaming también descarga bytes y puede repetir lectura en cada época.

### Almacenamiento que evita esperas repetidas

| Forma de acceso | Ventaja | Desventaja | Elección |
|---|---|---|---|
| GCS oficial → `/content` | No usa subida desde tu PC; lectura local durante entrenamiento | Copia inicial, datos temporales al terminar la VM | Recomendado para RGB |
| Archivo único en Drive → copiar y extraer en `/content` | Se guarda una vez; menos accesos a miles de archivos | Preparación inicial y espacio en Drive | Recomendado como cache persistente |
| Leer cada PNG desde Drive en cada época | Evita copiar al inicio | Latencia/I/O puede dejar la GPU esperando | Evitar para entrenamiento repetido |
| Leer cada imagen por URL / streaming GCS | No guarda todo de entrada | Red repetida y latencia por objeto | Útil para inspección; no primera opción para 1,45 GB |
| Cache de vectores o mapas | Reduce mucho cómputo de pruebas con encoder fijo | Deja de valer al ajustar el encoder o cambiar preprocesamiento | Usar para ablaciones de cabezales |

La [FAQ de Colab](https://research.google.com/colaboratory/faq.html) recomienda reducir pequeñas lecturas de Drive y copiar/extraer archivos en la VM. [gcloud storage rsync](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync) permite descarga selectiva y sincronización; su instalación es solo una opción, también se puede usar acceso HTTP a objetos públicos.

Guardaríamos en Drive el paquete de datos, pesos, checkpoints, manifest y resultados; entrenaríamos desde `/content`. Se puede preparar el paquete antes de consumir GPU. Cambiar de tipo de runtime puede recrear la VM: la persistencia debe estar en Drive, no solamente en `/content`.

No hace falta recrear una `.venv` de Windows en Colab. Para B0/ConvNeXt + MLP alcanza PyTorch/torchvision, NumPy, Pillow y tqdm; primero se verifican las versiones preinstaladas. Evitar instalar todo OmniFood8K, Open3D y transformers si no se selecciona esa arquitectura. Las dependencias del runtime pueden requerir instalación en sesiones nuevas; los datos/pesos cacheados sí se reutilizan.

## 13. Segmentación y profundidad: cuándo justifican la descarga

Un módulo de segmentación puede aportar regiones visibles y explicaciones. FoodSeg103 tiene las etiquetas apropiadas para entrenarlo, pero no ofrece pesos físicos de cada región. Área en píxeles no es masa: faltan altura, perspectiva, densidad e ingredientes ocultos.

No propondría agregar SAM/segmentación completa junto con otro encoder y profundidad en el primer presupuesto. Serían varios modelos, otra taxonomía y entrenamiento adicional. Los GFLOPs dependen del segmentador y resolución elegidos; no hay un costo único para “segmentación”. Si elegís esa ruta, la siguiente comparación debe fijar un segmentador concreto antes de implementarlo.

La atención por etiqueta permite estudiar localización sin descargar máscaras y suele ser una ampliación más acotada para este proyecto. También se puede producir Grad-CAM con el encoder ajustado para la presentación; su generación requiere backward y es un costo de visualización, no de inferencia nutricional habitual.

## 14. Paquetes de implementación a elegir

| Paquete | Módulos | Costo de inferencia orientativo | Qué se reutiliza | Datos nuevos |
|---|---|---|---|---|
| A: económico | B0, ajuste final, MLP pequeño, ASL, 100 clases, pérdida de gramos, tabla explícita | Aproximadamente 0,39 GFLOPs nominales del encoder; cabezales del orden de 0,0003 GMACs | Pesos B0 actuales, Nutrition5k, protocolo y conversión | Ninguno |
| B: capacidad mayor | ConvNeXt-Tiny, última etapa ajustable, MLP 256, mismas salidas/pérdidas de A | Aproximadamente 4,46 GFLOPs nominales; MLP de 100 clases ≈0,000248 GMACs | Nutrition5k y lógica de salida; nuevos pesos Tiny de ~109 MB | Ninguno |
| C: espacial | Encoder ganador de A/B + atención reducida para ingredientes/cantidades | Encoder + aproximadamente 0,02–0,04 GMACs de atención, según K/diseño | Mismo dataset; mapas del encoder | Ninguno |
| D: geométrico | RGB + profundidad estimada o supervisión auxiliar | Con segunda rama aumenta mucho; costo exacto por perfilar. Supervisión auxiliar puede quitarse en inferencia | Mismos metadatos, profundidad opcional | Raw depth o mapas generados, según diseño |

Con ConvNeXt-Tiny, entrada pooled 768, hidden 256 y K=100, el MLP tendría **250.057 parámetros** y 248.064 MACs lineales por muestra. El encoder original tiene 28.589.128 pesos; retirando el clasificador queda en 27.820.128. El sistema resultante tendría aproximadamente **28.070.185 parámetros**. La cantidad entrenable dependerá de las etapas descongeladas.

Mi propuesta para tu OK es implementar **A y B como comparación controlada**, más dos variantes pequeñas de cabezal. C queda como alternativa si querés priorizar evidencia espacial para la presentación. D quedaría para una segunda ronda si masa sigue siendo el principal error.

## 15. Experimentos y control del presupuesto de Colab

100 unidades son saldo, no 100 horas. El hardware y su disponibilidad varían; consultar la [documentación de Colab](https://research.google.com/colaboratory/faq.html). El precio y las condiciones finales se confirman en tu pantalla de compra.

Una distribución propuesta, no una promesa de duración:

| Etapa | Tope propuesto de unidades |
|---|---:|
| Prueba de rendimiento y referencia comparable | 10 |
| Ajuste de B0 y objetivos de cantidades | 25 |
| Ajuste de ConvNeXt-Tiny | 35 |
| Confirmación del candidato con otra semilla y evaluación | 20 |
| Reserva | 10 |

Las ablaciones de MLP sobre caches pueden hacerse en CPU. Antes de un entrenamiento GPU largo se medirán unos batches con forward/backward y validación, memoria máxima y segundos por época. Si Colab muestra consumo `r` unidades/hora, un bloque de `U` unidades permite aproximadamente `U/r` horas. La estimación de unidades de una corrida es `r × tiempo_en_horas`; depende del runtime asignado.

Secuencia de experimentos para aislar aportes:

1. Referencia reproducible con K=50, arquitectura/transformación actual y mismas métricas.
2. K=100 con encoder fijo, para medir el cambio de representación y entrenar objetivos directos de gramos.
3. Con igual encoder/vocabulario, comparar cabeza simple, dos capas y capas privadas; elegir usando validación.
4. Comparar BCE/ASL y pérdida de gramos manteniendo lo demás fijo; pocas pruebas, no grid search grande.
5. Ajustar el encoder B0 con ese diseño y aumentos que conserven alimento.
6. Comparar ConvNeXt-Tiny con iguales salidas y presupuesto; comenzar por última etapa y ampliar solo si conviene.
7. Confirmar el mejor candidato con otra semilla si el saldo alcanza; documentar si solo se pudo ejecutar una.
8. Evaluación final en test y análisis de fotografías nuevas del celular.

El split oficial protege test frente a escaneos incrementales del mismo plato. Nuestro `split_train_validation` separa aleatoriamente IDs de train, sin agrupar escaneos relacionados. **Antes de ajustar el encoder conviene auditar esta validación**, porque imágenes casi iguales podrían quedar en ambos lados. Si se cambia a validación por grupos/sesión, se deben recalcular vocabulario, perfiles y todas las referencias; no comparar el F1 de validación viejo con el nuevo como si fueran el mismo conjunto. Es un riesgo identificado en el protocolo, no fuga demostrada con estos archivos.

## 16. Qué medir para decidir

| Parte | Métricas |
|---|---|
| Reconocimiento | Micro-F1, Macro-F1, mAP, precision/recall por clase y grupos de frecuencia |
| Composición | MAE de shares de presentes, gramos por ingrediente presente, error por clase, gramos falsos en ausentes |
| Cantidad absoluta | MAE y PMAE de masa total, mediana del error, errores de platos chicos/grandes |
| Nutrición | MAE kcal y gramos de grasa/carbohidratos/proteína; PMAE por nutriente |
| Aislar problemas | Cantidades con masa total real; nutrientes con masas reales y tabla actual; resultados en vocabulario común |
| Eficiencia | Parámetros totales/entrenables, operaciones, VRAM máxima, segundos/época, latencia batch=1 y throughput, unidades/corrida |
| Generalización | Fotos nuevas; evaluar errores cuantitativos solo cuando haya ingredientes y masas medidos |

PMAE actual = `100 × suma(error absoluto) / suma(valor real absoluto)`, que equivale a MAE normalizado por la media real para valores no negativos. No es MAPE, que promedia porcentajes individuales y es problemática con ceros.

El checkpoint principal se debe seleccionar por una métrica acordada de validación; propondría error de gramos por plato/presentes junto con nutrientes, sin sacrificar excesivamente F1. Guardar checkpoints por tarea es útil, pero no seleccionar al ganador mirando test. Para K distinto, reportar además la evaluación colapsada al vocabulario común y el error en gramos acumulado por plato.

Finalmente, el script de inferencia limita a diez ingredientes y omite shares menores al 1 % para mostrar resultados, mientras la nutrición usa el vector completo. Para la presentación habrá que mostrar también la contribución residual/omitida, de modo que la suma visible sea interpretable.

## 17. Alcance de esta revisión

Se inspeccionaron código, configuraciones, resultados y metadatos locales; se consultaron papers y repositorios primarios; se calcularon tamaños teóricos y errores de representación usando etiquetas reales. No se ejecutó entrenamiento ni inferencia, no se descargaron modelos/datasets pesados y no se hizo commit ni push. La implementación, preparación de Colab y subida a Git corresponden a la próxima etapa después de tu elección de módulos.
