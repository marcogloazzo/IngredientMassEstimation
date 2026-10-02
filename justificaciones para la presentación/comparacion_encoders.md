# Opciones para reemplazar EfficientNet-B0

Para reemplazar a EfficientNet en su tarea de **extraer características visuales del plato**, priorizaría **EfficientNet-B2 y ConvNeXt-Tiny**: permiten comparar una mejora moderada de capacidad con una arquitectura más potente.

La siguiente tabla utiliza los pesos predeterminados de torchvision. **Top-1 es precisión en ImageNet, no reconocimiento de ingredientes ni estimación de masas.**

| Encoder y fuente | Parámetros¹ | GFLOPs/imagen | Cómputo relativo a B0 | Top-1 ImageNet | Evaluación para tu proyecto |
|---|---:|---:|---:|---:|---|
| [MobileNetV3-Large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_large.html) | 5,5 M | 0,22 | 0,6× | 75,27 % | Opción para reducir cómputo |
| [EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html) | 5,3 M | 0,39 | 1× | 77,69 % | Baseline actual; muy económica |
| [EfficientNet-B2](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b2.html) | 9,1 M | 1,09 | 2,8× | 80,61 % | Primer reemplazo razonable con poco cómputo |
| [ResNet-50](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) | 25,6 M | 4,09 | 10,5× | 80,86 % | Referencia clásica; menor prioridad |
| [ConvNeXt-Tiny](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html) | 28,6 M | 4,46 | 11,4× | 82,52 % | Mi candidato principal para buscar mayor precisión |
| [Swin-T](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html) | 28,3 M | 4,49 | 11,5× | 81,47 % | Alternativa Transformer de costo parecido a ConvNeXt |
| [EfficientNetV2-S](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_v2_s.html) | 21,5 M | 8,37 | 21,5× | 84,23 % | Potente, pero más costosa |
| [ViT-B/16](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html) | 86,6 M | 17,56 | 45× | 81,07 % | Poco atractivo para tus restricciones |

¹ Parámetros del modelo original, incluyendo el clasificador de ImageNet; al eliminarlo disminuyen. Las resoluciones son 224×224, salvo B2 —288×288— y V2-S —384×384—. Los GFLOPs comparan operaciones: **11× más operaciones no implica necesariamente 11× más tiempo**, porque depende del hardware y la implementación.

El costo de entrenamiento también depende de cómo uses el encoder:

| Estrategia | Qué entrenás | Costo | Ventaja |
|---|---|---|---|
| **Encoder congelado + características guardadas** | Solamente los cabezales | Bajo después de extraer las características | Es nuestro enfoque actual; viable en CPU |
| Ajustar los últimos bloques | Parte del encoder y cabezales | Intermedio; GPU recomendable | Adapta las características a alimentos |
| Ajustar todo el encoder | Todos los parámetros | Alto; GPU y más memoria | Mayor flexibilidad, con riesgo de sobreajuste |

También podés evaluar **ConvNeXt con los pesos de OmniFood8K**. Es una alternativa de preentrenamiento: habría que verificar qué variante contiene tu checkpoint y extraer la rama correspondiente. Puede aportar conocimiento sobre comida, pero su desempeño aislado para ingredientes y gramos todavía debe medirse.

Para comparar los encoders en **tu tarea**, las métricas relevantes son:

| Métrica | Qué evalúa | Baseline B0 ya obtenida |
|---|---|---:|
| Micro-F1 ↑ | Reconocimiento global de ingredientes | 0,641 |
| Macro-F1 ↑ | Desempeño equilibrado entre clases | 0,550 |
| MAE por ingrediente presente ↓ | Error en gramos por ingrediente | 18,13 g |
| MAE de masa total ↓ | Error en gramos del plato | 51,58 g |
| MAE de calorías ↓ | Error nutricional final | 99,94 kcal |

Estos resultados corresponden a **B0 más nuestros cabezales**, según [la evaluación guardada](../outputs/evaluation/metrics.json). Todavía no tenemos esas métricas para los otros encoders.

**Mi recomendación:** comparar B0, B2 y ConvNeXt-Tiny congelados, usando los mismos splits, cabezales y presupuesto de ajuste. Elegir con validación y evaluar finalmente en test. Registrar además tiempo de extracción, latencia por imagen y memoria máxima: así podrás justificar la elección por precisión y costo reales.
