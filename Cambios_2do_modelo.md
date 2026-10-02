| Componente | Modificación |
|---|---|
| **ConvNeXt Small** | Congelar stem y etapa 1. Ajustar etapas 2, 3 y 4 desde la época 4, después de 3 épocas de warmup. |
| **Características multiescala** | Usar mapas de la etapa 3, de **14 × 14**, y de la etapa 4, de **7 × 7**. |
| **Depth Anything V2** | Generar internamente profundidad relativa desde el RGB. Mantener sus pesos congelados inicialmente. |
| **Encoder de profundidad** | Agregar una rama pequeña, entrenable, para procesar ese mapa y producir características a las dos escalas. |
| **Fusión RGB–profundidad** | Combinar ambas ramas en cada escala y proyectarlas a dimensión **256**, con información de posición y escala. |
| **Atención Query2Label** | Pasar de una a **dos capas de atención cruzada**. Mantener 100 consultas y 4 cabezas; atender a **245 posiciones**: 196 + 49. |