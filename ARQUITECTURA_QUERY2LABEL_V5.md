# V5 — masa total auxiliar con MLP compartido

Estado: implementada; no entrenada. Experimento pendiente luego de V4 split_only.
V5 equivale a la configuración aux_only prevista en V4, con nuevo tipo de modelo,
configuración, notebook y namespace para preservar los resultados anteriores.
No introduce una arquitectura adicional diferente de esa ablación.

## Cambio aislado

Se vuelve al MLP compartido de V3, **768 → 384 → 192**, para presencia y cantidad.
Se activa el cabezal auxiliar de masa total sobre memoria visual multiescala:

```text
RGB → ConvNeXt + profundidad estimada congelada → fusión → memoria multiescala
                                                       ├→ Query2Label → MLP compartido → p y c → p×c → gramos
                                                       └→ media espacial → LayerNorm → 256→128→1 → masa auxiliar
```

La rama auxiliar no recibe masas/etiquetas como entrada. El entrenamiento la
supervisa con masa real; su escala e inicialización usan la media de train.

Se conservan BCE ponderada, corrección de prior para el gate, 100 clases,
RGB completo 320×320, fusión etapas 3/4, dos capas de Cross-Attention,
Depth Anything V2 Small congelado, CNN de profundidad entrenable y ajuste de
etapas 2–4 tras tres épocas. Mismos split provisional, vocabulario y mapas V3.

## Pérdidas y salidas

```text
L_V5 = L_V3
       + 0.25 × mean(|M_aux − M_real|) / media_train
       + 0.10 × mean(|M_aux − sum_i(p_i×c_i)|) / media_train
```

Consistencia transmite gradientes a ambas ramas. Los pesos son valores de partida
experimentales, no óptimos comprobados. La media train se recalcula; no fijar los
219,11 g del split histórico (la partición agrupada V3 tuvo 216,89 g).

En inferencia **m_i=p_i×c_i**, masa total=sum(m_i) y nutrientes=m@perfiles_train.
La masa auxiliar se informa separadamente: **no reemplaza ni reescala** las masas.
El umbral de presencia tampoco corta gramos. No se combina con los MLP separados.

## Qué evaluar

Comparar V5 vs V3 para aislar el aporte auxiliar; V5 vs V4 split_only compara
alternativas de arquitectura, no una única modificación incremental sobre V4.
Mismos datos/seed/presupuesto, checkpoint elegido por L1, paciencia 8, máximo 100.
La F1 calibrada usa umbrales elegidos en validación; no confundirla con F1 fija.

Revisar L1 de ingredientes, MAE presente, gramos falsos, MAE total y nutrientes.
Un MAE auxiliar menor no demuestra mejora de los resultados desplegados.
No ajustar arquitectura/hiperparámetros según test; una sola corrida no demuestra
significancia de diferencias pequeñas.

## Implementación y compatibilidad

- Tipo: `convnext_small_query2label_auxiliary_mass_v5`.
- Reutiliza `specialized_model.py` con MLP compartido y auxiliar obligatorio.
- Config única: `config/query2label_100_v5.json`.
- Checkpoints anteriores no son resume de V5. Empezar corrida nueva desde los
  mismos pesos ImageNet que los controles; no heredar pesos entrenados V3/V4.
- Resultados V5 fuera de namespaces `v2/`, `v3/`, `v4/`.
- Datos y mapas V3 pueden leerse/reutilizarse sin regenerarlos.
- Se guardan métricas auxiliares, pérdidas separadas, predicciones de mejor
  validación y checkpoints resumibles, igual que la infraestructura V4.

Pruebas: `tests/test_v5.py` con tensores sintéticos, sin descargas ni training real.
