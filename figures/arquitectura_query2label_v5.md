# Figura de arquitectura V5

Versión final:

- [PNG de alta resolución, 4800 × 2720](arquitectura_query2label_v5_final.png).
- [SVG editable, texto y elementos vectoriales](arquitectura_query2label_v5_final.svg).

Estilo inspirado en `OmniFood8K/Code/source/network7.png`: fondo blanco,
contenedores discontinuos azules, bloques pastel y pilas de características.
La topología se verificó contra la configuración y los módulos actuales de V5;
no se copiaron los módulos de la arquitectura de referencia.

## Qué representa

Una única imagen RGB alimenta ConvNeXt Small y el estimador congelado Depth
Anything V2 Small. Las etapas 3/4 se fusionan con características de profundidad
estimada mediante dos compuertas residuales escalares. Los 400+100 tokens
alimentan dos capas de atención cruzada y un MLP **compartido** 768→384→192.
Presencia y cantidad condicional producen `m_i=p_i*c_i`; la masa principal se
suma y los nutrientes se calculan con la matriz fija del vocabulario de train.

El cabezal auxiliar hace pooling de la memoria visual, no de las consultas ni
los gramos. No reescala las salidas. Las líneas naranjas muestran únicamente
supervisión: masa real→L_aux y masa predicha principal→L_cons; M_aux entra en ambas.

El plato dibujado es una ilustración esquemática, no una predicción. Las flechas
que se cruzan sin un punto marcado no comparten señal. GN significa GroupNorm.

Los prompts de exploración están en [especificación visual](arquitectura_query2label_v5_prompt.md).
La exportación final es determinista para evitar ambigüedades en flechas/textos.
No se ejecutó ningún entrenamiento ni inferencia al construir esta figura.

## Reexportar sin sobrescribir

```powershell
& .\.venv-v3\Scripts\python.exe scripts/draw_architecture_v5.py --output-dir outputs/figuras_v5_reexport
```

El script exporta PNG, SVG y un PDF vectorial auxiliar a una carpeta nueva.
Verifica las dimensiones/flags V5 y que todos los textos quepan en paneles/lienzo.
