# Auditoría de secuencias y split agrupado de Nutrition5k

Estos scripts no entrenan, no ejecutan inferencia ni modifican los notebooks,
los checkpoints o `data/processed_100`. No requieren torch/GPU; usan numpy y
Pillow. En esta PC se pueden ejecutar con `py -3.12`; en Colab, con `python`.

## 1. Generar candidatos y revisión visual

Desde la carpeta del proyecto:

```powershell
py -3.12 scripts/audit_scan_groups.py --output-dir reports/scan_group_audit
```

El directorio debe ser nuevo o vacío. Para repetir con otros parámetros, usar
otro nombre, por ejemplo `reports/scan_group_audit_gap300`. Nunca se borra ni
sobrescribe una auditoría anterior.

Archivos generados:

- `audit_manifest.json`: metadata completa, grupos **candidatos**, enlaces,
  hashes de fuentes/RGB, parámetros y comparación con el split anterior.
- `audit_summary.json`: cantidades, grupos mezclados y enlaces excluidos.
- `index.html`, `review_*.html`, `assets/`: revisión visual local y portable.

Abrir `index.html` en un navegador. Al moverlo a otro equipo, copiar TODA la
carpeta, no solamente el índice. El informe no consulta servicios externos.

Para regenerar solo la presentación de una auditoría guardada (en otra carpeta):

```powershell
py -3.12 scripts/audit_scan_groups.py --render-existing-audit reports/scan_group_audit/audit_manifest.json --output-dir reports/scan_group_review
```

Comprueba las fuentes/hashes y conserva el mismo `audit_id`; las decisiones
exportadas siguen siendo compatibles. La carpeta de salida debe ser nueva.

### Qué significa un grupo candidato

La detección usa la cafetería, timestamps, conservación no decreciente de las
cantidades previas y coherencia del incremento de masa. Incluye comienzos con
un ingrediente, recetas que agregan varios componentes y metadata sin RGB.
La ventana temporal por defecto es de 600 segundos y la tolerancia de
ingredientes de 0,1 g. Son heurísticas, NO identificadores oficiales ni pruebas
de identidad. La diferencia RGB del área central es descriptiva: no decide
automáticamente si dos imágenes representan el mismo plato.

El split oficial train/test es una restricción dura. Los enlaces que lo
contradicen se registran como `excluded_from_grouping`; también se controlan
puentes indirectos mediante escaneos sin RGB. Test nunca se mueve para acomodar
una hipótesis. Inspeccionar esos enlaces en `audit_manifest.json` si hay dudas
sobre la consistencia de las fuentes oficiales.

### Revisar y exportar

En cada grupo:

1. Comparar posiciones de los alimentos anteriores, pesos y alimentos añadidos;
   el plato/fondo o compartir receta no bastan.
2. Mirar vecinos temporales, especialmente en singleton y grupos grandes.
3. Si es una secuencia única, escribir fundamento y pulsar **Mantener juntos**.
4. Si mezcla platos, escribir una secuencia por línea en Particiones, con IDs
   separados por coma; pulsar **Guardar particiones editadas**. Todos los IDs
   deben aparecer una sola vez.
5. Si aparece una continuación de otro grupo, registrar una fusión manual al
   final de la página: `dish_ID1,dish_ID2 | fundamento`. Se fusionan las
   secuencias revisadas a las que pertenecen esos escaneos.
6. Exportar el JSON de CADA página. Guardar/exportar avances: el almacenamiento
   del navegador puede no persistir. Usar solamente la última exportación de
   cada página, no versiones contradictorias.

Los JSON están ligados al `audit_id`; no se pueden mezclar auditorías. Revisar
un grupo no demuestra que no existan enlaces omitidos: se recomienda buscar
nuevamente candidatos con otras ventanas/tolerancias y revisar los cruces
residuales. **No se afirma independencia certificada**, ni siquiera después
de la revisión humana.

## 2. Split PROVISIONAL conservador, sin revisión

```powershell
py -3.12 scripts/prepare_grouped_nutrition5k.py --provisional --output-dir data/processed_100_grouped_provisional
```

Esta opción explícita mantiene juntos todos los candidatos compatibles con
la frontera oficial. Elimina cruces de los enlaces detectados, pero puede
unir distintos platos y omitir otras relaciones. Sirve como preparación
provisional, NO como garantía de validación limpia.

## 3. Preparar el split después de revisar

Guardar los JSON exportados en una carpeta propia. En PowerShell:

```powershell
$reviewFiles = (Get-ChildItem -LiteralPath reports/scan_group_audit/reviews -Filter 'review_*.json').FullName
py -3.12 scripts/prepare_grouped_nutrition5k.py --review $reviewFiles --output-dir data/processed_100_grouped_reviewed
```

Alternativa con rutas explícitas:

```powershell
py -3.12 scripts/prepare_grouped_nutrition5k.py --review reports/scan_group_audit/reviews/review_001.json reports/scan_group_audit/reviews/review_002.json --output-dir data/processed_100_grouped_reviewed
```

Incluir todas las páginas reales, no solo las dos del ejemplo. El modo revisado
se niega a preparar mientras falten decisiones de grupos con RGB utilizado.
Valida cobertura, hashes, notas, particiones, fusiones y ausencia de cruces con
test oficial. Rechaza cambios en metadata o imágenes desde la auditoría.

El 15% de validación se aplica a **grupos con RGB del train oficial**, con seed
42, no a imágenes individuales. La fracción real de imágenes puede diferir;
se informa junto con soporte por ingrediente y distribuciones de masa. No se
busca una partición según rendimiento del modelo ni se optimiza sobre test.

## 4. Resultados y próximo entrenamiento (NO ejecutado por estos scripts)

Cada nueva carpeta preparada contiene:

- `records.jsonl` con `scan_group_id` y los nuevos splits.
- `vocabulary.json` recalculado exclusivamente con el nuevo train.
- `scan_groups.json`, `preparation_config.json` y `summary.json` con procedencia.
- `train_ids.txt`, `validation_ids.txt`, `test_ids.txt`.

El informe controla test original, cruces de grupos, cruces de candidatos,
soporte de clases, clases de validación ausentes en train y media de masa train.
Los enlaces cruzados después de revisión pueden corresponder a propuestas
rechazadas por el revisor: inspeccionarlos, no asumir automáticamente fuga.

Para una nueva corrida, pasar esta carpeta como `--processed-dir` al entrenador
y una carpeta NUEVA de `--output-dir`. **No usar `--resume` de V2 anterior**:
esos pesos ya vieron parte de la nueva validación. No reusar sus umbrales,
predicciones, perfiles nutricionales ni caches de etiquetas/features. El
entrenador recalcula la media de masa y pesos positivos con el nuevo train.

El cache de profundidad por `dish_id` puede reutilizarse si permanecen iguales
RGB, geometría y pesos de Depth Anything; verificarlo con los controles existentes.
La tabla de clases puede cambiar al recalcular el top 100, por lo que comparar
con V2 anterior exige documentar esa diferencia. Comparar ASL/BCE entre nuevas
corridas usando EXACTAMENTE la misma carpeta agrupada.

Git: los dos scripts, `src/ingredient_mass/scan_groups.py`, el ajuste de import
perezoso en `src/ingredient_mass/__init__.py`, pruebas y este documento. Los
notebooks NO se modifican. Drive: auditoría completa, exports de revisión y
carpeta preparada. Los reportes/imagenes/arrays grandes siguen fuera de Git.
Una vez estable, conviene versionar también el manifiesto revisado y sus hashes
(sin imágenes) para reproducibilidad.

Pruebas locales:

```powershell
py -3.12 -m unittest discover -s tests -p test_scan_groups.py -v
```
