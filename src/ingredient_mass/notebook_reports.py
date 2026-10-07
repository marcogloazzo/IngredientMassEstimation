"""Native notebook figures for saved JSON reports, without custom HTML or models."""
import json
import math
from pathlib import Path
from textwrap import fill


def _number(value):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.2f}" if math.isfinite(value) else "—"
    return str(value)


def _show(fig):
    import matplotlib.pyplot as plt
    plt.show()
    plt.close(fig)
    return fig


def _table(title, rows, columns, width=13, first_width=.36):
    import matplotlib.pyplot as plt
    if not rows:
        print(title + ": sin datos guardados.")
        return None
    height = 1.5 + .48 * len(rows)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=13, pad=16)
    labels = [fill(label, 18) for key, label in columns]
    cells = [[fill(_number(row.get(key)), 32 if index == 0 else 18)
              for index, (key, label) in enumerate(columns)] for row in rows]
    widths = [first_width] + [(1-first_width)/(len(columns)-1)]*(len(columns)-1)
    table = ax.table(cellText=cells, colLabels=labels, colWidths=widths,
                     cellLoc="right", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce3ea")
        cell.set_linewidth(.4)
        if row == 0:
            cell.set_facecolor("#e8eff7")
            cell.set_text_props(weight="bold", ha="left" if col == 0 else "center")
        else:
            cell.set_facecolor("white" if row % 2 else "#f7f9fc")
            if col == 0:
                cell.set_text_props(ha="left")
    fig.tight_layout(pad=1.5)
    return _show(fig)


def show_metrics(report, title="Métricas guardadas", details=None):
    recognition = report.get("ingredient_recognition", {})
    mass = report.get("end_to_end_lightweight_mass", {})
    rows = [{"metric":label, "value":recognition[key]*100}
            for key, label in (("micro_f1","F1 micro (%)"), ("macro_f1","F1 macro (%)"),
                               ("micro_precision","Precisión (%)"), ("micro_recall","Recall (%)"))
            if key in recognition]
    rows += [{"metric":label,"value":mass[key]} for key, label in (
        ("ingredient_mass_l1_per_plate_grams","L1 ingredientes (g/plato)"),
        ("ingredient_mass_mae_present_grams","MAE presente (g/ingrediente)"),
        ("false_ingredient_mass_per_plate_grams","Gramos falsos/plato"),
        ("total_mass_mae_grams","MAE masa total (g)")) if key in mass]
    for key, label in (("calories","Calorías (kcal)"), ("fat","Grasa (g)"),
                       ("carbohydrates","Carbohidratos (g)"), ("protein","Proteína (g)")):
        nutrient = mass.get("nutrients", {}).get(key, {})
        if "mae" in nutrient:
            rows.append({"metric":"MAE " + label, "value":nutrient["mae"]})
    for key, label in (("mae_grams","MAE masa auxiliar (g)"),
                       ("bias_grams","Sesgo masa auxiliar (g)"),
                       ("consistency_gap_mae_grams","Discrepancia auxiliar/suma (g)")):
        auxiliary = report.get("auxiliary_total_mass", {})
        if key in auxiliary:
            rows.append({"metric":label,"value":auxiliary[key]})
    if "unfiltered_end_to_end" in report:
        dense = report["unfiltered_end_to_end"]
        rows += [{"metric":"L1 sin filtro (g/plato)","value":dense["ingredient_mass_l1_per_plate_grams"]},
                 {"metric":"MAE masa sin filtro (g)","value":dense["total_mass_mae_grams"]},
                 {"metric":"MAE calorías sin filtro (kcal)","value":dense["nutrients"]["calories"]["mae"]}]
        print("Filtro de masa:", report.get("mass_filter",{}))
    if details:
        print(f"{title}: época {details.get('epoch', 'no registrada')}; "
              f"umbral presencia {_number(details.get('threshold'))}.")
    return [_table(title, rows, [("metric","Métrica"),("value","Valor")], width=10, first_width=.74)]


def show_diagnostic(report, title="Diagnóstico p × c", top_k=15):
    print(f"{title}: {report.get('split', 'split no registrado')}, "
          f"{report.get('samples', '?')} imágenes, época {report.get('epoch', '?')}.")
    print("Oracle usa presencia real y no es desplegable. Umbrales elegidos aquí son exploratorios, no resultados de test.")
    labels = {
        "baseline":"Modelo guardado: p × c", "oracle_remove_false_mass":"Oracle: quitar clases ausentes",
        "oracle_presence_conditional":"Oracle: presencia real × c",
        "hard_product_report_threshold":"Recorte de p × c al umbral del reporte",
        "hard_conditional_report_threshold":"Usar c al umbral del reporte",
        "temperature_0.5":"Gate con temperatura 0,5", "temperature_2":"Gate con temperatura 2",
        "hard_product":"Mejor recorte p × c en esta validación",
        "hard_conditional":"Mejor recorte c en esta validación",
        "unfiltered":"Sin filtro: p × c",
    }
    if "mass_filter" in report:
        labels["baseline"] = "Modelo guardado: p × c con filtro"
    present = report.get("global_present", {})
    rows = [{"metric":"MAE condicional c presente (g/ingrediente)","value":present.get("conditional_mae_g")},
            {"metric":"MAE final CON filtro (g/ingrediente)" if "mass_filter" in report else "MAE final p × c presente (g/ingrediente)","value":present.get("final_mae_g")},
            {"metric":"Sesgo condicional presente (g/ingrediente)","value":present.get("conditional_bias_g")},
            {"metric":"Sesgo final presente (g/ingrediente)","value":present.get("final_bias_g")}]
    if "unfiltered_mae_g" in present:
        rows.append({"metric":"MAE p × c presente SIN filtro (g/ingrediente)","value":present["unfiltered_mae_g"]})
    auxiliary = report.get("auxiliary_total_mass", {})
    if auxiliary:
        rows += [{"metric":"MAE masa auxiliar (g)","value":auxiliary.get("mae_g")},
                 {"metric":"Discrepancia auxiliar/suma (g)","value":auxiliary.get("gap_to_ingredient_sum_mae_g")}]
    figures = [_table(title + " · cantidad presente",rows,[("metric","Métrica"),("value","Valor")],width=11,first_width=.75)]
    scenarios = report.get("scenarios", []) + list(report.get("best_validation_thresholds", {}).values())
    scenarios = [{**row,"label":labels.get(row.get("name"),row.get("name","Sin nombre"))} for row in scenarios]
    figures.append(_table("Contrafactuales con c fijo", scenarios, [
        ("label","Escenario"),("threshold","Umbral τ"),
        ("ingredient_l1_g_per_plate","L1 g/plato"),("false_g_per_plate","Falsos g/plato"),
        ("total_mass_mae_g","MAE total g")],width=15,first_width=.42))
    classes = report.get("classes_by_l1", [])[:top_k]
    figures.append(_table(f"{len(classes)} ingredientes con mayor contribución al L1", classes, [
        ("name","Ingrediente"),("support","Soporte"),("conditional_mae_present_g","MAE c presente g"),
        ("final_mae_present_g","MAE p×c presente g"),("gate_effect_mae_g","Δ MAE por gate g"),
        ("final_bias_present_g","Sesgo final g"),("l1_contribution_g_per_plate","L1 g/plato"),
        ("false_g_per_plate","Falsos g/plato")],width=18,first_width=.24))
    print("Δ MAE por gate positivo: p empeora el error con c fijo. Sesgo negativo: subestimación. Soporte bajo: promedio incierto.")
    if "mass_filter" in report:
        print("Filtro de masa aplicado:",report["mass_filter"])
        figures.append(_table("Escalas train y efecto del recorte",classes,[
            ("name","Ingrediente"),("scale_grams","Escala train g"),("unfiltered_mae_present_g","MAE sin filtro g"),
            ("final_mae_present_g","MAE filtrado g"),("filter_effect_mae_g","Δ MAE por corte g")],width=15,first_width=.35))
    figures.append(_table("Presencia correcta y errores de reconocimiento", report.get("recognition_buckets", []), [
        ("group","Grupo"),("ingredient_entries","Entradas"),("mean_probability","p medio"),
        ("conditional_mae_g","MAE c g"),("final_mae_g","MAE p×c g"),("final_bias_g","Sesgo final g")],
        width=13,first_width=.18))
    print("TP/FN son ingredientes reales; FP/TN son ausentes. MAE presente es por ingrediente, L1 se suma por plato.")
    return [fig for fig in figures if fig is not None]


def show_comparison(report, title="Comparación con V5"):
    import matplotlib.pyplot as plt
    candidate = "V6" if "v6" in (title+str(report.get("right",""))).lower() else "V5"
    print(report.get("warning", "No se registró la verificación de datos/split."))
    rows = []
    for row in report.get("metrics", []):
        delta = row["right_minus_left"]
        improvement = delta > 0 if row["higher_is_better"] else delta < 0
        rows.append({**row,"change":"Igual" if abs(delta)<1e-8 else "Mejora" if improvement else "Empeora"})
    figures = [_table(title,rows,[("metric","Métrica"),("left","Referencia"),("right",candidate),
                                  ("right_minus_left",f"Δ {candidate} − referencia"),("change","Cambio numérico")],width=14,first_width=.40)]
    print("Para F1, Δ se expresa en puntos porcentuales. Mejora/empeora indica dirección numérica, no significancia estadística.")
    grams = [row for row in rows if row["metric"] in ("L1 ingredientes (g/plato)","MAE masa total (g)","Gramos falsos/plato")]
    if grams:
        fig, ax = plt.subplots(figsize=(11,4.5))
        positions = list(range(len(grams)))
        ax.bar([x-.18 for x in positions],[r["left"] for r in grams],width=.36,label="Referencia",color="#7f99b4")
        ax.bar([x+.18 for x in positions],[r["right"] for r in grams],width=.36,label=candidate,color="#226b96")
        ax.set_xticks(positions,[fill(r["metric"],26) for r in grams])
        ax.set_ylabel("Gramos (L1/falsos por plato; MAE total por imagen)")
        ax.set_title(title + " · errores de masa")
        ax.legend(); ax.grid(axis="y",alpha=.2); ax.set_axisbelow(True)
        fig.tight_layout()
        figures.append(_show(fig))
    return [fig for fig in figures if fig is not None]


def show_saved_report(path, kind, title=None):
    """Reads a cached JSON. Does not run training/inference or write artifacts."""
    path = Path(path)
    if not path.is_file():
        print(f"Todavía no está disponible el reporte: {path}")
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    if kind == "diagnostic":
        return show_diagnostic(report,title or "Diagnóstico p × c")
    if kind == "comparison":
        return show_comparison(report,title or "Comparación con V5")
    if kind == "metrics":
        details_path = path.with_name("evaluation_details.json")
        details = json.loads(details_path.read_text(encoding="utf-8")) if details_path.is_file() else None
        return show_metrics(report,title or "Métricas guardadas",details)
    raise ValueError(f"Unknown report kind: {kind}")
