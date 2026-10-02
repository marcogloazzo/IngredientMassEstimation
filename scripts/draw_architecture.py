"""Export a detailed architecture diagram; no model loading or training."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK = "#172b45"
MUTED = "#506580"
COLORS = {
    "frozen": ("#edf2f7", "#8799ad"),
    "partial": ("#fff2df", "#c48a35"),
    "learned": ("#eef0fc", "#6a71c2"),
    "derived": ("#e8f5f1", "#479981"),
    "fixed": ("#f4f7fa", "#9aacbd"),
}


def build():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, ax = plt.subplots(figsize=(24, 18))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 24), ylim=(0, 18))
    ax.axis("off")
    texts = []

    def text(x, y, value, size=12, color=INK, weight="normal", **kwargs):
        artist = ax.text(x, y, value, fontsize=size, color=color,
                         fontweight=weight, va="center", **kwargs)
        texts.append(artist)
        return artist

    def box(x, y, w, h, title, lines, kind="learned", title_size=14, body_size=12):
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.12",
            linewidth=1.15, edgecolor=edge, facecolor=fill, zorder=2,
        ))
        text(x + 0.18, y + h - 0.30, title, title_size, weight="bold", zorder=3)
        top = y + h - 0.70
        step = min(0.33, (h - 0.86) / max(len(lines) - 1, 1))
        for i, line in enumerate(lines):
            text(x + 0.18, top - step * i, line, body_size, zorder=3)

    def arrow(points, color=MUTED, dashed=False):
        path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
        ax.add_patch(FancyArrowPatch(
            path=path, arrowstyle="-|>", mutation_scale=13,
            linewidth=1.35, color=color, linestyle="--" if dashed else "-",
            zorder=1,
        ))

    def section(y, number, title):
        text(0.7, y, f"{number}  {title}", 15, weight="bold")

    text(0.7, 17.35, "IngredientMassEstimation", 29, weight="bold")
    text(0.7, 16.88,
         "ConvNeXt Small + adaptación Query2Label · 100 clases (incluye other) · B = tamaño del lote",
         15, color=MUTED)
    for x, kind, label in [
        (0.75, "frozen", "Encoder congelado"),
        (5.15, "partial", "Encoder ajustable desde época 4"),
        (12.05, "learned", "Módulos nuevos: entrenables desde época 1"),
        (20.20, "derived", "Operaciones / salidas"),
    ]:
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch((x, 16.27), 0.20, 0.20,
                                   boxstyle="round,pad=0.01", facecolor=fill,
                                   edgecolor=edge, linewidth=1))
        text(x + 0.31, 16.37, label, 10.8)

    section(15.86, "01", "Imagen → representación visual espacial")
    box(0.7, 13.55, 2.05, 1.88, "Foto RGB", [
        "Imagen de un plato", "RGB: 3 canales", "Resolución original libre",
    ], "fixed", title_size=13, body_size=10.6)
    box(3.05, 13.55, 3.05, 1.88, "Preprocesamiento", [
        "Resize: lado corto 230", "Center crop: 224 × 224",
        "Normalización ImageNet", "En train: flip + color jitter",
        "Entrada: B × 3 × 224 × 224",
    ], "fixed", body_size=10.7)
    box(6.45, 13.55, 7.25, 1.88, "ConvNeXt Small · stem + etapas 1–3", [
        "Etapa 1: 3 bloques     → B × 96 × 56 × 56",
        "Etapa 2: 3 bloques     → B × 192 × 28 × 28",
        "Etapa 3: 27 bloques   → B × 384 × 14 × 14",
        "Pesos ImageNet-1K · sin gradientes · modo eval",
    ], "frozen")
    box(14.05, 13.55, 4.45, 1.88, "Última etapa del encoder", [
        "Downsampling 384 → 768",
        "Etapa 4: 3 bloques + LayerNorm",
        "Mapa Fvisual: B × 768 × 7 × 7",
        "Épocas 1–3: también congelada",
    ], "partial", title_size=13, body_size=11.5)
    box(18.85, 13.55, 4.45, 1.88, "Proyección espacial", [
        "Conv 1 × 1: 768 → 256",
        "Flatten espacial + transpose",
        "Memoria F: B × 49 × 256",
        "49 regiones = grilla 7 × 7",
    ], body_size=11.5)
    for start, end in [(2.75, 3.05), (6.10, 6.45), (13.70, 14.05), (18.50, 18.85)]:
        arrow([(start, 14.50), (end, 14.50)])

    section(12.97, "02", "Consultas por ingrediente → atención cruzada → MLP compartido")
    box(0.7, 10.97, 5.50, 1.47, "100 consultas aprendidas E", [
        "Embedding: 100 × 256; se replica para B",
        "Una consulta por clase · Z₀ = ceros",
    ], body_size=12)
    box(0.7, 9.35, 5.50, 1.30, "Posiciones 2D P", [
        "Seno/coseno · 49 × 256 · sin parámetros",
        "Codifican la posición de cada región",
    ], "fixed", body_size=11.5)
    box(7.30, 9.35, 7.70, 3.09, "Decoder Query2Label · 1 capa", [
        "Q = Z₀ + E        K = F + P        V = F",
        "Cross-attention: 4 cabezas, dimensión 256",
        "Residual + LayerNorm",
        "FFN: 256 → 1024 → 256, GELU + dropout 0,1",
        "Residual + LayerNorm · sin self-attention",
        "Salida Z: B × 100 × 256",
        "Atención exportable: B × 100 × 7 × 7",
    ], body_size=12)
    box(15.35, 9.35, 3.15, 3.09, "Proyección", [
        "Linear: 256 → 768", "Por cada ingrediente", "B × 100 × 768",
    ], title_size=13, body_size=11.6)
    box(18.85, 9.35, 4.45, 3.09, "MLP compartido", [
        "LayerNorm(768)",
        "Linear: 768 → 384",
        "GELU + dropout 0,3",
        "Linear: 384 → 192",
        "GELU + dropout 0,3",
        "H: B × 100 × 192",
        "Mismos pesos para las 100 clases",
    ], body_size=11.5)
    arrow([(6.20, 11.72), (7.30, 11.72)])
    arrow([(6.20, 10.00), (7.30, 10.00)])
    arrow([(21.075, 13.55), (21.075, 13.22), (14.10, 13.22), (14.10, 12.44)])
    text(15.20, 13.32, "Memoria visual F", 10.7, color=MUTED)
    arrow([(15.0, 10.90), (15.35, 10.90)])
    arrow([(18.5, 10.90), (18.85, 10.90)])

    section(8.74, "03", "Tres ramas de predicción → gramos → nutrientes")
    # Shared representation branches along a common bus below the decoder.
    ax.plot([3.40, 21.075], [8.30, 8.30], color=MUTED, linewidth=1.35, zorder=1)
    ax.plot([21.075, 21.075], [9.35, 8.30], color=MUTED, linewidth=1.35, zorder=1)
    for x in (3.40, 9.25, 15.10):
        arrow([(x, 8.30), (x, 7.98)])

    box(0.70, 6.18, 5.40, 1.80, "Presencia por ingrediente", [
        "GroupWiseLinear: 192 → 1 por clase",
        "Logits zᵢ → sigmoid → pᵢ",
        "100 probabilidades · B × 100",
        "Pesos específicos para cada ingrediente",
    ], body_size=11.7)
    box(6.55, 6.18, 5.40, 1.80, "Cantidad positiva por ingrediente", [
        "GroupWiseLinear: 192 → 1 por clase",
        "aᵢ = softplus(logit de cantidad)",
        "100 cantidades positivas · B × 100",
        "No son gramos ni proporciones todavía",
    ], body_size=11.7, title_size=13)
    box(12.40, 6.18, 5.40, 1.80, "Masa total del plato", [
        "Promedio de H sobre las 100 clases",
        "192 → 128 → 1; GELU + dropout 0,3",
        "Softplus → masa M > 0, en gramos",
        "Salida: B · promedio sin umbral de presencia",
    ], body_size=11.5)
    text(18.85, 7.50, "Representación H", 12, weight="bold")
    text(18.85, 7.12, "compartida por las tres ramas", 11, color=MUTED)
    text(18.85, 6.74, "No se usan 100 MLP independientes.", 10.8, color=MUTED)

    box(0.70, 3.90, 5.40, 1.35, "Reporte de presencia", [
        "Presente si pᵢ ≥ τ; τ se calibra en validación",
        "El umbral no recorta gramos ni nutrientes",
    ], "fixed", body_size=11.1, title_size=13)
    box(6.55, 3.90, 5.40, 1.80, "Proporciones de masa s", [
        "uᵢ = pᵢ · aᵢ + ε; ε = 10⁻⁸",
        "sᵢ = uᵢ / Σⱼ uⱼ",
        "Σᵢ sᵢ = 1 · salida: B × 100",
        "Normalización suave y diferenciable",
    ], "derived", body_size=11.7)
    box(12.40, 3.90, 5.40, 1.80, "Gramos por ingrediente", [
        "gᵢ = sᵢ · M",
        "Σᵢ gᵢ = M · salida: B × 100",
        "Conservación de la masa total predicha",
        "Incluye ingredientes ausentes y other",
    ], "derived", body_size=11.7)
    box(18.25, 3.90, 5.05, 1.80, "Tabla nutricional fija T", [
        "T: 100 × 4, nutrientes por gramo",
        "n = g @ T → B × 4",
        "kcal, grasa, carbohidratos y proteína",
        "Tabla sin gradientes; operación diferenciable",
    ], "derived", body_size=11.1, title_size=13)
    arrow([(9.25, 6.18), (9.25, 5.70)])
    arrow([(15.10, 6.18), (15.10, 5.70)])
    arrow([(3.40, 6.18), (3.40, 5.93), (6.10, 5.93), (6.10, 4.95), (6.55, 4.95)])
    arrow([(3.40, 5.93), (3.40, 5.25)], dashed=True)
    arrow([(11.95, 4.80), (12.40, 4.80)])
    arrow([(17.80, 4.80), (18.25, 4.80)])

    section(3.37, "04", "Supervisión conjunta durante entrenamiento")
    losses = [
        ("L_presencia", ["ASL sobre logits, por defecto",
                         "γ− = 4; γ+ = 1; clip = 0,05",
                         "Alternativa: BCE ponderada"]),
        ("L_proporciones", ["L1 ponderada sobre sᵢ",
                            "Peso 5 si presente; 1 si ausente",
                            "Dividir por suma de pesos"]),
        ("L_masa", ["MAE de masa total / μtrain",
                    "μtrain ≈ 219,11 g; escala fija",
                    "Calculada solo con train"]),
        ("L_gramos", ["Media por plato de Σᵢ |gᵢ − gᵢ*|",
                      "Dividir por 300 g",
                      "Supervisa las 100 clases"]),
        ("L_nutrición", ["Media de |n − n*| / escalas",
                         "[500 kcal, 30 g, 80 g, 40 g]",
                         "Orden: kcal, grasa, carb., prot."]),
    ]
    for i, (title, lines) in enumerate(losses):
        box(0.70 + i * 4.58, 1.54, 4.28, 1.44, title, lines,
            "fixed", title_size=12.7, body_size=10.5)
    text(12.0, 1.07,
         "L = L_presencia + 5 L_proporciones + L_masa + L_gramos + 0,25 L_nutrición",
         16, weight="bold", ha="center")
    text(0.70, 0.61,
         "AdamW · LR módulos nuevos: 3 × 10⁻⁴ · LR encoder ajustable: 10⁻⁵ · batch 8 × acumulación 4 · AMP CUDA",
         11.1, color=MUTED)
    text(0.70, 0.27,
         "Selección: menor L1 de gramos por plato en validación · Fuente: spatial_model.py, model.py y config/query2label_100.json",
         10.3, color=MUTED)

    # Check all text against the canvas before exporting; visual review follows.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for artist in texts:
        bounds = artist.get_window_extent(renderer)
        if not (canvas.x0 <= bounds.x0 and bounds.x1 <= canvas.x1
                and canvas.y0 <= bounds.y0 and bounds.y1 <= canvas.y1):
            raise ValueError(f"Text outside figure: {artist.get_text()}")
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        path = OUT / f"arquitectura_query2label_detallada.{extension}"
        fig.savefig(path, dpi=200)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    build()
