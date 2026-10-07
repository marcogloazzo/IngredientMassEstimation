"""Exact V5 architecture in the paper-reference style. No model execution."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#3e6fae"
GREEN = "#17936d"
ORANGE = "#df8c29"
INK = "#182836"
COLORS = {"linear":"#eed0d2", "activation":"#c8dfb8", "attention":"#bbd1ec",
          "map":"#d7e5ef", "plain":"#eff5fb", "loss":"#fff2df"}


def build():
    config = json.loads((ROOT/"config/query2label_100_v5.json").read_text(encoding="utf-8"))
    assert config["split_task_mlps"] is False and config["auxiliary_total_mass"] is True
    assert config["image_size"] == 320 and config["attention_dim"] == 256
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "savefig.facecolor":"white"})
    fig, ax = plt.subplots(figsize=(30,17))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.set(xlim=(0,3000),ylim=(1700,0)); ax.axis("off")
    artists, constraints = [], []

    def text(x,y,label,size=18,weight="normal",color=INK,align="center",**kwargs):
        item = ax.text(x,y,label,ha=align,va="center",fontsize=size,fontweight=weight,color=color,zorder=8,**kwargs)
        artists.append(item)
        return item

    def group(x,y,w,h,title,orange=False):
        color = ORANGE if orange else BLUE
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=12",
            facecolor="#fffaf3" if orange else "#f5f9fe",edgecolor=color,linewidth=1.8,
            linestyle=(0,(6,4)),zorder=0))
        item = text(x+w/2,y+35,title,size=22,weight="bold")
        constraints.append((item,w-28,48))

    def box(x,y,w,h,label,kind="plain",size=17):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0,rounding_size=4",
            facecolor=COLORS[kind],edgecolor="#66788a",linewidth=1.2,zorder=4))
        item = text(x+w/2,y+h/2,label,size=size)
        constraints.append((item,w-18,h-14))

    def arrow(points,color=BLUE,dashed=False,label=None):
        for start,end in zip(points[:-2],points[1:-1]):
            ax.plot([start[0],end[0]],[start[1],end[1]],color=color,linewidth=1.7,
                    linestyle=(0,(5,3)) if dashed else "solid",zorder=2)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle="-|>",mutation_scale=15,
            color=color,linewidth=1.7,linestyle=(0,(5,3)) if dashed else "solid",zorder=2))
        if label:
            text(*label,size=13,color=color)

    def junction(x,y,color=BLUE):
        ax.add_patch(Circle((x,y),4,color=color,zorder=5))

    def operation(x,y,label):
        ax.add_patch(Circle((x,y),20,facecolor="#d4e1f4",edgecolor=BLUE,linewidth=1.4,zorder=4))
        text(x,y,label,size=23)

    def stack(x,y,w,h):
        d = 12
        ax.add_patch(Polygon([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],facecolor="#aabccb",edgecolor="#5d7387",linewidth=1,zorder=4))
        ax.add_patch(Polygon([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],facecolor="#b7cede",edgecolor="#5d7387",linewidth=1,zorder=4))
        ax.add_patch(Polygon([(x,y),(x+w,y),(x+w,y+h),(x,y+h)],facecolor=COLORS["map"],edgecolor="#5d7387",linewidth=1,zorder=4))
        for offset in (w*.25,w*.55):
            ax.plot([x+offset,x+offset],[y,y+h],color="#7a92a6",lw=.65,zorder=5)

    text(1500,48,"IngredientMassEstimation V5",size=34,weight="bold",color="#163e65")
    text(1500,96,"Una imagen RGB → presencia, gramos por ingrediente y nutrientes",size=21)

    group(20,150,190,650,"Entrada RGB")
    group(240,150,470,650,"Encoder RGB · ConvNeXt Small")
    group(740,150,660,650,"Fusión residual multiescala")
    group(1430,150,550,650,"Query2Label")
    group(2010,150,480,650,"Presencia y cantidad")
    group(2520,150,460,650,"Salidas")
    group(20,850,1380,570,"Profundidad estimada desde el mismo RGB")
    group(1430,850,1000,430,"Cabezal auxiliar de masa total")
    group(2460,850,520,570,"Supervisión auxiliar · training",orange=True)

    # A schematic RGB plate, not a prediction or a modified reference photo.
    ax.add_patch(Ellipse((115,350),150,145,facecolor="white",edgecolor="#b4bbc1",lw=2,zorder=3))
    ax.add_patch(Ellipse((93,361),54,72,angle=25,facecolor="#bb8261",edgecolor="#80573b",zorder=4))
    for cx,cy in ((140,330),(130,310),(146,351),(134,366)):
        ax.add_patch(Circle((cx,cy),12,facecolor="#5b8743",edgecolor="#376435",zorder=4))
    for cx,cy in ((98,313),(108,322),(90,326),(111,341),(97,338),(107,309)):
        ax.add_patch(Ellipse((cx,cy),7,4,angle=25,facecolor="#efe4c8",edgecolor="#dbcaad",zorder=5))
    text(115,235,"320 × 320\nletterbox",size=16)
    text(115,480,"Único input\nexterno",size=16)
    arrow([(192,350),(240,350),(240,424),(285,424)])
    arrow([(115,425),(115,980),(105,980)],color=GREEN)

    # ConvNeXt's four stages, with E3 and E4 feeding the corresponding RGB rows.
    rgb_maps = [(285,300,85,235),(405,330,65,200),(510,366,55,165),(615,400,40,130)]
    for i,(x,y,w,h) in enumerate(rgb_maps):
        stack(x,y,w,h)
        text(x+w/2,578,f"Etapa {i+1}",size=14,weight="bold")
        channels,size = ((96,80),(192,40),(384,20),(768,10))[i]
        text(x+w/2,615,f"{channels}\n{size} × {size}",size=13)
        if i<3:
            nx,ny,nw,nh = rgb_maps[i+1]
            arrow([(x+w+12,y+h/2),(nx,ny+nh/2)])
    text(327,265,"❄ Stem + E1",size=19,color=BLUE)
    text(475,730,"E2–E4: ajuste desde época 4",size=18)
    arrow([(565,366),(565,238),(892,238),(892,300),(915,300)])
    arrow([(667,465),(715,465),(715,600),(915,600)])

    for scale,dy,spatial,tokens in ((3,0,20,400),(4,300,10,100)):
        text(775,245+dy,f"Escala {scale}",size=19,weight="bold",align="left")
        text(820,280+dy,f"RGB\n{384 if scale==3 else 768} × {spatial}²",size=16)
        box(915,273+dy,75,55,"Conv\n1×1","linear",size=16)
        box(1005,273+dy,60,55,"GN","activation",size=16)
        arrow([(990,300+dy),(1005,300+dy)])
        arrow([(1065,300+dy),(1180,300+dy)])
        stack(810,374+dy,40,50)
        text(842,447+dy,f"Depth\n{64 if scale==3 else 128} × {spatial}²",size=15)
        arrow([(862,399+dy),(915,399+dy)],color=GREEN)
        box(915,372+dy,75,55,"Conv\n1×1","linear",size=16)
        box(1005,372+dy,60,55,"GN","activation",size=16)
        arrow([(990,399+dy),(1005,399+dy)],color=GREEN)
        arrow([(1065,399+dy),(1100,399+dy)],color=GREEN)
        operation(1120,399+dy,"×")
        text(1120,345+dy,f"σ(g{scale})",size=17,color=GREEN)
        arrow([(1120,358+dy),(1120,379+dy)],color=GREEN)
        arrow([(1140,399+dy),(1200,399+dy),(1200,320+dy)],color=GREEN)
        operation(1200,300+dy,"+")
        arrow([(1220,300+dy),(1260,300+dy)])
        stack(1260,276+dy,35,50)
        text(1280,363+dy,f"T{scale}\n{tokens} × 256",size=16)
    text(1080,781,"Fₛ = GN(Rₛ) + σ(gₛ) · GN(Dₛ)",size=16)
    arrow([(1307,300),(1350,300),(1350,440)])
    arrow([(1307,600),(1350,600),(1350,480)])
    operation(1350,460,"C")
    text(1280,716,"Flatten\n+ LayerNorm",size=13)
    arrow([(1370,460),(1460,460)])

    stack(1460,365,60,190)
    text(1490,615,"Memoria\n500 × 256",size=13,weight="bold")
    box(1560,220,170,60,"100 consultas\n256 dim (Q)",size=16)
    box(1770,220,180,60,"Posición 2D\n+ escala (K)",size=16)
    box(1570,370,140,105,"Cross-Attn 1\n4 heads\nFFN 1024","attention",size=16)
    box(1780,370,140,105,"Cross-Attn 2\n4 heads\nFFN 1024","attention",size=16)
    arrow([(1645,280),(1645,305),(1610,305),(1610,370)])
    arrow([(1645,280),(1645,305),(1820,305),(1820,370)])
    arrow([(1860,280),(1860,330),(1670,330),(1670,370)])
    arrow([(1860,280),(1860,330),(1880,330),(1880,370)])
    arrow([(1532,460),(1545,460),(1545,510),(1640,510),(1640,475)])
    arrow([(1545,510),(1850,510),(1850,475)])
    junction(1545,510)
    text(1750,540,"Misma memoria K/V en ambas capas",size=14)
    arrow([(1710,423),(1780,423)])
    arrow([(1920,423),(1960,423),(1960,552),(1835,552),(1835,575)])
    box(1750,575,170,70,"Proyección\n256 → 768","linear",size=18)
    arrow([(1835,645),(1835,662),(1750,662),(1750,680)])
    box(1580,680,340,80,"MLP COMPARTIDO\n768 → 384 → 192",size=19)
    text(1750,781,"LayerNorm · GELU · Dropout 0,3",size=14)
    text(1640,620,"Sin Self-Attention",size=13)

    # The shared hidden features feed both ingredient heads, never the grams directly.
    arrow([(1920,720),(1995,720),(1995,260),(2050,260)])
    arrow([(1995,555),(2050,555)])
    junction(1995,555)
    box(2050,225,190,70,"GroupWiseLinear\nPresencia","linear",size=17)
    box(2280,220,180,105,"pᵢ\nsigmoid\n(zᵢ − log wᵢ)",size=18)
    arrow([(2240,260),(2280,260)])
    box(2050,520,190,70,"GroupWiseLinear\nCantidad","linear",size=17)
    box(2280,510,180,105,"cᵢ\nsoftplus(aᵢ)\n× 300 g",size=18)
    arrow([(2240,555),(2280,555)])
    arrow([(2370,325),(2370,355),(2225,355),(2225,400)])
    arrow([(2370,510),(2370,485),(2225,485),(2225,440)])
    operation(2225,420,"×")
    arrow([(2225,440),(2225,650)])
    box(2090,650,360,105,"GRAMOS POR INGREDIENTE\nmᵢ = pᵢ × cᵢ\n100 clases",size=20)
    arrow([(2450,702),(2505,702),(2505,280),(2570,280)])
    arrow([(2505,480),(2580,480)])
    junction(2505,480)
    operation(2590,280,"Σ")
    arrow([(2610,280),(2650,280)])
    box(2650,235,290,90,"Masa total\nM = Σᵢ mᵢ",size=21)
    box(2580,430,350,100,"Matriz nutricional de train\n100 × 4 · N = m · D",size=19)
    for i,label in enumerate(("Calorías","Grasas","Carbohidratos","Proteínas")):
        y = 570+i*50
        box(2640,y,250,38,label,kind=("linear" if i==0 else "activation"),size=18)
        if i==0:
            arrow([(2755,530),(2755,570)])
        else:
            arrow([(2930,480),(2945,480),(2945,y+19),(2890,y+19)])

    # Relative-depth predictor outside the trainable network; maps are cached.
    box(105,930,270,150,"❄ Depth Anything\nV2 Small\nCONGELADO",kind="attention",size=21)
    arrow([(375,1005),(420,1005)],color=GREEN)
    box(420,930,140,150,"Mapa\nrelativo\n320 × 320",size=18)
    arrow([(560,1005),(620,1005),(620,1045),(650,1045)],color=GREEN)
    text(245,1170,"Solo RGB · sin depth de sensor\nProfundidad relativa, no centímetros",size=17)
    text(980,935,"SmallDepthEncoder (entrenable)",size=21,weight="bold")
    for i,(cin,cout) in enumerate(((1,16),(16,32),(32,64),(64,64),(64,128))):
        x = 650+i*125
        box(x,1005,90,80,"Conv\nGN · GELU","linear",size=15)
        text(x+45,1130,f"{cin} → {cout}",size=17)
        if i<4:
            arrow([(x+90,1045),(x+125,1045)],color=GREEN)
    text(950,1220,"Conv 3×3 · stride 2 · padding 1\nGroupNorm(8) · GELU",size=17)
    stack(1280,945,40,60)
    text(1300,918,"64 × 20²",size=15)
    arrow([(1115,1045),(1132,1045),(1132,975),(1280,975)],color=GREEN)
    junction(1132,1045,GREEN)
    stack(1280,1165,40,50)
    text(1300,1250,"128 × 10²",size=15)
    arrow([(1240,1045),(1260,1045),(1260,1190),(1280,1190)],color=GREEN)
    arrow([(1332,975),(1382,975),(1382,820),(770,820),(770,399),(810,399)],color=GREEN)
    arrow([(1332,1190),(1370,1190),(1370,838),(795,838),(795,699),(810,699)],color=GREEN)

    # Auxiliary forward reads the exact same visual-memory values, not PE/queries/grams.
    arrow([(1545,510),(1545,825),(1530,825),(1530,985)])
    box(1460,985,140,90,"Mean Pool\n500 tokens","activation",size=17)
    box(1630,985,120,90,"Layer\nNorm","activation",size=17)
    box(1780,985,190,90,"MLP\n256 → 128 → 1","linear",size=17)
    box(2000,985,160,90,"Softplus\n× μ_train","activation",size=18)
    box(2190,985,190,90,"M_aux",kind="attention",size=23)
    for left,right in ((1600,1630),(1750,1780),(1970,2000),(2160,2190)):
        arrow([(left,1030),(right,1030)])
    text(1900,1140,"GELU · Dropout 0,3 · escala e inicialización: media de train",size=17)
    text(1920,1205,"SALIDA AUXILIAR: NO REESCALA LOS GRAMOS",size=20,weight="bold")

    # Training-only auxiliary terms: real mass to L_aux; predicted summed mass to L_cons.
    box(2530,930,360,80,"Masa real de train · M_real",kind="loss",size=18)
    box(2510,1070,400,105,"L_aux = |M_aux − M_real| / μ_train\nPeso 0,25",kind="loss",size=18)
    box(2510,1270,400,105,"L_cons = |M_aux − M| / μ_train\nPeso 0,10",kind="loss",size=18)
    arrow([(2710,1010),(2710,1070)],color=ORANGE,dashed=True)
    arrow([(2380,1030),(2445,1030),(2445,1122),(2510,1122)],color=ORANGE,dashed=True)
    arrow([(2445,1122),(2445,1322),(2510,1322)],color=ORANGE,dashed=True)
    junction(2445,1122,ORANGE)
    arrow([(2940,280),(2960,280),(2960,1215),(2710,1215),(2710,1270)],color=ORANGE,dashed=True)
    text(1860,1350,"Se mantienen BCE y pérdidas de gramos, masa y nutrientes de V3",size=18)
    text(1860,1400,"μ_train: solo train · 300 g es escala, no límite · sin segmentación",size=17)

    group(20,1500,2960,150,"Leyenda")
    text(160,1590,"❄ Congelado",size=18,color=BLUE)
    box(350,1570,150,40,"Conv / Linear","linear",size=15)
    box(520,1570,180,40,"Activación / Pool","activation",size=15)
    arrow([(750,1590),(825,1590)])
    text(930,1590,"Forward RGB / común",size=17)
    arrow([(1160,1590),(1235,1590)],color=GREEN)
    text(1320,1590,"Depth",size=17)
    arrow([(1450,1590),(1525,1590)],color=ORANGE,dashed=True)
    text(1690,1590,"Supervisión auxiliar",size=17)
    operation(1920,1590,"+"); text(2010,1590,"Suma",size=17)
    operation(2170,1590,"×"); text(2280,1590,"Producto",size=17)
    operation(2450,1590,"C"); text(2620,1590,"Concat. de tokens",size=17)
    text(1500,1680,"La ilustración del plato es esquemática. Flechas sin punto de unión no comparten señal.",size=15,color="#506272")
    fig.canvas.draw()
    for artist,max_width,max_height in constraints:
        renderer = fig.canvas.get_renderer()
        while artist.get_fontsize() > 10:
            bounds = artist.get_window_extent(renderer)
            if bounds.width <= max_width and bounds.height <= max_height:
                break
            artist.set_fontsize(artist.get_fontsize()-.5)
        bounds = artist.get_window_extent(renderer)
        if bounds.width > max_width+1 or bounds.height > max_height+1:
            raise RuntimeError("Label does not fit its panel: " + artist.get_text())
    fig.canvas.draw()
    # All labels must stay on canvas; useful for preserving a printable figure.
    renderer = fig.canvas.get_renderer()
    for artist in artists:
        bounds = artist.get_window_extent(renderer)
        if bounds.x0 < -2 or bounds.y0 < -2 or bounds.x1 > fig.bbox.width+2 or bounds.y1 > fig.bbox.height+2:
            raise RuntimeError("Label outside canvas: " + artist.get_text())
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"figures")
    args = parser.parse_args()
    files = [args.output_dir/f"arquitectura_query2label_v5_final.{suffix}" for suffix in ("svg","png","pdf")]
    if any(path.exists() for path in files):
        raise FileExistsError("Use a new output directory to preserve an existing figure")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    fig = build()
    for path in files:
        fig.savefig(path,dpi=160 if path.suffix==".png" else 100)
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__": main()
