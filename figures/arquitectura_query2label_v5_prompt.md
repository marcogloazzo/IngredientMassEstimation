# Figura V5 — especificación y prompt

Referencia visual: `OmniFood8K/Code/source/network7.png`, solo para el estilo.
Arquitectura verificada en `config/query2label_100_v5.json`, `spatial_model.py`,
`conditional_model.py`, `multiscale_model.py` y `specialized_model.py`.
Exploración visual: herramienta integrada de imágenes, sin entrenar ni ejecutar
el modelo. Sus borradores ayudaron a definir el estilo, pero no son la figura final.
La figura final se dibujó de forma vectorial con `scripts/draw_architecture_v5.py`
para controlar exactamente los conectores y exportar el mismo esquema a PNG/SVG.
Archivos seleccionados: `arquitectura_query2label_v5_final.png` y
`arquitectura_query2label_v5_final.svg`.

## Prompt

Use case: scientific-educational / infographic-diagram.
Asset: a precise research-paper neural-network architecture figure for the implemented IngredientMassEstimation V5 model, in Spanish.
Input image 1 is STYLE REFERENCE ONLY, NOT the architecture to copy. Create an entirely new diagram, keeping its visual language: white background, thin blue arrows, blue dashed rounded group borders, pale blue panels, pastel pink convolution/linear blocks, pastel green activation/pooling blocks, small isometric feature-map stacks, circle operators, bold clean sans-serif headings. Landscape, high resolution, approximately 16:9. Large readable labels, carefully routed unambiguous arrows. No clipping, no watermark.
Title exactly: "IngredientMassEstimation V5".
Subtitle exactly: "Estimación de ingredientes, masa y nutrientes desde una única imagen RGB".

SCIENTIFIC INVARIANTS: This model has exactly ONE external input: RGB. Depth is estimated FROM THE SAME RGB, not a second sensor input. Exactly ONE shared MLP (NOT separate task MLPs). Exactly TWO sequential cross-attention layers, no self-attention. Nutrients are a fixed matrix multiplication of ingredient grams, NOT learned nutritional heads. The auxiliary mass head bypasses Query2Label and pools the visual memory; it does NOT rescale ingredient grams or replace their summed mass. NO Swin, FFT, segmentation, masks, SSRA, LoRA or CLIP.

Please allocate the layout in three clear horizontal bands:
TOP/main band: RGB extractor → multiscale fusion → Query2Label and shared MLP → ingredient heads and outputs.
MIDDLE/lower left: Depth Anything and depth CNN feeding the two fusion scales.
BOTTOM: an auxiliary mass branch directly from visual memory, plus an orange dashed training-only supervision panel and a small legend.

Exact architecture and labels, preserve all connections:
1. FAR LEFT: one small food photograph thumbnail, label "Entrada RGB", "320 × 320 · letterbox". Split this SAME image into ConvNeXt and Depth Anything paths. Only one input thumbnail.
2. Group "Encoder RGB · ConvNeXt Small": four connected decreasing feature-map stacks, labels "Etapa 1", "Etapa 2", "Etapa 3", "Etapa 4". Snowflake on Etapa 1 only, small text "Stem + E1 congelados". Etapa 3 output "384 × 20 × 20" and Etapa 4 output "768 × 10 × 10". Clearly tap BOTH E3 and E4 into fusion. Text below "E2–E4 ajustables desde época 4". Stem/E1 frozen; remaining stages warm up frozen only at first.
3. Lower left group "Profundidad estimada desde RGB": box "Depth Anything V2 Small" with snowflake, followed by a relative-depth heatmap thumbnail labelled "Depth relativo · 320 × 320", then trainable "SmallDepthEncoder". Draw its five pink Conv + green GELU blocks with channel labels "1 → 16 → 32 → 64 → 64 → 128", small caption "Conv 3×3, stride 2 · GroupNorm · GELU". Tap the penultimate depth output "64 × 20 × 20" and final output "128 × 10 × 10" into their CORRESPONDING fusion rows. Frozen predictor is preprocessing/cache; depth CNN is trainable. Tiny caption "Sin depth de sensor · sin escala métrica".
4. Group "Fusión residual multiescala": two parallel rows, "Escala 3" and "Escala 4". Each RGB branch and matching depth branch uses a pink "Conv 1×1" then "GN" to project to 256 channels. In EACH depth branch a separate learned scalar sigmoid gate "σ(g3)" or "σ(g4)" and circle ×, followed by circle + adding normalized RGB. NO concatenation of RGB/depth at this stage. Text formula under rows "F_s = GN(R_s) + σ(g_s) · GN(D_s)". Each resulting feature map is flattened and LayerNormed: "T3: 400 × 256", "T4: 100 × 256". Concatenate these token sequences through circle C, into a stack labelled "Memoria visual · 500 × 256".
5. Group "Query2Label": learned "100 consultas · 256 dim" enter two sequential blue blocks "Cross-Attention 1" and "Cross-Attention 2". BOTH attend to the same visual memory. A small separate positional block "Posición 2D + embedding de escala" feeds attention, not auxiliary head. Caption "4 heads · FFN 1024 · residual + LayerNorm" and "Sin Self-Attention". Output label "100 × 256"; then pink block "Proyección 256 → 768"; then exactly ONE shared MLP group labelled "MLP compartido", "768 → 384 → 192", small "LayerNorm · GELU · Dropout 0,3". Both heads emerge from this same MLP.
6. Group "Presencia y cantidad": upper pink "GroupWiseLinear" to "Presencia p_i" with formula "p_i = sigmoid(z_i − log w_i)". Lower pink "GroupWiseLinear" to "Cantidad condicional c_i" with formula "c_i = softplus(a_i) × 300 g". Draw p and c both entering one circle ×. Output vector "Gramos por ingrediente", "m_i = p_i × c_i", "100 clases".
7. Group "Salidas": fork the ingredient-gram vector to:
(a) a sum operator Σ and box "Masa total", "M = Σ_i m_i";
(b) a fixed box "Matriz nutricional de train", "100 × 4", then four labelled small output bars "Calorías", "Grasas", "Carbohidratos", "Proteínas". Explicit formula "N = m · D". These four output bars are not learned heads.
8. LOWER branch "Cabezal auxiliar de masa total": a distinct arrow FROM "Memoria visual · 500 × 256", not from the MLP or grams, to green "Mean Pool" (over 500 spatial tokens), then "LayerNorm", pink "MLP 256 → 128 → 1", green "Softplus × μ_train", then box "M_aux". Caption "GELU · Dropout 0,3" and IMPORTANT exactly "Salida auxiliar: NO reescala los gramos".
9. BOTTOM RIGHT orange dashed group "Supervisión auxiliar · solo entrenamiento": orange dashed arrows from M_aux and main M and small "Masa real de train" label, to TWO distinct compact orange boxes:
"L_aux = |M_aux − M_real| / μ_train" with "Peso 0,25".
"L_cons = |M_aux − M| / μ_train" with "Peso 0,10".
No solid arrow from auxiliary mass to ingredient outputs. Text below "Se mantienen BCE y pérdidas de gramos / nutrientes de V3".
10. Small compact legend: snowflake "Congelado"; pink/green/blue learned blocks "Entrenable"; solid blue arrow "Forward"; orange dashed arrow "Supervisión"; circles "+" "Suma", "×" "Producto", "C" "Concatenación". Tiny footer "Depth relativo ≠ volumen métrico · 300 g es escala, no límite".

Prioritize unambiguous correct graph and readable Spanish lettering. All panels should be balanced and spaced, resembling a polished journal figure, not a website or a decorative slide. The supplied image is only a style guide; do NOT import its incorrect modules or topology.

## Corrección de conectores

Edit this architecture figure. Keep the excellent typography, exact labels, colors, all blocks, dimensions, title, photographs and overall composition unchanged. The ONLY change is to repair the data-flow connector lines so the scientific graph is correct. Use thin clearly visible blue solid arrows for forward and orange dashed arrows for losses. Avoid floating arrows, unintended intersections and lines ending on borders. Slightly move blocks if needed to create clear routing lanes.

MANDATORY connector corrections:
1. The existing shared MLP box ("MLP compartido 768 → 384 → 192") must output to BOTH "GroupWiseLinear" boxes under "Presencia y cantidad". Connect its right edge via a routed blue bus to the existing top fork above the two GroupWiseLinear boxes. The top fork must not float. Remove the current erroneous shortcut from MLP to "Gramos por ingrediente": the ONLY input to that grams box is the downward arrow from the × circle combining p_i and c_i.
2. Correct ConvNeXt taps. "Etapa 3" → ONLY the upper RGB projection labelled "384 × 20 × 20" inside Escala 3. "Etapa 4" → ONLY the lower RGB projection labelled "768 × 10 × 10" inside Escala 4. The current upper RGB line incorrectly starts from Etapa 4; redraw it from Etapa 3. No RGB line may feed any Depth branch. Keep normal Etapa1→Etapa2→Etapa3→Etapa4 sequence.
3. Connect the actual depth CNN penultimate output 64 channels to the Escala 3 DEPTH projection (64 × 20 × 20). Connect its final 128-channel output to the Escala 4 DEPTH projection (128 × 10 × 10). Depth arrows must terminate at those depth feature stacks / depth Conv1×1, never the fusion group border or RGB lines.
4. The "Memoria visual 500 × 256" must feed BOTH Cross-Attention1 and Cross-Attention2 as a shared memory bus. A separate sequential arrow connects Cross-Attention1 output to Cross-Attention2. Positional input and label queries remain separate inputs.
5. The blue arrow that descends from "Memoria visual 500 × 256" to the auxiliary panel must continue visibly ALL THE WAY to "Mean Pool (500)"; presently the two lines are disconnected. This auxiliary input is visual memory, not MLP features, conditional grams or outputs. No positional features feed auxiliary pool.
6. Make all normal prediction flow arrows explicit: Cross-Attention2 → label features → Proyección256→768 → sharedMLP → the two ingredient heads → p_i,c_i → × → grams. Grams feed BOTH Σ→Masa total and Matriz nutricional→four nutrients.
7. Training panel: M_aux connects by ORANGE DASHED branch to BOTH L_aux and L_cons. "Masa real de train M_real" connects ONLY to L_aux, NOT L_cons. Instead add an orange dashed connector from the main "Masa total M=Σ_i m_i" output to L_cons. L_cons uses M_aux and main predicted M, not M_real. Do not send any auxiliary-head arrow back into main grams or nutrient predictions.
8. Small accuracy label: between both fused feature stacks T3/T4 and circle C add a compact common caption "Flatten + LayerNorm". Keep C=concatenation of the 400 and 100 tokens.
Do not add Self-Attention, separate task MLPs, depth sensor input, segmentation, FFT, Swin, or nutrient learned heads. Keep note "Salida auxiliar: NO reescala los gramos" unchanged. No extra arrows to titles. This is scientific connector correction, not a redesign.
