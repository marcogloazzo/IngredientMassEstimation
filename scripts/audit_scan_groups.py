"""Build candidate groups and an offline, portable visual review. No training."""
import argparse
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingredient_mass.data import read_jsonl, write_json
from ingredient_mass.scan_groups import (connected_groups, detect_links, file_digest, fingerprint,
                                        image_evidence, load_inventory, new_output_directory,
                                        verify_audit, verify_sources)


def review_html(audit, dataset, output, page_size=30):
    assets = output / "assets"
    assets.mkdir()
    lookup = {node["dish_id"]: node for node in audit["nodes"]}
    for node in audit["nodes"]:
        if node["image"]:
            with Image.open(dataset / node["image"]) as source:
                preview = source.convert("RGB")
                preview.thumbnail((240, 180))
                preview.save(assets / (node["dish_id"] + ".jpg"), quality=75)
    baseline = audit["baseline_splits"]
    active = [group for group in audit["groups"] if group["used_rgb_members"]]
    active.sort(key=lambda group: (
        0 if {baseline.get(key) for key in group["members"]} >= {"train", "validation"} else 1,
        -len(group["members"]), group["group_id"]))
    pages = [active[i:i + page_size] for i in range(0, len(active), page_size)]
    links_by_group = {}
    group_by_member = {key: group["group_id"] for group in audit["groups"] for key in group["members"]}
    for link in audit["links"]:
        if not link.get("excluded_from_grouping"):
            links_by_group.setdefault(group_by_member[link["before"]], []).append(link)
    sorted_nodes = sorted(audit["nodes"], key=lambda node: (node["cafeteria"], node["timestamp"]))
    positions = {node["dish_id"]: index for index, node in enumerate(sorted_nodes)}

    def image_card(identifier, brief=False):
        node = lookup[identifier]
        picture = f'<a href="assets/{identifier}.jpg" target="_blank"><img src="assets/{identifier}.jpg" loading="lazy" alt="{identifier}"></a>' if node["image"] else '<p>Sin RGB; revisar metadata/continuidad.</p>'
        ingredients = "; ".join(f"{c['name']}: {c['grams']:.2f} g" for c in node["ingredients"])
        return (f'<article><strong>{identifier}</strong>{picture}<p>{node["total_mass"]:.1f} g · {escape(node["official_split"])} oficial'
                f' · anterior: {escape(baseline.get(identifier, "no disponible"))}</p>' +
                (f'<details><summary>Ingredientes y pesos</summary><p>{escape(ingredients)}</p></details>' if not brief else '') + '</article>')

    css = """body{font:15px system-ui;color:#18324b;background:#f4f7fb;max-width:1250px;margin:auto;padding:24px}
    button,textarea{font:inherit}button{padding:9px 14px;margin:4px;border:1px solid #aac1d8;border-radius:14px;cursor:pointer}
    .toolbar{position:sticky;top:0;background:#f4f7fb;padding:10px;border-bottom:1px solid #aac1d8;z-index:2}
    .group{background:white;border:1px solid #d2dfec;border-radius:18px;padding:18px;margin:18px 0}
    .photos{display:flex;gap:12px;flex-wrap:wrap}.photos article{width:240px;padding:10px;background:#eff5fc;border-radius:14px}
    img{width:100%;height:auto}.warning{color:#823a12}textarea{width:95%;min-height:65px}
    .state{font-weight:bold}.done{border-left:8px solid #25745c}p{line-height:1.5;overflow-wrap:anywhere}
    """
    index_links = []
    for page_number, page in enumerate(pages, 1):
        filename = f"review_{page_number:03d}.html"
        index_links.append(f'<li><a href="{filename}">Página {page_number}: {len(page)} grupos</a></li>')
        cards = []
        for group in page:
            members = group["members"]
            neighbors = set()
            for identifier in (members[0], members[-1]):
                pos = positions[identifier]
                for other in sorted_nodes[max(0, pos - 2):pos + 3]:
                    if other["cafeteria"] == lookup[identifier]["cafeteria"] and other["dish_id"] not in members:
                        neighbors.add(other["dish_id"])
            duration = lookup[members[-1]]["timestamp"] - lookup[members[0]]["timestamp"]
            warning = "GRUPO GRANDE: revisar posibles uniones falsas." if len(members) > 12 else "Candidato, no secuencia confirmada."
            evidence = links_by_group.get(group["group_id"], [])
            text = "\n".join(f"{link['before']} → {link['after']}: {link['seconds']} s; +{link['mass_increment']:.1f} g; "
                             f"sin cambio {link['unchanged_classes']}/{link['shared_classes']} clases; "
                             f"diferencia RGB central {link.get('central_rgb_mae_255', 'sin foto')}" for link in evidence)
            cards.append(f'''<section class="group" id="{group['group_id']}">
            <h2>{group['group_id']} · {len(members)} escaneos · {duration} s</h2>
            <p class="warning">{warning} La diferencia RGB NO es un puntaje de parentesco.</p>
            <p class="state">Pendiente</p><div class="photos">{''.join(image_card(key) for key in members)}</div>
            <details><summary>Vecinos temporales: buscar continuaciones omitidas</summary><div class="photos">{''.join(image_card(key, True) for key in sorted(neighbors))}</div></details>
            <details><summary>Enlaces y evidencia numérica</summary><pre>{escape(text)}</pre></details>
            <p>Particiones: una secuencia por línea; IDs separados por coma. Deben aparecer TODOS los IDs del grupo una sola vez.</p>
            <textarea class="parts" aria-label="Particiones del grupo">{escape(','.join(members))}</textarea>
            <textarea class="note" aria-label="Fundamento de revisión" placeholder="Fundamento: alimentos/disposición conservados, inicio de otro plato, etc."></textarea>
            <button data-action="keep">Mantener juntos</button><button data-action="custom">Guardar particiones editadas</button>
            <button data-action="reset">Dejar pendiente</button></section>''')
        payload = {group["group_id"]: group["members"] for group in page}
        js = r"""
        const members=MEMBERS, auditId=AUDIT, page=PAGE;
        const storageKey='nutrition5k-review-'+auditId+'-'+page;
        let decisions={}; let merges=[];
        function save(){try{localStorage.setItem(storageKey,JSON.stringify({decisions,merges}));}catch(e){document.querySelector('#storage').textContent='No se pudo guardar en el navegador: exportá antes de cerrar.';}}
        function refresh(){let n=0;for(const key of Object.keys(members)){const el=document.getElementById(key),d=decisions[key];
          el.classList.toggle('done',!!d);el.querySelector('.state').textContent=d?'Revisado ('+d.partitions.length+' secuencias)':'Pendiente';
          if(d){n++;el.querySelector('.parts').value=d.partitions.map(p=>p.join(',')).join('\n');el.querySelector('.note').value=d.note;}}
          document.querySelector('#count').textContent=n+'/'+Object.keys(members).length+' grupos revisados';}
        try{const old=JSON.parse(localStorage.getItem(storageKey)||'{}');decisions=old.decisions||{};merges=old.merges||[];}catch(e){}
        document.querySelector('#merges').value=merges.map(m=>m.members.join(',')+' | '+m.note).join('\n');
        document.querySelectorAll('.group button').forEach(button=>button.onclick=()=>{const el=button.closest('.group'),key=el.id;
          if(button.dataset.action==='reset'){delete decisions[key];save();refresh();return;}
          const note=el.querySelector('.note').value.trim();if(!note){alert('Escribí el fundamento de tu revisión.');return;}
          const parts=button.dataset.action==='keep'?[members[key]]:el.querySelector('.parts').value.trim().split(/\n+/).map(s=>s.split(',').map(x=>x.trim()).filter(Boolean));
          const flat=parts.flat();if(parts.some(p=>!p.length)||new Set(flat).size!==flat.length||JSON.stringify([...flat].sort())!==JSON.stringify([...members[key]].sort())){alert('Cada ID debe aparecer exactamente una vez.');return;}
          decisions[key]={status:'reviewed',partitions:parts,note};save();refresh();});
        document.querySelector('#export').onclick=()=>{merges=[];const text=document.querySelector('#merges').value.trim();
          if(text)for(const row of text.split(/\n+/)){const pos=row.indexOf('|');if(pos<0){alert('Fusión: IDs separados por coma | fundamento');return;}
            const ids=row.slice(0,pos).split(',').map(s=>s.trim()).filter(Boolean),note=row.slice(pos+1).trim();if(ids.length<2||!note){alert('La fusión necesita al menos dos IDs y fundamento.');return;}merges.push({members:ids,note});}
          save();const data={schema_version:1,audit_id:auditId,decisions,merges};const a=document.createElement('a');const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));
          a.href=url;a.download='review_'+String(page).padStart(3,'0')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};refresh();
        """.replace("MEMBERS", json.dumps(payload).replace("<", "\\u003c")).replace("AUDIT", json.dumps(audit["audit_id"])).replace("PAGE", str(page_number))
        body = (f'<h1>Revisión de secuencias · página {page_number}</h1><p>Manifest: {audit["audit_id"]}. '
                'Revisar continuidad de comida, no solo plato/fondo. No marcar un singleton como independiente sin mirar vecinos. '
                'Las fronteras oficiales train/test nunca se fusionan automáticamente.</p>'
                '<div class="toolbar"><a href="index.html">Índice</a> · <span id="count"></span> '
                '<button id="export">Exportar revisión JSON</button><span id="storage"></span></div>' + ''.join(cards) +
                '<h2>Fusiones entre grupos si encontraste un enlace omitido</h2><p>Una por línea: ID de un escaneo de cada secuencia a unir | fundamento. '
                'No permite mezclar train/test oficial. Exportar nuevamente esta página después de editar.</p><textarea id="merges" aria-label="Fusiones manuales"></textarea>')
        (output / filename).write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                      f'<title>Revisión {page_number}</title><style>{css}</style><body>{body}<script>{js}</script></body></html>', encoding="utf-8")
    (output / "index.html").write_text('<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                                     f'<title>Auditoría de escaneos</title><style>{css}</style><body><h1>Secuencias candidatas Nutrition5k</h1>'
                                     '<p>El agrupamiento es HEURÍSTICO. Revisar también singleton/vecinos y grupos grandes. '
                                     'Exportar los JSON de TODAS las páginas; la preparación revisada rechaza decisiones pendientes.</p>'
                                     f'<p>{len(active)} grupos con RGB utilizado · {len(pages)} páginas · audit_id {audit["audit_id"]}</p>'
                                     '<p>Exportar avances: el almacenamiento del navegador puede no persistir entre equipos o rutas.</p><ol>' + ''.join(index_links) + '</ol></body></html>', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/raw/nutrition5k_dataset")
    parser.add_argument("--baseline-records", type=Path, default=ROOT / "data/processed_100/records.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/scan_group_audit")
    parser.add_argument("--max-gap-seconds", type=int, default=600)
    parser.add_argument("--gram-tolerance", type=float, default=.1)
    parser.add_argument("--page-size", type=int, default=30)
    parser.add_argument("--render-existing-audit", type=Path, help="Rebuild HTML in a NEW output directory using an existing verified manifest")
    args = parser.parse_args()
    if args.page_size < 1:
        parser.error("page-size must be positive")
    root = args.dataset_root.resolve()
    if args.render_existing_audit:
        audit = json.loads(args.render_existing_audit.read_text(encoding="utf-8"))
        verify_audit(audit)
        verify_sources(audit, root)
        output = new_output_directory(args.output_dir, protected=(root, args.render_existing_audit.parent))
        review_html(audit, root, output, args.page_size)
        print(f"Open: {output / 'index.html'}")
        return
    print("Reading full metadata and official splits...", flush=True)
    nodes, sources = load_inventory(root)
    links = detect_links(nodes, args.max_gap_seconds, args.gram_tolerance)
    groups = connected_groups(nodes, links)
    baseline = {}
    if args.baseline_records.is_file():
        rows = read_jsonl(args.baseline_records)
        baseline = {row["dish_id"]: row["split"] for row in rows}
        if len(baseline) != len(rows):
            raise ValueError("Baseline records contain duplicate IDs.")
    nodes, links = image_evidence(root, nodes, links)
    audit = {"schema_version": 1, "method": "metadata_monotonic_candidates_with_official_boundaries",
             "parameters": {"max_gap_seconds": args.max_gap_seconds, "gram_tolerance": args.gram_tolerance},
             "source_sha256": sources, "baseline_splits": baseline,
             "baseline_records_sha256": file_digest(args.baseline_records) if args.baseline_records.is_file() else None,
             "nodes": nodes, "links": links, "groups": groups,
             "limitations": ["Not official group IDs; heuristics can miss links and merge different physical plates.",
                             "RGB difference is descriptive only, not an automatic identity decision.",
                             "Official train/test boundaries override heuristic compatibility; inspect excluded links separately."]}
    audit["audit_id"] = fingerprint(audit)
    summary = {"metadata_scans": len(nodes), "candidate_links": len(links), "candidate_groups": len(groups),
               "groups_with_used_rgb": sum(g["used_rgb_members"] > 0 for g in groups),
               "largest_candidate_group": max(len(g["members"]) for g in groups),
               "excluded_official_boundary_links": sum(bool(link.get("excluded_from_grouping")) for link in links),
               "baseline_cross_train_validation_pairs": sum({baseline.get(link['before']), baseline.get(link['after'])} == {"train", "validation"} for link in links),
               "baseline_mixed_candidate_groups": sum({baseline.get(key) for key in g['members']} >= {"train", "validation"} for g in groups),
               "used_rgb_samples": dict(Counter(n["official_split"] for n in nodes if n["image"] and n["official_split"] in ("train", "test"))),
               "audit_id": audit["audit_id"], "certified_independence": False}
    output = new_output_directory(args.output_dir, protected=(root, args.baseline_records.parent))
    write_json(output / "audit_manifest.json", audit)
    write_json(output / "audit_summary.json", summary)
    print("Building offline visual review...", flush=True)
    review_html(audit, root, output, args.page_size)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Open: {output / 'index.html'}")


if __name__ == "__main__":
    main()
