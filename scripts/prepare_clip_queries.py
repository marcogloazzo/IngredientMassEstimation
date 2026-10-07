"""Prepare CLIP TEXT vectors once; no food images, depth inference or training."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from ingredient_mass.clip_queries import encode_text_queries,read_clip_queries
from ingredient_mass.depth import file_sha256
from ingredient_mass.features import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"config/query2label_100_v6.json")
    parser.add_argument("--vocabulary",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--clip-cache",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda","auto"),default="cpu")
    parser.add_argument("--check-cache",action="store_true",help="Validate only; no CLIP imports/downloads")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    vocabulary = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    digest = file_sha256(args.vocabulary)
    if args.check_cache or args.output.is_file():
        _,mask,metadata = read_clip_queries(args.output,vocabulary,config,digest)
        print(f"Verified CLIP cache: {mask.sum()} explicit ingredients; other remains trainable/random")
        print(json.dumps(metadata,indent=2)); return
    if len(vocabulary["entries"]) != int(config["top_k"]):
        raise ValueError("Vocabulary/config top_k mismatch")
    if {"v2","v3","v4","v5"} & {p.lower() for p in args.output.resolve().parts}:
        raise ValueError("Write the new CLIP cache in v6, never a historical results namespace")
    try:
        import clip
    except ImportError as exc:
        raise ImportError("Install requirements-v6.txt for the one-time CLIP preparation") from exc
    from importlib import metadata as package_metadata
    direct_url = package_metadata.distribution("clip").read_text("direct_url.json")
    installed_revision = json.loads(direct_url or "{}").get("vcs_info",{}).get("commit_id")
    if installed_revision != config["clip_queries"]["repo_revision"]:
        raise RuntimeError("CLIP source revision differs: reinstall the pinned requirements-v6.txt")
    device = resolve_device(args.device)
    embeddings,mask,texts = encode_text_queries(vocabulary,config,clip,device,args.clip_cache)
    metadata = {"schema_version":1,**{k:config["clip_queries"][k] for k in ("model","repo_revision","template")},
                "vocabulary_sha256":digest,"explicit_classes":int(mask.sum()),"other_policy":"random_trainable",
                "embedding_dim":int(embeddings.shape[1]),"text_prompts":texts,
                "note":"Text semantics only; trainable projection aligns to ConvNeXt memory during training"}
    weights_file = args.clip_cache/"ViT-B-32.pt"
    if weights_file.is_file(): metadata["weights_sha256"] = file_sha256(weights_file)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive write: never overwrite a previous initialization cache.
    with args.output.open("xb") as stream:
        np.savez_compressed(stream,text_embeddings=embeddings,initialized_mask=mask,
            vocabulary_keys=np.asarray([e["key"] for e in vocabulary["entries"]]),metadata=np.asarray(json.dumps(metadata)))
    read_clip_queries(args.output,vocabulary,config,digest)
    print(f"CLIP query initialization saved: {args.output}")


if __name__ == "__main__": main()
