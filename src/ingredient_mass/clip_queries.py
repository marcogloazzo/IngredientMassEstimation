"""Frozen CLIP text initialization cache. CLIP is never needed by inference."""
import json
from pathlib import Path

import numpy as np


def read_clip_queries(path, vocabulary, config, vocabulary_sha256=None):
    if path is None or not Path(path).is_file():
        raise FileNotFoundError("V6 requires --clip-queries; prepare_clip_queries.py does not train")
    with np.load(path,allow_pickle=False) as source:
        if not {"text_embeddings","initialized_mask","vocabulary_keys","metadata"} <= set(source.files):
            raise ValueError("Incomplete CLIP query cache")
        embeddings = source["text_embeddings"].copy()
        mask = source["initialized_mask"].copy()
        keys = source["vocabulary_keys"].astype(str).tolist()
        metadata = json.loads(str(source["metadata"].item()))
    expected_keys = [entry["key"] for entry in vocabulary["entries"]]
    settings = config["clip_queries"]
    if keys != expected_keys or mask.dtype != np.bool_ or mask.shape != (len(keys),):
        raise ValueError("CLIP cache vocabulary/mask differ from this experiment")
    expected_mask = np.asarray([key != settings["other_key"] for key in keys],dtype=bool)
    if not np.array_equal(mask,expected_mask):
        raise ValueError("Only explicit ingredients may receive CLIP initialization; keep other learned")
    if embeddings.shape != (len(keys),int(settings.get("embedding_dim",512))) or embeddings.dtype != np.float32:
        raise ValueError("CLIP embedding shape/dtype mismatch")
    if not np.isfinite(embeddings).all() or not mask.any():
        raise ValueError("Nonfinite/empty CLIP initialization")
    if not np.allclose(np.linalg.norm(embeddings[mask],axis=1),1.,atol=1e-4):
        raise ValueError("CLIP text vectors must have unit norm")
    for key in ("model","repo_revision","template"):
        if metadata.get(key) != settings[key]:
            raise ValueError(f"CLIP cache metadata differs at {key}")
    if metadata.get("schema_version") != 1:
        raise ValueError("Unsupported CLIP cache schema")
    if vocabulary_sha256 and metadata.get("vocabulary_sha256") != vocabulary_sha256:
        raise ValueError("CLIP initialization vocabulary fingerprint differs")
    return embeddings,mask,metadata


def encode_text_queries(vocabulary, config, clip_module, device, download_root):
    import torch
    from torch.nn import functional as F
    settings = config["clip_queries"]
    entries = vocabulary["entries"]
    mask = np.asarray([e["key"] != settings["other_key"] for e in entries],dtype=bool)
    if not mask.any():
        raise ValueError("No explicit ingredient names for CLIP")
    texts = [settings["template"].format(name=e["name"]) for e,use in zip(entries,mask) if use]
    model,_ = clip_module.load(settings["model"],device=device,jit=False,download_root=str(download_root))
    model.eval().requires_grad_(False)
    with torch.inference_mode():
        tokens = clip_module.tokenize(texts,truncate=True).to(device)
        features = model.encode_text(tokens).float()
        if not torch.isfinite(features).all() or (features.norm(dim=-1) <= 1e-8).any():
            raise ValueError("CLIP produced invalid text features")
        features = F.normalize(features,dim=-1).cpu().numpy()
    expected_dimension = int(settings.get("embedding_dim",512))
    if features.shape != (int(mask.sum()),expected_dimension):
        raise ValueError("Configured CLIP embedding dimension differs from text encoder")
    embeddings = np.zeros((len(entries),expected_dimension),dtype=np.float32)
    embeddings[mask] = features
    return embeddings,mask,texts
