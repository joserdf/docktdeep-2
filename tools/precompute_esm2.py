#!/usr/bin/env python3
"""Pre-compute frozen ESM-2 protein embeddings for unique receptor sequences.

For each unique sequence in ``data/embeddings/proteins_unique.fasta``, compute a
per-sequence vector by running ESM-2 (HuggingFace ``facebook/esm2_*``) and
mean-pooling over the last hidden states of the sequence (excluding padding).
Supports several model sizes (largest to smallest) and caches one ``.npy`` file
per sequence under ``data/embeddings/esm2/<model>/<seq_hash>.npy``.

Model ids (HuggingFace):
  esm2_t36_3B_UR50D   (~3B params, 2560-d, needs bf16 on 12GB GPU)
  esm2_t33_650M_UR50D (~650M, 1280-d)
  esm2_t30_150M_UR50D (~150M, 640-d)
  esm2_t12_35M_UR50D  (~35M, 480-d)
  esm2_t6_8M_UR50D    (~8M, 320-d)

Run on GPU when available; falls back to CPU. Embeddings are L2-normalized and
mean-pooled (spec 01: frozen features, attention pooling is the alternative).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

MODELS = {
    # esm2-3B (~3B, 2560-d) needs ~24GB VRAM with long sequences -> diamante GPU.
    "esm2-650M": ("facebook/esm2_t33_650M_UR50D", 1280),
    "esm2-150M": ("facebook/esm2_t30_150M_UR50D", 640),
    "esm2-35M": ("facebook/esm2_t12_35M_UR50D", 480),
    "esm2-8M": ("facebook/esm2_t6_8M_UR50D", 320),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(MODELS),
                   help="Model keys to precompute (order = largest to smallest).")
    p.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default=None,
                   help="torch device; default = cuda if available else cpu.")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of sequences (sanity).")
    p.add_argument("--max-length", type=int, default=1024,
                   help="Truncate sequences longer than this (ESM-2 supports up to 1022 tokens).")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoModel, AutoTokenizer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else ''})")

    fasta = args.embeddings_dir / "proteins_unique.fasta"
    seqs = []
    with open(fasta) as f:
        header, buf = None, []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    seqs.append((header, "".join(buf)))
                header = line[1:]
                buf = []
            else:
                buf.append(line)
        if header is not None:
            seqs.append((header, "".join(buf)))
    if args.limit:
        seqs = seqs[: args.limit]
    print(f"{len(seqs)} sequências únicas")

    for key in args.models:
        model_id, dim = MODELS[key]
        out_dir = args.embeddings_dir / "esm2" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key} ({model_id}, dim={dim}) ===")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        model.to(device).eval()
        use_amp = device == "cuda"
        dtype = torch.float16 if use_amp else torch.float32
        with torch.no_grad():
            for h, seq in seqs:
                out_file = out_dir / f"{h}.npy"
                if out_file.exists():
                    continue
                inp = tok([seq], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    out = model(**inp).last_hidden_state
                out = out.float()  # upcast before pooling for numeric stability
                mask = inp["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                np.save(out_file, pooled.cpu().numpy()[0])
        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  ok: {out_dir}")

    print("\ndone")


if __name__ == "__main__":
    main()
