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
    # Os grandes so cabem em fp16: 3B sao 11GB em fp32 (estoura a 3060 de 12GB
    # antes das ativacoes) e 15B sao ~60GB. Use --dtype float16 para eles.
    "esm2-15B": ("facebook/esm2_t48_15B_UR50D", 5120),
    "esm2-3B": ("facebook/esm2_t36_3B_UR50D", 2560),
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
    p.add_argument("--device-map", type=str, default=None,
                   help="device_map do accelerate (ex.: auto). Necessario para o 15B: "
                        "sao ~30GB em fp16 e nao cabem nos 24GB de uma 4090; o que "
                        "nao couber transborda p/ RAM. Exige --batch-size alto.")
    p.add_argument("--max-gpu-mem", type=str, default=None,
                   help="teto da fatia na GPU com --device-map (ex.: 6GiB). Sem isto o "
                        "accelerate toma toda a memoria livre — indelicado numa GPU "
                        "compartilhada, onde o job do outro usuario ainda pode crescer.")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"],
                   help="dtype dos PESOS. float32 e o default historico (650M e menores foram "
                        "gerados assim); 3B/15B so cabem em float16.")
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
        weight_dtype = getattr(torch, args.dtype)
        if args.device_map:
            kw = {}
            if args.max_gpu_mem:
                kw["max_memory"] = {0: args.max_gpu_mem, "cpu": "40GiB"}
            model = AutoModel.from_pretrained(model_id, torch_dtype=weight_dtype,
                                              device_map=args.device_map, **kw)
        else:
            model = AutoModel.from_pretrained(model_id, torch_dtype=weight_dtype)
            model.to(device)
        model.eval()
        use_amp = device == "cuda"
        dtype = torch.float16 if use_amp else torch.float32
        pending = [(h, s) for h, s in seqs if not (out_dir / f"{h}.npy").exists()]
        # Ordenar por tamanho agrupa sequencias parecidas no mesmo batch: com
        # padding dinamico isso corta o desperdicio (as unicas vao de ~30 a 1022
        # residuos). A ordem nao importa — cada saida e nomeada pelo hash.
        pending.sort(key=lambda hs: len(hs[1]))
        bs = max(1, args.batch_size)
        print(f"  {len(pending)} pendentes, batch={bs}")
        with torch.no_grad():
            for i in range(0, len(pending), bs):
                chunk = pending[i:i + bs]
                inp = tok([s for _, s in chunk], return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(device)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                    out = model(**inp).last_hidden_state
                out = out.float()  # upcast before pooling for numeric stability
                mask = inp["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                arr = pooled.cpu().numpy()
                for j, (h, _) in enumerate(chunk):
                    np.save(out_dir / f"{h}.npy", arr[j])
        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  ok: {out_dir}")

    print("\ndone")


if __name__ == "__main__":
    main()
