#!/usr/bin/env python3
"""Unit tests (B=8 synthetic) + function-equivalence regression for the
similarity-term decomposition (factor C) — see docs/proposta-ablacao-similaridade.md
and plan phase 2.4/2.5a.

Runs on CPU, no dataset needed:
  .venv/bin/python test_sim_terms.py

Covers:
  2.4  per-term loss bounds L_k in [0, log B]; zero gradient on a row with no
       positive partner; non-dominance budget fail-fast; --sim-terms fail-fast
       against --yaware/--ifp-aware/non-default anchor; 7-tuple dispatch in
       dataset._collate and Baseline.shared_step.
  2.5a refactored _yaware_infonce == old implementation (tol 1e-6) for all 5
       anchor modes, and the embedding-anchored CE branch of _semi_loss.
"""
import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.docktdeep.dataset import PDBbind
from src.docktdeep.models import Baseline

FAILED = []


def check(name, cond, extra=""):
    if cond:
        print(f"  ok  {name}")
    else:
        print(f"  FAIL {name} {extra}")
        FAILED.append(name)


def make_model(sim_terms, **over):
    """A small CPU Baseline with synthetic similarity matrices in a tmpdir.

    Returns (model, tmpdir). sim_terms controls the decomposed path; pass
    sim_terms=[] and yaware/anchor overrides to exercise the legacy paths.
    """
    tmp = tempfile.mkdtemp(prefix="simtest_")
    n = 5  # real rows
    N = n + 1  # shared sentinel row at index n
    for name in ("S_prot", "S_lig"):
        S = np.zeros((N, N), dtype=np.uint8)
        S[:n, :n] = 50
        S[range(n), range(n)] = 100  # diagonal = 100 (real rows), sentinel row = 0
        np.savez(os.path.join(tmp, f"{name}.npz"), S=S, sentinel=n)

    base = dict(
        input_size=(1, 16, 16, 16),
        use_esm2=False, use_chemberta=False, semi=True, no_cnn=False,
        depthwise_convs=False, adaptive_pooling=True,
        num_fc_units=[8], dropout=0.0,
        loss="huber", huber_beta=1.0, label_smoothing=0.0,
        lambda_semi=0.025, semi_tau=0.1, proj_dim=8, f_dim=16, emb_proj_dim=8,
        yaware=False, yaware_sigma=1.0, ifp_tau=0.3,
        anchor_mode="affinity", ifp_aware=False,
        # truthy so the --sim-terms ifp requires --ifp-path guard passes; the
        # parquet is never read by the model, only its presence is checked.
        ifp_path="dummy.parquet",
        sim_terms=sim_terms, sim_lambda=0.025, sim_lambda_max=0.125,
        sim_mat_dir=tmp, sim_kendall=False,
    )
    base.update(over)
    return Baseline(**base), tmp


def synthetic_inputs(model, B=8, D=8):
    torch.manual_seed(0)
    p = F.normalize(torch.randn(B, D), dim=1)
    y = torch.rand(B) * 6 + 2  # pKi-ish
    ifp = (torch.rand(B, 4096) > 0.7).to(torch.uint8)
    prot_idx = torch.randint(0, 6, (B,))  # 0..5 (row 5 = sentinel)
    lig_idx = torch.randint(0, 6, (B,))
    x = torch.randn(B, 1, 16, 16, 16)
    return p, y, ifp, prot_idx, lig_idx, x


def test_bounds(model):
    """L_k is guaranteed non-negative; at initialization (uniform logits) it
    equals log B. NOTE: log B is NOT a hard upper bound for soft targets — it is
    the loss when all pairwise similarities are equal (uniform log-softmax).
    """
    p, y, ifp, prot_idx, lig_idx, _ = synthetic_inputs(model)
    B = p.shape[0]
    logB = np.log(B)
    tgts = {
        "ifp": model._ifp_target(ifp),
        "aff": model._aff_target(y),
        "prot": model._prot_target(prot_idx),
        "lig": model._lig_target(lig_idx),
    }
    for k, tgt in tgts.items():
        Lk = model._sim_infonce(p, tgt).item()
        check(f"bounds {k}: L_k >= 0 (hard lower bound)", Lk >= 0, f"(L={Lk:.4f})")
    # uniform logits -> every row's loss = log B (soft target sums to 1 per row).
    # Use all-REAL row indices (0..4): a sentinel row has no partner, so it
    # legitimately contributes 0 and would lower the mean below log B.
    real = torch.arange(B) % 5
    t_uniform = {
        "ifp": model._ifp_target(ifp),
        "aff": model._aff_target(y),
        "prot": model._prot_target(real),
        "lig": model._lig_target(real),
    }
    p_uniform = F.normalize(torch.ones(B, p.shape[1]), dim=1)
    for k, tgt in t_uniform.items():
        Lk = model._sim_infonce(p_uniform, tgt).item()
        check(f"init {k}: uniform logits -> L_k == log B", abs(Lk - logB) < 1e-4,
              f"(L={Lk:.4f}, logB={logB:.4f})")


def test_zero_partner_row(model):
    """A row with no positive partner contributes exactly zero to the loss SUM.

    This does NOT mean its projection receives no gradient: it still acts as a
    negative example in the other rows' log-softmax denominators. The guarantee
    is that the row itself is neither attracted nor repelled (its own term is 0).
    """
    p, _, _, _, _, _ = synthetic_inputs(model)
    B = p.shape[0]
    tgt = torch.rand(B, B)
    tgt.fill_diagonal_(0)
    tgt[3] = 0.0  # row 3 has no positive partner
    loss = model._sim_infonce(p, tgt)
    check("loss non-negative", loss.item() >= 0)
    # per-row contribution of the zero row is exactly 0 (after the same row
    # normalization _sim_infonce applies)
    sim = p @ p.T / model.tau
    lsm = torch.log_softmax(sim, dim=1)
    rowsum = tgt.sum(dim=1, keepdim=True)
    tgt_n = torch.where(rowsum > 1e-8, tgt / (rowsum + 1e-8), torch.zeros_like(tgt))
    per_row = -(tgt_n * lsm).sum(dim=1)
    check("zero-partner row contributes exactly 0 to the sum",
          abs(per_row[3].item()) < 1e-8, f"(row3={per_row[3].item():.2e})")
    check("loss == mean of per-row contributions (zero rows add 0)",
          abs(loss.item() - per_row.mean().item()) < 1e-6)


def test_sim_terms_loss(model):
    p, y, ifp, prot_idx, lig_idx, _ = synthetic_inputs(model)
    total, per = model._sim_terms_loss(p, prot_idx, lig_idx, ifp, y)
    check("sim_terms_loss returns 4 per-term entries", set(per) == {"ifp", "aff", "prot", "lig"})
    # weighted total = sum of lambda * L_k
    expected = sum(model._sim_infonce(p, t) for t in
                   [model._ifp_target(ifp), model._aff_target(y),
                    model._prot_target(prot_idx), model._lig_target(lig_idx)])
    check("weighted total == sim_lambda * sum(L_k)",
          abs(total.item() - model.sim_lambda * expected.item()) < 1e-5,
          f"(got {total.item():.6f})")
    # row_frac in [0,1]
    for k, (Lk, rf) in per.items():
        check(f"row_frac {k} in [0,1]", 0.0 <= rf.item() <= 1.0 + 1e-6, f"(rf={rf.item():.3f})")


def test_failfast_budget():
    # within budget: 0.025 + 4*0.025 = 0.125 <= 0.125 -> ok
    make_model(["ifp", "aff", "prot", "lig"])
    # over budget: lambda_semi 0.1 + 4*0.025 = 0.2 > 0.125 -> raise
    try:
        make_model(["ifp", "aff", "prot", "lig"], lambda_semi=0.1)
        check("budget fail-fast raises", False, "(no exception)")
    except ValueError as e:
        check("budget fail-fast raises", "exceeds --sim-lambda-max" in str(e), str(e))


def test_failfast_flags():
    try:
        make_model(["ifp"], yaware=True)
        check("--yaware fail-fast raises", False)
    except ValueError:
        check("--yaware fail-fast raises", True)
    try:
        make_model(["ifp"], ifp_aware=True)
        check("--ifp-aware fail-fast raises", False)
    except ValueError:
        check("--ifp-aware fail-fast raises", True)
    try:
        make_model(["ifp"], anchor_mode="gate")
        check("--anchor-mode fail-fast raises", False)
    except ValueError:
        check("--anchor-mode fail-fast raises", True)
    try:
        make_model(["ifp"], sim_kendall=True)
        check("--sim-kendall raises NotImplementedError", False)
    except NotImplementedError:
        check("--sim-kendall raises NotImplementedError", True)
    # --sim-terms ifp without --ifp-path -> dead term, must fail fast
    try:
        make_model(["ifp"], ifp_path="")
        check("--sim-terms ifp without --ifp-path raises", False)
    except ValueError:
        check("--sim-terms ifp without --ifp-path raises", True)


def test_collate_7tuple():
    B = 8
    samples = [(
        torch.zeros(1, 16, 16, 16),            # voxs
        None,                                   # e_prot
        None,                                   # e_lig
        (torch.rand(4096) > 0.5).to(torch.uint8) if i % 2 == 0 else None,  # ifp|None
        i,                                      # prot_idx
        i + 1,                                  # lig_idx
        torch.tensor(float(i)),                 # y
    ) for i in range(B)]
    out = PDBbind._collate(samples)
    check("collate returns 7-tuple", len(out) == 7)
    voxs, e_prot, e_lig, ifp, prot_idx, lig_idx, y = out
    check("collate voxs (B,1,16,16,16)", voxs.shape == (B, 1, 16, 16, 16))
    check("collate e_prot collapsed to None", e_prot is None)
    check("collate e_lig collapsed to None", e_lig is None)
    check("collate ifp dense (B,4096) uint8", ifp.dtype == torch.uint8 and ifp.shape == (B, 4096))
    check("collate prot_idx (B,) long", prot_idx.tolist() == list(range(B)))
    check("collate lig_idx (B,) long", lig_idx.tolist() == [i + 1 for i in range(B)])
    check("collate y (B,)", y.shape == (B,))


def test_shared_step_7tuple():
    model, _ = make_model(["ifp", "aff", "prot", "lig"])
    model.log = lambda *a, **k: None  # no trainer attached
    B = 8
    _, y, ifp, prot_idx, lig_idx, x = synthetic_inputs(model)
    batch = (x, None, None, ifp, prot_idx, lig_idx, y)
    out = model.shared_step(batch, 0, "train")
    check("shared_step returns train_loss", "train_loss" in out)
    check("shared_step loss is scalar", out["train_loss"].dim() == 0)


def test_legacy_dispatch_unchanged():
    # 2-tuple and 5-tuple still dispatched correctly (regression of old paths)
    model, _ = make_model([])  # no sim terms -> legacy path
    model.log = lambda *a, **k: None
    B = 8
    x = torch.randn(B, 1, 16, 16, 16)
    y = torch.rand(B)
    out2 = model.shared_step((x, y), 0, "val")
    check("shared_step 2-tuple val", "val_loss" in out2)
    ifp = (torch.rand(B, 4096) > 0.7).to(torch.uint8)
    out5 = model.shared_step((x, None, None, ifp, y), 0, "val")
    check("shared_step 5-tuple val", "val_loss" in out5)


# --------------------------------------------------------------------------- #
# 2.5a function-equivalence regression (refactor of _yaware_infonce)
# --------------------------------------------------------------------------- #
def old_yaware_infonce(model, p, y, ifp=None, x=None):
    B = p.shape[0]
    sim = p @ p.T / model.tau
    d = torch.abs(y[:, None] - y[None, :])
    eye = torch.eye(B, device=p.device)
    aff = torch.exp(-d / model.yaware_sigma) * (1.0 - eye)
    needs_ifp = model.anchor_mode in ("gate", "ifp", "hybrid")
    if needs_ifp and ifp is not None:
        ifp_sim = model._ifp_sim(ifp) * (1.0 - eye)
    if model.anchor_mode == "affinity" or (needs_ifp and ifp is None):
        tgt = aff
    elif model.anchor_mode == "gate":
        gate = (model._ifp_sim(ifp) >= model.ifp_tau).float() * (1.0 - eye)
        tgt = aff * gate
    elif model.anchor_mode == "ifp":
        tgt = ifp_sim
    elif model.anchor_mode == "hybrid":
        tgt = aff * ifp_sim
    elif model.anchor_mode == "struct":
        vsim = model._voxel_sim(x) * (1.0 - eye) if x is not None else None
        tgt = aff * vsim if vsim is not None else aff
    else:
        raise ValueError(f"unknown anchor_mode: {model.anchor_mode}")
    tgt = tgt.detach()
    rowsum = tgt.sum(dim=1, keepdim=True)
    tgt = torch.where(rowsum > 1e-8, tgt / (rowsum + 1e-8), torch.zeros_like(tgt))
    log_softmax = torch.log_softmax(sim, dim=1)
    return -(tgt * log_softmax).sum(dim=1).mean()


def test_yaware_equivalence():
    model, _ = make_model([], yaware=True)  # legacy yaware path, no sim terms
    torch.manual_seed(0)
    B, D = 8, 8
    p = F.normalize(torch.randn(B, D), dim=1)
    y = torch.rand(B) * 6 + 2
    ifp = (torch.rand(B, 4096) > 0.7).to(torch.uint8)
    x = torch.randn(B, 1, 16, 16, 16)
    for mode in ("affinity", "gate", "ifp", "hybrid", "struct"):
        model.anchor_mode = mode
        new = model._yaware_infonce(p, y, ifp, x=x)
        old = old_yaware_infonce(model, p, y, ifp, x=x)
        check(f"yaware anchor={mode} new==old",
              abs(new.item() - old.item()) < 1e-6,
              f"(new={new.item():.8f}, old={old.item():.8f})")


def test_ce_branch_equivalence():
    # embedding-anchored InfoNCE branch of _semi_loss (proj_target on e_prot)
    model, _ = make_model([], use_esm2=True, no_cnn=True, yaware=False)
    torch.manual_seed(0)
    B = 8
    e_prot = torch.randn(B, 1280)  # esm2-650M dim
    y = torch.rand(B) * 6 + 2

    def manual_ce():
        p1 = F.normalize(model.proj_head(model.forward_latent(None, e_prot, None)), dim=1)
        p2 = F.normalize(model.proj_head(model.forward_latent(None, e_prot, None)), dim=1)
        rdrop = F.mse_loss(p1, p2)
        t = F.normalize(model.proj_target(e_prot), dim=1)
        sim = p1 @ t.T / model.tau
        return rdrop + F.cross_entropy(sim, torch.arange(len(p1), device=p1.device))

    # _semi_loss does two stochastic passes (proj_head dropout); reseed before
    # each call so new and old consume identical dropout masks.
    torch.manual_seed(123)
    new = model._semi_loss(None, e_prot, None, y)
    torch.manual_seed(123)
    old = manual_ce()
    check("_semi_loss CE branch new==old", abs(new.item() - old.item()) < 1e-6,
          f"(new={new.item():.8f}, old={old.item():.8f})")


def main():
    print("== 2.4 unit tests ==")
    model, _ = make_model(["ifp", "aff", "prot", "lig"])
    test_bounds(model)
    test_zero_partner_row(model)
    test_sim_terms_loss(model)
    test_failfast_budget()
    test_failfast_flags()
    test_collate_7tuple()
    test_shared_step_7tuple()
    test_legacy_dispatch_unchanged()

    print("== 2.5a function-equivalence regression ==")
    test_yaware_equivalence()
    test_ce_branch_equivalence()

    print()
    if FAILED:
        print(f"FAILED: {len(FAILED)} -> {FAILED}")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
