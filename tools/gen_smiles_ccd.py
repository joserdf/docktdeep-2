#!/usr/bin/env python3
"""Generate ligand SMILES with bond orders for PDBbind2020 complexes.

``gen_smiles.py`` adds every bond as SINGLE, so its SMILES are heavy-atom
skeletons. This version restores bond orders, following DockTData
(``projects/docktdata``): the wwPDB Chemical Component Dictionary (CCD) is the
source of chemistry, and peptides are built from their sequence.

Each complex takes the first path that works, and records it in ``source``:

  ``template``    ``lig_name`` is a CCD code (or ``A/B`` alternates): bond orders
                  from the CCD molecule via ``AssignBondOrdersFromTemplate``
                  onto the pocket coordinates; stereo from the 3D.
  ``sequence``    ``n-mer`` peptide: RCSB sequence (``fetch_rcsb_peptides.py``)
                  cut to the modeled residues and built by DockTData's
                  ``sequence_to_mol``; the instance whose heavy-atom count is
                  closest to the pickle wins, accepted within ``--pep-tol``.
  ``determine``   everything else: ``rdDetermineBonds.DetermineBondOrders`` on the
                  pocket coordinates (the pickles carry explicit hydrogens),
                  total charge 0, +-1, +-2, killed after ``--determine-timeout``
                  (it can hang inside C++ on some ligands).
  ``ccd_direct``  a CCD code whose template failed but whose single-bond graph
                  contains the pocket's, or is contained in it (missing atoms,
                  covalent adduct): SMILES of the CCD molecule itself, tried
                  before ``determine``. When neither graph contains the other,
                  ``lig_name`` disagrees with the pocket (1hty says TRS, the
                  coordinates are MPD) and the CCD molecule is not used.
  ``skeleton``    nothing worked: the old single-bond SMILES.

Every molecule is neutralized (``rdMolStandardize.Uncharger``) before the
SMILES and InChIKey are written. The sources start from different protonation
states (CCD ideal, CCD monomers, the prepared pocket); Morgan fingerprints hash
formal charge, so without this the same ligand would differ by source.

Outputs (under ``--out-dir``; the old ``data/embeddings`` files are untouched):
  ``ligand_smiles.csv``       complex_id, smiles, source, inchikey, heavy_pocket, heavy_out, detail
  ``ligands_unique.txt``      one SMILES per line (input of precompute_chemberta.py)
  ``complex_to_smiles.json``  ``{complex_id: smiles}``

Needs DockTData on the path for the peptide builder:
  PYTHONPATH=<docktdata> python tools/gen_smiles_ccd.py \\
      --ccd-sdf <components-pub.sdf.gz> --ccd-cif <components.cif[.gz]>
"""

import argparse
import gzip
import json
import os
import pickle
import re
import select
import signal
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from rdkit.Chem.MolStandardize import rdMolStandardize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_smiles import coords_to_mol, coords_to_smiles  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Filled in the parent before the fork, read by the workers.
CCD = {}
RCSB = {}
PEP_LIB = None
ARGS = None


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index", type=Path, default=Path("data/pdbbind2020/index-pfam.csv"))
    p.add_argument("--root-dir", type=Path, default=Path("data/pdbbind2020/processed"))
    p.add_argument("--out-dir", type=Path, default=Path("data/embeddings/lig-v2"))
    p.add_argument("--ligand-pattern", type=str, default="{c}_ligand_rnum.pdb.pkl")
    p.add_argument("--bond-tol", type=float, default=1.35)
    p.add_argument("--ccd-sdf", type=Path, required=True, help="wwPDB components-pub.sdf.gz")
    p.add_argument("--ccd-cif", type=Path, required=True, help="wwPDB components.cif[.gz] (peptide monomers)")
    p.add_argument("--ccd-cache", type=Path, default=Path("data/ccd-cache"),
                   help="SQLite cache of the parsed monomer library.")
    p.add_argument("--rcsb-peptides", type=Path, default=Path("data/embeddings/lig-v2/rcsb_peptides.json"))
    p.add_argument("--pep-tol", type=int, default=5,
                   help="Max |heavy atoms built - heavy atoms in the pocket| for a peptide.")
    p.add_argument("--determine-timeout", type=float, default=20.0,
                   help="Seconds before a DetermineBondOrders call is killed.")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0, help="First N complexes only (0 = all).")
    return p.parse_args()


def category(lig_name):
    n = str(lig_name)
    if re.fullmatch(r"\d+-mer", n):
        return "peptide", []
    codes = n.split("/")
    if all(re.fullmatch(r"[A-Z0-9]{1,5}", c) for c in codes):
        return "code", codes
    return "other", []


def load_ccd(path, needed):
    """{comp_id: heavy-atom Mol} for the needed codes; DATIVE -> SINGLE as in DockTData."""
    out = {}
    with gzip.open(path, "rb") as fh:
        for mol in Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=False):
            if mol is None or not mol.HasProp("_Name"):
                continue
            code = mol.GetProp("_Name")
            if code not in needed:
                continue
            for b in mol.GetBonds():
                if b.GetBondType() == Chem.BondType.DATIVE:
                    b.SetBondType(Chem.BondType.SINGLE)
            # RemoveAllHs: RemoveHs keeps the H that defines double-bond stereo,
            # which breaks the atom-count match against the heavy-atom skeleton.
            try:
                out[code] = Chem.RemoveAllHs(mol)
            except Exception:  # DATIVE -> SINGLE can leave it unkekulizable
                continue
    return out


def finish(mol):
    """Neutralize and sanitize; returns (smiles, inchikey, heavy)."""
    mol = rdMolStandardize.Uncharger().uncharge(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol), mol.GetNumHeavyAtoms()


def from_template(skel_heavy, codes):
    """CCD molecule with the pocket's coordinates mapped onto it.

    The connectivity comes from the CCD, not from the skeleton: the distance
    rule adds spurious bonds (S-C across thiophenes), and
    ``AssignBondOrdersFromTemplate`` would keep them because it matches the
    template as a substructure. Only the 3D (hence the stereo) is taken from
    the pocket, through the atom mapping.
    """
    pocket = single_graph(skel_heavy)
    pos = skel_heavy.GetConformer().GetPositions()
    for code in codes:
        tpl = CCD.get(code)
        if tpl is None or tpl.GetNumAtoms() != skel_heavy.GetNumAtoms():
            continue
        match = pocket.GetSubstructMatch(single_graph(tpl))
        if not match:
            continue
        mol = Chem.Mol(tpl)
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, j in enumerate(match):
            conf.SetAtomPosition(i, Point3D(*pos[j]))
        mol.RemoveAllConformers()
        mol.AddConformer(conf, assignId=True)
        Chem.AssignStereochemistryFrom3D(mol)
        return mol, code
    return None, None


def single_graph(mol):
    """Heavy-atom graph with every bond single, no aromaticity, no charges."""
    m = Chem.RWMol(mol)
    for a in m.GetAtoms():
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    m = m.GetMol()
    m.UpdatePropertyCache(strict=False)
    return m


def ccd_related(skel_heavy, codes):
    """First CCD code whose single-bond graph contains the pocket's or is contained in it."""
    pocket = single_graph(skel_heavy)
    for code in codes:
        if code not in CCD:
            continue
        tpl = single_graph(CCD[code])
        if tpl.HasSubstructMatch(pocket) or pocket.HasSubstructMatch(tpl):
            return code
    return None


def unobserved(inst):
    s = set()
    for f in inst.get("rcsb_polymer_instance_feature") or []:
        if f["type"] == "UNOBSERVED_RESIDUE_XYZ":
            for p in f["feature_positions"] or []:
                s.update(range(p["beg_seq_id"], (p["end_seq_id"] or p["beg_seq_id"]) + 1))
    return s


def from_sequence(cid, heavy_pocket):
    """Modeled slice of the RCSB peptide entity closest in heavy atoms to the pocket ligand."""
    from docktdata.ETL.transform.PDB.peptide.api import sequence_to_mol
    from docktdata.ETL.transform.PDB.peptide.parser import parse_sequence

    entry = RCSB.get(cid.upper())
    if entry is None:
        return None, "no_rcsb"
    built, best = {}, None
    for ent in entry["polymer_entities"] or []:
        ep = ent["entity_poly"]
        if not ep["type"].startswith("polypeptide") or (ep["rcsb_sample_sequence_length"] or 999) > 60:
            continue
        try:
            toks = parse_sequence(ep["pdbx_seq_one_letter_code"].replace("\n", ""))
        except Exception:
            continue
        for inst in ent["polymer_entity_instances"] or []:
            un = unobserved(inst)
            mod = [i for i in range(1, len(toks) + 1) if i not in un]
            if not mod:
                continue
            key = "-".join(toks[min(mod) - 1:max(mod)])
            if key not in built:
                try:
                    built[key] = Chem.RemoveHs(sequence_to_mol(key, library=PEP_LIB))
                except Exception:
                    built[key] = None
            if built[key] is None:
                continue
            d = abs(built[key].GetNumHeavyAtoms() - heavy_pocket)
            if best is None or d < best[0]:
                best = (d, key, ent["rcsb_id"])
    if best is None:
        return None, "no_build"
    if best[0] > ARGS.pep_tol:
        return None, f"heavy_diff={best[0]}"
    return Chem.Mol(built[best[1]]), f"{best[2]}:{best[1]} diff={best[0]}"


def in_child(fn, timeout):
    """Run fn() in a forked child and kill it after timeout seconds.

    Pool workers are daemonic and cannot start multiprocessing children, and
    SIGALRM does not interrupt a C++ call, so this forks by hand.
    """
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            payload = pickle.dumps(fn())
        except BaseException:
            payload = pickle.dumps((None, None))
        with os.fdopen(w, "wb") as fh:
            fh.write(payload)
        os._exit(0)
    os.close(w)
    ready, _, _ = select.select([r], [], [], timeout)
    if not ready:
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)
        os.close(r)
        return None, "timeout"
    with os.fdopen(r, "rb") as fh:
        data = fh.read()
    os.waitpid(pid, 0)
    return pickle.loads(data) if data else (None, None)


def from_determine(skel):
    for charge in (0, 1, -1, 2, -2):
        mol = Chem.Mol(skel)
        try:
            rdDetermineBonds.DetermineBondOrders(mol, charge=charge)
            Chem.SanitizeMol(mol)
            if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                continue
            Chem.AssignStereochemistryFrom3D(mol)
            return Chem.RemoveHs(mol), f"charge={charge}"
        except Exception:
            continue
    return None, None


def work(item):
    cid, lig_name = item
    path = ARGS.root_dir / ARGS.ligand_pattern.format(c=cid)
    if not path.exists():
        return None
    lig = pickle.load(open(path, "rb"))
    coords = lig.coords.numpy() if hasattr(lig.coords, "numpy") else np.array(lig.coords)
    elements = np.array(lig.element_symbols)
    skel = coords_to_mol(coords, elements, ARGS.bond_tol)
    skel_heavy = Chem.RemoveHs(skel, sanitize=False)
    heavy_pocket = skel_heavy.GetNumAtoms()
    cat, codes = category(lig_name)

    tries = []
    if cat == "code":
        tries.append(("template", lambda: from_template(skel_heavy, codes)))
        tries.append(("ccd_direct", lambda: (lambda c: (Chem.Mol(CCD[c]), c) if c else (None, "unrelated"))(
            ccd_related(skel_heavy, codes))))
    elif cat == "peptide":
        tries.append(("sequence", lambda: from_sequence(cid, heavy_pocket)))
    tries.append(("determine", lambda: in_child(lambda: from_determine(skel), ARGS.determine_timeout)))

    notes = []
    for source, fn in tries:
        mol, detail = fn()
        if mol is None:
            if detail:
                notes.append(f"{source}:{detail}")
            continue
        try:
            smi, ik, heavy = finish(mol)
            return (cid, smi, source, ik, heavy_pocket, heavy, "; ".join(notes + [str(detail)]))
        except Exception as e:
            notes.append(f"{source}:finish:{type(e).__name__}")
    try:
        smi = coords_to_smiles(coords, elements, ARGS.bond_tol)
    except Exception:  # same as gen_smiles.py: no SMILES for this complex
        return None
    return (cid, smi, "skeleton", "", heavy_pocket, Chem.MolFromSmiles(smi, sanitize=False).GetNumAtoms()
            if smi else 0, "; ".join(notes))


def main():
    global CCD, RCSB, PEP_LIB, ARGS
    ARGS = args = parse_args()
    t0 = time.time()
    df = pd.read_csv(args.index, usecols=["id", "lig_name"], low_memory=False)
    if args.limit:
        df = df.head(args.limit)
    needed = {c for n in df["lig_name"] for c in category(n)[1]}
    CCD = load_ccd(args.ccd_sdf, needed)
    print(f"CCD: {len(CCD)} of {len(needed)} codes ({time.time() - t0:.0f}s)")
    RCSB = json.loads(args.rcsb_peptides.read_text())
    from docktdata.ETL.transform.PDB.peptide.monomer_library import load_library
    PEP_LIB = load_library(args.ccd_cif, cache_dir=args.ccd_cache)
    print(f"monomer library: {len(PEP_LIB)} ({time.time() - t0:.0f}s)")

    with Pool(args.workers) as pool:
        rows = [r for r in pool.imap(work, zip(df["id"], df["lig_name"]), chunksize=16) if r]

    cols = ["complex_id", "smiles", "source", "inchikey", "heavy_pocket", "heavy_out", "detail"]
    out = pd.DataFrame(rows, columns=cols)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "ligand_smiles.csv", index=False)
    (args.out_dir / "ligands_unique.txt").write_text("\n".join(sorted(set(out["smiles"]))) + "\n")
    (args.out_dir / "complex_to_smiles.json").write_text(json.dumps(dict(zip(out["complex_id"], out["smiles"]))))
    print(f"complexes: {len(out)}, unique SMILES: {out['smiles'].nunique()}, "
          f"sources: {dict(Counter(out['source']))} ({time.time() - t0:.0f}s) -> {args.out_dir}/")


if __name__ == "__main__":
    main()
