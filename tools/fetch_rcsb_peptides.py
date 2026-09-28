#!/usr/bin/env python3
"""Fetch RCSB polymer entities for the PDBbind peptide ligands (``n-mer``).

The processed ligand pickles store a peptide as one pseudo-residue ``MOL`` with
a renamed chain, so the sequence has to come from the RCSB. This script asks
the RCSB GraphQL API for every polymer entity of each ``n-mer`` entry, with the
fields DockTData's ``query_data.py`` uses plus the unobserved-residue features
that ``gen_smiles_ccd.py`` needs to cut the modeled slice.

The network step is kept apart so the SMILES generation runs offline.

Output: ``--out`` JSON ``{PDB_ID: entry}`` (upper-case ids, raw GraphQL entry).
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

URL = "https://data.rcsb.org/graphql"
QUERY = """query($ids:[String!]!){entries(entry_ids:$ids){rcsb_id
 polymer_entities{rcsb_id entity_poly{type rcsb_entity_polymer_type pdbx_seq_one_letter_code
   pdbx_seq_one_letter_code_can rcsb_sample_sequence_length}
  polymer_entity_instances{rcsb_id
   rcsb_polymer_entity_instance_container_identifiers{asym_id auth_asym_id auth_to_entity_poly_seq_mapping}
   rcsb_polymer_instance_info{modeled_residue_count}
   rcsb_polymer_instance_feature{type feature_positions{beg_seq_id end_seq_id}}}}}}"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, default=Path("data/pdbbind2020/index-pfam.csv"))
    p.add_argument("--out", type=Path, default=Path("data/embeddings/lig-v2/rcsb_peptides.json"))
    p.add_argument("--batch", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.index, usecols=["id", "lig_name"], low_memory=False)
    ids = sorted({c.upper() for c, n in zip(df["id"], df["lig_name"])
                  if re.fullmatch(r"\d+-mer", str(n))})
    out = {}
    for k in range(0, len(ids), args.batch):
        for attempt in range(4):
            r = requests.post(URL, json={"query": QUERY, "variables": {"ids": ids[k:k + args.batch]}},
                              timeout=120)
            if r.ok:
                break
            time.sleep(5 * (attempt + 1))
        r.raise_for_status()
        for e in r.json()["data"]["entries"] or []:
            if e:
                out[e["rcsb_id"]] = e
        time.sleep(0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out))
    print(f"n-mer entries: {len(ids)}, returned by RCSB: {len(out)} -> {args.out}")


if __name__ == "__main__":
    main()
