"""
Command-line interface for docktdeep.
"""

import argparse
import csv
import logging
import os
import sys
import tempfile
from typing import List, Tuple

from biopandas.mol2.mol2_io import split_multimol2

from .inference import predict_binding_affinity

logger = logging.getLogger(__name__)


def expand_ligands(
    ligand_paths: List[str], temp_dir: str
) -> Tuple[List[str], List[str]]:
    """Expand multi-mol2 files into individual ligand files.

    For each ligand path, if it is a .mol2 file containing multiple molecules,
    split it into individual files inside temp_dir. Otherwise, keep the path as-is.

    Returns:
        A tuple of (expanded_file_paths, display_names).
        Display names use "original_path:index" for molecules from multi-mol2 files.
    """
    expanded_paths = []
    display_names = []

    for arg_idx, ligand_path in enumerate(ligand_paths):
        if ligand_path.lower().endswith((".mol2", ".mol2.gz")):
            molecules = list(split_multimol2(ligand_path))
            if len(molecules) == 1:
                expanded_paths.append(ligand_path)
                display_names.append(ligand_path)
            else:
                logger.info(
                    f"Expanded {ligand_path} into {len(molecules)} individual ligands"
                )
                for mol_idx, (mol_id, mol_lines) in enumerate(molecules):
                    temp_file = os.path.join(
                        temp_dir, f"lig{arg_idx}_{mol_idx}.mol2"
                    )
                    with open(temp_file, "w") as f:
                        f.writelines(mol_lines)
                    expanded_paths.append(temp_file)
                    display_names.append(f"{ligand_path}:{mol_idx}")
        else:
            expanded_paths.append(ligand_path)
            display_names.append(ligand_path)

    return expanded_paths, display_names


def match_proteins_to_ligands(
    proteins: List[str], ligands: List[str]
) -> List[str]:
    """Replicate a single protein to match multiple ligands if needed.

    Returns the (possibly expanded) list of protein paths.

    Raises:
        ValueError: If the number of proteins doesn't match the number of ligands
            and there isn't exactly one protein to replicate.
    """
    if len(proteins) == len(ligands):
        return proteins
    if len(proteins) == 1 and len(ligands) > 1:
        logger.info(
            f"Single protein provided with {len(ligands)} ligands; "
            "replicating protein for all ligands"
        )
        return proteins * len(ligands)
    raise ValueError(
        f"Number of proteins ({len(proteins)}) does not match number of ligands "
        f"({len(ligands)}). Provide either one protein (auto-replicated) or one "
        "protein per ligand."
    )


def predict_command(args):
    """Execute the predict command."""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            expanded_ligands, ligand_names = expand_ligands(args.ligands, temp_dir)
            matched_proteins = match_proteins_to_ligands(
                args.proteins, expanded_ligands
            )

            predictions = predict_binding_affinity(
                proteins=matched_proteins,
                ligands=expanded_ligands,
                model_checkpoint=args.model_checkpoint,
                batch_size=args.max_batch_size,
            )

            # Write results to CSV
            with open(args.output_csv, "w", newline="") as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(["protein", "ligand", "delta_g"])
                for i in range(len(predictions)):
                    csvwriter.writerow(
                        [args.proteins[0] if len(args.proteins) == 1 else args.proteins[i],
                         ligand_names[i],
                         predictions[i].squeeze()]
                    )

            print(f"Predictions saved to {args.output_csv}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="docktdeep",
        description="Deep learning model for protein-ligand binding affinity prediction",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Predict command
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict protein-ligand binding affinities",
        description="""
Predict protein-ligand binding affinities using docktdeep.

Example usage:
    # basic usage with single files:
    docktdeep predict --proteins protein.pdb --ligands ligand.mol2

    # multiple pairs:
    docktdeep predict --proteins protein1.pdb protein2.pdb --ligands ligand1.mol2 ligand2.mol2

    # single protein with multiple ligands (protein auto-replicated):
    docktdeep predict --proteins protein.pdb --ligands ligand1.mol2 ligand2.mol2 ligand3.mol2

    # multi-mol2 file (e.g., docking output with multiple poses):
    docktdeep predict --proteins protein.pdb --ligands docked_poses.mol2

Requirements:
    - Provide one protein per ligand, or a single protein (auto-replicated)
    - Protein files should be in PDB format (.pdb)
    - Ligand files should be in PDB or MOL2 format (.pdb, .mol2)
    - Multi-mol2 files are automatically split into individual ligands

Output:
    A CSV file with the results. Predictions are given in kcal/mol.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    predict_parser.add_argument(
        "--proteins",
        nargs="+",
        required=True,
        help="Path(s) to the protein file(s) (.pdb). A single protein is auto-replicated to match multiple ligands.",
    )
    predict_parser.add_argument(
        "--ligands",
        nargs="+",
        required=True,
        help="Path(s) to the ligand file(s) (.pdb, .mol2). Multi-mol2 files are automatically expanded.",
    )
    predict_parser.add_argument(
        "--max-batch-size", type=int, default=32, help="Max batch size for inference."
    )
    predict_parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to a custom model checkpoint (.ckpt). If not provided, uses the default model.",
    )
    predict_parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions.csv",
        help="Path to the output CSV file.",
    )

    predict_parser.set_defaults(func=predict_command)

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute the command
    args.func(args)


if __name__ == "__main__":
    main()
