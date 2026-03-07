"""Tests for CLI preprocessing functions (multi-mol2 expansion and protein replication)."""

import os
import tempfile

import pytest

from docktdeep.cli import expand_ligands, match_proteins_to_ligands

SINGLE_MOL2 = """\
@<TRIPOS>MOLECULE
mol_1
 3 2 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG         0.0000
      2 C2          1.5400    0.0000    0.0000 C.3     1  LIG         0.0000
      3 C3          2.3100    1.3300    0.0000 C.3     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    1
     2     2     3    1
"""

MULTI_MOL2 = """\
@<TRIPOS>MOLECULE
mol_1
 2 1 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          0.0000    0.0000    0.0000 C.3     1  LIG         0.0000
      2 C2          1.5400    0.0000    0.0000 C.3     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    1
@<TRIPOS>MOLECULE
mol_2
 2 1 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          3.0000    0.0000    0.0000 C.3     1  LIG         0.0000
      2 C2          4.5400    0.0000    0.0000 C.3     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    1
@<TRIPOS>MOLECULE
mol_3
 2 1 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1 C1          6.0000    0.0000    0.0000 C.3     1  LIG         0.0000
      2 C2          7.5400    0.0000    0.0000 C.3     1  LIG         0.0000
@<TRIPOS>BOND
     1     1     2    1
"""


class TestExpandLigands:
    def test_single_mol2_returns_original_path(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as temp_dir:
            mol2_path = os.path.join(data_dir, "ligand.mol2")
            with open(mol2_path, "w") as f:
                f.write(SINGLE_MOL2)

            paths, names = expand_ligands([mol2_path], temp_dir)

            assert len(paths) == 1
            assert paths[0] == mol2_path
            assert names[0] == mol2_path

    def test_multi_mol2_splits_into_individual_files(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as temp_dir:
            mol2_path = os.path.join(data_dir, "multi.mol2")
            with open(mol2_path, "w") as f:
                f.write(MULTI_MOL2)

            paths, names = expand_ligands([mol2_path], temp_dir)

            assert len(paths) == 3
            for p in paths:
                assert os.path.exists(p)

    def test_multi_mol2_display_names_include_index(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as temp_dir:
            mol2_path = os.path.join(data_dir, "multi.mol2")
            with open(mol2_path, "w") as f:
                f.write(MULTI_MOL2)

            _, names = expand_ligands([mol2_path], temp_dir)

            assert names == [f"{mol2_path}:0", f"{mol2_path}:1", f"{mol2_path}:2"]

    def test_multi_mol2_temp_files_contain_valid_content(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as temp_dir:
            mol2_path = os.path.join(data_dir, "multi.mol2")
            with open(mol2_path, "w") as f:
                f.write(MULTI_MOL2)

            paths, _ = expand_ligands([mol2_path], temp_dir)

            for p in paths:
                content = open(p).read()
                assert "@<TRIPOS>MOLECULE" in content
                assert "@<TRIPOS>ATOM" in content

    def test_pdb_ligand_passes_through(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            paths, names = expand_ligands(["ligand.pdb"], temp_dir)

            assert paths == ["ligand.pdb"]
            assert names == ["ligand.pdb"]

    def test_mixed_inputs(self):
        with tempfile.TemporaryDirectory() as data_dir, tempfile.TemporaryDirectory() as temp_dir:
            single_path = os.path.join(data_dir, "single.mol2")
            with open(single_path, "w") as f:
                f.write(SINGLE_MOL2)

            multi_path = os.path.join(data_dir, "multi.mol2")
            with open(multi_path, "w") as f:
                f.write(MULTI_MOL2)

            paths, names = expand_ligands(
                ["ligand.pdb", single_path, multi_path], temp_dir
            )

            # 1 pdb + 1 single mol2 + 3 from multi mol2 = 5
            assert len(paths) == 5
            assert names[0] == "ligand.pdb"
            assert names[1] == single_path
            assert names[2] == f"{multi_path}:0"
            assert names[3] == f"{multi_path}:1"
            assert names[4] == f"{multi_path}:2"


class TestMatchProteinsToLigands:
    def test_equal_lengths_returns_unchanged(self):
        proteins = ["a.pdb", "b.pdb"]
        ligands = ["x.mol2", "y.mol2"]
        result = match_proteins_to_ligands(proteins, ligands)
        assert result == ["a.pdb", "b.pdb"]

    def test_single_protein_replicates(self):
        result = match_proteins_to_ligands(["a.pdb"], ["x.mol2", "y.mol2", "z.mol2"])
        assert result == ["a.pdb", "a.pdb", "a.pdb"]

    def test_single_protein_single_ligand(self):
        result = match_proteins_to_ligands(["a.pdb"], ["x.mol2"])
        assert result == ["a.pdb"]

    def test_mismatched_raises_error(self):
        with pytest.raises(ValueError, match="does not match"):
            match_proteins_to_ligands(["a.pdb", "b.pdb"], ["x.mol2", "y.mol2", "z.mol2"])
