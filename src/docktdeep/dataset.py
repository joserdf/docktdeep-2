import copy
import hashlib
import json
import os
import pickle
from typing import Optional

import docktgrid
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from docktgrid.transforms import RandomRotation
from torch.utils.data import Dataset

from .transforms import MolecularDropout, Random90DegreesRotation


class PDBbind(pl.LightningDataModule):
    def __init__(
        self,
        voxel_grid: docktgrid.VoxelGrid,
        batch_size: int,
        dataframe_path: str = "/home/mpds/data/pdbbind2020-refined-prepared/index.csv",
        transforms=None,
        molecular_dropout: float = 0.0,
        molecular_dropout_unit: str = "",
        root_dir: str = "",
        experiment: str = "",
        protein_path_pattern: str = "{c}_protein_prep.pdb.pkl",
        ligand_path_pattern: str = "{c}_ligand_rnum.pdb.pkl",
        split_column: str = "random_split",  # Column name in the dataframe used to select train/validation/test splits
        target_column: str = "pki",  # Regression label: pK (pKd/pKi/pIC50). See configs/grid/README.md.
        merge_val_test: bool = False,
        num_workers: int = 4,
        use_esm2: bool = False,
        use_chemberta: bool = False,
        embeddings_dir: str = "data/embeddings",
        esm2_model: str = "esm2-650M",
        no_cnn: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.voxel_grid = voxel_grid
        self.batch_size = batch_size
        self.df_path = dataframe_path
        self.transforms = transforms
        self.molecular_dropout = molecular_dropout
        self.molecular_dropout_unit = molecular_dropout_unit
        self.root_dir = root_dir
        self.experiment = experiment
        self.protein_path_pattern = protein_path_pattern
        self.ligand_path_pattern = ligand_path_pattern
        self.split_column = split_column
        self.target_column = target_column
        self.merge_val_test = merge_val_test
        self.num_workers = num_workers
        self.use_esm2 = use_esm2
        self.use_chemberta = use_chemberta
        self.embeddings_dir = embeddings_dir
        self.esm2_model = esm2_model
        self.no_cnn = no_cnn

    @staticmethod
    def add_specific_args(parent_parser):
        # fmt: off
        parser = parent_parser.add_argument_group("Data args")
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--num-workers", type=int, default=4)
        parser.add_argument("--vox-size", type=float, default=1.0)
        parser.add_argument("--box-dims", type=list, default=[24.0, 24.0, 24.0])
        parser.add_argument("--voxel-device", type=str, default="cpu", help="Device for voxelization. Keep 'cpu': DataLoader workers under start method 'fork' cannot re-initialize CUDA. 'cuda' only makes sense with --num-workers 0.")
        parser.add_argument("--view", nargs="+", type=str, default=["VolumeView", "BasicView"])
        parser.add_argument("--occupancy", type=str, default="vdw", help="Type of occupancy to use in voxelization: vdw or gaussian")
        parser.add_argument("--random-rotation", action="store_true", default=False)
        parser.add_argument("--rotation-90-degrees", action="store_true", default=False)
        parser.add_argument("--molecular-dropout", type=float, default=0.0)
        parser.add_argument("--molecular-dropout-unit", type=str, default="protein", help="protein, ligand, or complex")
        parser.add_argument("--protein-path-pattern", type=str, default="{c}_protein_prep.pdb.pkl", help="Path pattern for protein files, use {c} as placeholder for PDB ID")
        parser.add_argument("--ligand-path-pattern", type=str, default="{c}_ligand_rnum.pdb.pkl", help="Path pattern for ligand files, use {c} as placeholder for PDB ID")
        parser.add_argument("--split-column", type=str, default="random_split", help="Column name in the dataframe used to select train/validation/test splits")
        parser.add_argument("--target-column", type=str, default="pki", help="Regression label column: 'pki' (pK units, default) or 'delta_g' (kcal/mol). delta_g = -RT*ln(10)*pK, so they differ only by a factor of -1.364.")
        parser.add_argument("--merge-val-test", action="store_true", default=False, help="Whether to merge validation and test sets for evaluation")
        parser.add_argument("--use-esm2", action="store_true", default=False, help="Condition on frozen ESM-2 protein embeddings (factor A).")
        parser.add_argument("--use-chemberta", action="store_true", default=False, help="Condition on frozen ChemBERTa ligand embeddings (factor B).")
        parser.add_argument("--embeddings-dir", type=str, default="data/embeddings", help="Dir with cached embeddings and mapping jsons.")
        parser.add_argument("--esm2-model", type=str, default="esm2-650M", help="ESM-2 model key used for cached protein embeddings.")
        # fmt: on

        return parent_parser

    def setup(self, stage) -> None:
        self.df = pd.read_csv(self.df_path, low_memory=False)  # [:200]

        self.train_dataset = self._get_dataset("train")
        self.val_dataset = self._get_dataset("validation")
        if not self.merge_val_test:
            self.test_dataset = self._get_dataset("test")

        print(f"Train dataset: {len(self.train_dataset)}")
        print(f"Validation dataset: {len(self.val_dataset)}")
        if not self.merge_val_test:
            print(f"Test dataset: {len(self.test_dataset)}")

    def _split_dataset(self, split: str) -> Dataset:
        if self.split_column not in self.df.columns:
            raise ValueError(f"Split column '{self.split_column}' not found in dataframe.")
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe.")
        if self.split_column == "coreset_v2016":
            if split == "train":
                dataset = self.df[((self.df['random_split'] == "train") 
                                    | (self.df['random_split'] == "validation")
                                    | (self.df['random_split'] == "test")
                                    | (self.df['coreset_v2013'] == True))
                                    & (self.df['coreset_v2016'] == False)
                                    & (self.df['random_split'] != "ERR")]
            else:
                dataset = self.df[(self.df['coreset_v2016'] == True) & (self.df['random_split'] != "ERR")]
        elif self.split_column == "coreset_v2013":
            if split == "train":
                dataset = self.df[((self.df['random_split'] == "train") 
                                    | (self.df['random_split'] == "validation")
                                    | (self.df['random_split'] == "test")
                                    | (self.df['coreset_v2016'] == True))
                                    & (self.df['coreset_v2013'] == False)
                                    & (self.df['random_split'] != "ERR")]
            else:
                dataset = self.df[(self.df['coreset_v2013'] == True) & (self.df['random_split'] != "ERR")]
        elif [False, True] == self.df[self.split_column].unique().tolist():
            if split == "train":
                dataset = self.df[self.df[self.split_column] == False]
            else:
                dataset = self.df[self.df[self.split_column] == True]
        elif {"train", "validation", "test"}.issubset(
                set(self.df[self.split_column].dropna().astype(str).unique())):
            dataset = self.df[self.df[self.split_column] == split]
        else:
            raise ValueError(f"Unexpected values in split column '{self.split_column}'.")
        return dataset

    def _get_dataset(self, split: str):
        dataset = self._split_dataset(split)

        protein_files = [self.protein_path_pattern.format(c=c) for c in dataset.id]
        ligand_files = [self.ligand_path_pattern.format(c=c) for c in dataset.id]

        protein_mols = []
        ligand_mols = []
        labels = []
        sample_ids = []

        # o rotulo e coletado aqui, no mesmo laco que aceita a amostra, para ficar
        # alinhado com protein_mols/ligand_mols/sample_ids quando algum pickle falta
        for protein_file, ligand_file, label in zip(protein_files, ligand_files, dataset[self.target_column].values):
            if os.path.exists(os.path.join(self.root_dir, protein_file)) and os.path.exists(
                os.path.join(self.root_dir, ligand_file)
            ):
                if self.no_cnn:  # placeholders: o grid nunca e construido
                    protein_mols.append(None)
                    ligand_mols.append(None)
                else:
                    protein_mols.append(pickle.load(open(os.path.join(self.root_dir, protein_file), "rb")))
                    ligand_mols.append(pickle.load(open(os.path.join(self.root_dir, ligand_file), "rb")))
                sample_ids.append(str(protein_file.split("_")[0]))
                labels.append(label)

        # exclude atoms outside the box
        for i, ptn in enumerate([] if self.no_cnn else protein_mols):
            radius = np.ceil(np.sqrt(3) * max(self.voxel_grid.shape[1:]) / 2)
            inside_atoms_idx = docktgrid.molparser.extract_binding_pocket(
                ptn.coords, ligand_mols[i].coords.mean(dim=1), radius
            )

            # keep only the atoms inside the binding pocket, rewrite the MolecularData attributes
            ptn.coords = ptn.coords[:, inside_atoms_idx]
            ptn.element_symbols = ptn.element_symbols[inside_atoms_idx]

        e_prot, e_lig = self._load_embeddings(sample_ids)

        # drop samples missing a required embedding when a factor is active
        keep = [True] * len(sample_ids)
        if self.use_esm2:
            keep = [k and (e is not None) for k, e in zip(keep, e_prot)]
        if self.use_chemberta:
            keep = [k and (e is not None) for k, e in zip(keep, e_lig)]
        if not all(keep):
            idx = [i for i, k in enumerate(keep) if k]
            protein_mols = [protein_mols[i] for i in idx]
            ligand_mols = [ligand_mols[i] for i in idx]
            labels = [labels[i] for i in idx]
            sample_ids = [sample_ids[i] for i in idx]
            e_prot = [e_prot[i] for i in idx] if e_prot is not None else None
            e_lig = [e_lig[i] for i in idx] if e_lig is not None else None
            print(f"  ({split}) removidos por embedding ausente: {len(keep) - len(idx)}")

        # apply molecular dropout view
        voxel_grid = self.voxel_grid
        if self.molecular_dropout > 0.0 and split == "train":
            voxel_grid = copy.deepcopy(self.voxel_grid)
            views = [
                MolecularDropout(v, self.molecular_dropout, self.molecular_dropout_unit)
                for v in voxel_grid.views
            ]
            voxel_grid.views = views

        data = VoxelDataset(
            protein_files=protein_mols,
            ligand_files=ligand_mols,
            labels=labels,
            voxel=voxel_grid,
            transform=self.transforms if split == "train" else None,
            molecular_dropout=self.molecular_dropout if split == "train" else 0.0,
            e_prot=e_prot,
            e_lig=e_lig,
            skip_voxel=self.no_cnn,
            ids=sample_ids,
        )

        return data

    def _load_embeddings(self, sample_ids):
        """Return (e_prot_list, e_lig_list) aligned with sample_ids, or (None, None).

        e_prot: per-sample ESM-2 vector (or None for that sample) from the cache.
        e_lig: per-sample ChemBERTa vector (or None for that sample) from the cache.
        Missing entities yield None so the dataloader can still batch the sample.
        """
        if not (self.use_esm2 or self.use_chemberta):
            return None, None

        ed = self.embeddings_dir
        c2s = json.load(open(os.path.join(ed, "complex_to_seq.json"))) if self.use_esm2 else {}
        c2sm = json.load(open(os.path.join(ed, "complex_to_smiles.json"))) if self.use_chemberta else {}

        def _load(path):
            return np.load(path) if path is not None else None

        e_prot = None
        e_lig = None
        if self.use_esm2:
            e_prot = [
                _load(os.path.join(ed, "esm2", self.esm2_model, f"{c2s.get(cid)}.npy") if c2s.get(cid) else None)
                for cid in sample_ids
            ]
        if self.use_chemberta:
            e_lig = [
                _load(os.path.join(ed, "chemberta", f"{hashlib.sha1(c2sm[cid].encode()).hexdigest()[:16]}.npy")
                      if c2sm.get(cid) else None)
                for cid in sample_ids
            ]
        return e_prot, e_lig

    @staticmethod
    def _collate(batch):
        """Collate samples that may carry embeddings (4-tuple) or not (2-tuple).

        Embedding positions that are all None collapse back to None; mixed
        None/array positions are zero-padded (should not happen after filtering).
        """
        if len(batch[0]) == 4:
            voxs = torch.stack([b[0] for b in batch])
            y = torch.stack([b[3] for b in batch])

            def maybe_stack(lst):
                if any(v is not None for v in lst):
                    ref = lst[0]
                    return torch.stack(
                        [torch.as_tensor(v) if v is not None else torch.zeros_like(torch.as_tensor(ref))
                         for v in lst]
                    )
                return None

            return voxs, maybe_stack([b[1] for b in batch]), maybe_stack([b[2] for b in batch]), y
        voxs = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        return voxs, y

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=PDBbind._collate,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=PDBbind._collate,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=PDBbind._collate,
        )


class VoxelDataset(Dataset):
    """Dataset for protein-ligand voxel data (generates voxel grids on-the-fly).

    Protein and ligand files must be in a list of strings or a list of MolecularData
    objects and must appear in the same order.
    """

    def __init__(
        self,
        protein_files: list[str] | list[docktgrid.molparser.MolecularData],
        ligand_files: list[str] | list[docktgrid.molparser.MolecularData],
        labels: list[float],
        voxel: docktgrid.VoxelGrid,
        molparser: docktgrid.molparser.MolecularParser = docktgrid.molparser.MolecularParser(),
        transform: Optional[list[docktgrid.transforms.Transform]] = None,
        molecular_dropout: float = 0.0,
        rng: np.random.Generator = np.random.default_rng(),
        root_dir: str = "",
        e_prot: Optional[list] = None,
        e_lig: Optional[list] = None,
        skip_voxel: bool = False,
        ids: Optional[list[str]] = None,
    ):
        assert len(protein_files) == len(ligand_files), "must have the same length!"
        assert len(protein_files) == len(labels), "must have the same length!"
        if e_prot is not None:
            assert len(e_prot) == len(labels), "e_prot must be aligned with labels!"
        if e_lig is not None:
            assert len(e_lig) == len(labels), "e_lig must be aligned with labels!"
        if ids is not None:
            assert len(ids) == len(labels), "ids must be aligned with labels!"

        self.ptn_files = protein_files
        self.lig_files = ligand_files
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.voxel = voxel
        self.molparser = molparser
        self.root_dir = root_dir
        self.transform = transform
        self.molecular_dropout = molecular_dropout
        self.rng = rng
        self.e_prot = e_prot
        self.e_lig = e_lig
        self.skip_voxel = skip_voxel
        self.ids = ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        if self.skip_voxel:
            # ablacao --no-cnn: o modelo ignora `voxs`, entao nem o complexo nem o
            # grid sao construidos. Devolve um placeholder so para manter a forma
            # da tupla que o _collate espera.
            voxs = torch.zeros(1, dtype=torch.float32)
            return voxs, self.e_prot[idx] if self.e_prot is not None else None, \
                self.e_lig[idx] if self.e_lig is not None else None, self.labels[idx]

        molecule = docktgrid.molecule.MolecularComplex(
            self.ptn_files[idx], self.lig_files[idx], self.molparser, self.root_dir
        )
        label = self.labels[idx]

        # apply random rotation
        for transform in self.transform or []:
            if isinstance(transform, RandomRotation):
                transform(molecule.coords, molecule.ligand_center)

        # apply molecular dropout
        if self.molecular_dropout > 0.0:
            alpha, beta = self.rng.uniform(size=2)
            for v in self.voxel.views:
                v.set_random_nums(alpha, beta)

            if alpha <= self.molecular_dropout:
                label = torch.tensor(0.0, dtype=torch.float32)

        voxs = self.voxel.voxelize(molecule)  # <- voxelization happens here

        # apply random 90 degree rotation
        for transform in self.transform or []:
            if isinstance(transform, Random90DegreesRotation):
                voxs = transform(voxs)

        if self.e_prot is not None or self.e_lig is not None:
            e_prot = self.e_prot[idx] if self.e_prot is not None else None
            e_lig = self.e_lig[idx] if self.e_lig is not None else None
            return voxs, e_prot, e_lig, label
        return voxs, label
