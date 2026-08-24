import numpy as np
import torch
from docktgrid import VoxelGrid
from docktgrid.view import VolumeView

from src.docktdeep.dataset import VoxelDataset
from src.docktdeep.transforms import MolecularDropout

PROTEIN_CHANNEL = 1  # VolumeView: ['complex_volume', 'protein_volume', 'ligand_volume']
LIGAND_CHANNEL = 2


def build_dataset(molecular_unit, molecular_dropout_embeddings, seed=0, p=1.0):
    """Um unico complexo, descartado sempre (p=1.0) ou nunca (p ~ 0).

    Reproduz o caminho que a grade usa: as views do voxel embrulhadas em
    MolecularDropout (como PDBbind._get_dataset faz) e o VoxelDataset sorteando
    alpha/beta e repassando para elas.
    """
    view = MolecularDropout(VolumeView(), p=p, molecular_unit=molecular_unit)
    voxel = VoxelGrid(views=[view], vox_size=1.0, box_dims=[12.0, 12.0, 12.0])

    return VoxelDataset(
        protein_files=["6rnt_protein.pdb"],
        ligand_files=["6rnt_ligand.pdb"],
        labels=[7.5],
        voxel=voxel,
        root_dir="tests/data/",
        molecular_dropout=p,
        molecular_dropout_unit=molecular_unit,
        molecular_dropout_embeddings=molecular_dropout_embeddings,
        rng=np.random.default_rng(seed),
        e_prot=[np.ones(4, dtype=np.float32)],
        e_lig=[np.ones(3, dtype=np.float32)],
    )


def test_keeps_both_embeddings_when_flag_is_off():
    dataset = build_dataset("protein", molecular_dropout_embeddings=False)

    voxs, e_prot, e_lig, label = dataset[0]

    assert not voxs[PROTEIN_CHANNEL].any()  # a proteina saiu do voxel
    assert label == torch.tensor(0.0)  # e o rotulo foi zerado
    assert e_prot.all()  # mas o embedding dela continua intacto
    assert e_lig.all()


def test_masks_protein_embedding_when_protein_is_dropped():
    dataset = build_dataset("protein", molecular_dropout_embeddings=True)

    voxs, e_prot, e_lig, _ = dataset[0]

    assert not voxs[PROTEIN_CHANNEL].any()
    assert not e_prot.any()
    assert e_lig.all()  # o parceiro que ficou nao e tocado
    assert dataset.e_prot[0].all()  # o cache compartilhado nao e mutado


def test_masks_ligand_embedding_when_ligand_is_dropped():
    dataset = build_dataset("ligand", molecular_dropout_embeddings=True)

    voxs, e_prot, e_lig, _ = dataset[0]

    assert not voxs[LIGAND_CHANNEL].any()
    assert not e_lig.any()
    assert e_prot.all()
    assert dataset.e_lig[0].all()


def test_masked_embedding_follows_the_same_draw_as_the_voxel():
    """Com unit 'complex' o beta escolhe o parceiro; embedding tem que concordar."""
    seen = set()
    for seed in range(10):
        dataset = build_dataset("complex", molecular_dropout_embeddings=True, seed=seed)

        voxs, e_prot, e_lig, _ = dataset[0]

        protein_dropped = not voxs[PROTEIN_CHANNEL].any()
        ligand_dropped = not voxs[LIGAND_CHANNEL].any()
        assert protein_dropped != ligand_dropped
        assert (not e_prot.any()) == protein_dropped
        assert (not e_lig.any()) == ligand_dropped
        seen.add("protein" if protein_dropped else "ligand")

    assert seen == {"protein", "ligand"}  # as duas escolhas de beta foram exercidas


def test_keeps_embeddings_on_samples_the_dropout_did_not_hit():
    dataset = build_dataset("complex", molecular_dropout_embeddings=True, p=1e-9)

    voxs, e_prot, e_lig, label = dataset[0]

    assert voxs[PROTEIN_CHANNEL].any()  # nada saiu do voxel
    assert voxs[LIGAND_CHANNEL].any()
    assert e_prot.all()
    assert e_lig.all()
    assert label == torch.tensor(7.5)
