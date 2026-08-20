import torch

from docktgrid.view import View
from docktgrid.molecule import MolecularComplex

class VolumeViewComplex(View):
    """Default volume channel sets.

    This view includes all atoms from either protein, ligand or protein-ligand complex
    in a single channel.
    """

    def get_num_channels(self):
        return sum((1, 0, 0))

    def get_channels_names(self):
        return ["complex_volume"]

    def get_molecular_complex_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        return torch.ones((1, molecular_complex.n_atoms), dtype=torch.bool)

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        return None

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        return None

if __name__ == "__main__":
    protein_file = "tests/data/6rnt_protein.pdb"
    ligand_file = "tests/data/6rnt_ligand.pdb"
    mol = MolecularComplex(protein_file, ligand_file)

    custom_view = VolumeViewComplex()
    channels = custom_view(mol)
    print(channels.shape)
    print(channels)
