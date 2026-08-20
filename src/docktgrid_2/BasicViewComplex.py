import torch
import numpy as np

from docktgrid.view import View
from docktgrid.molecule import MolecularComplex

class BasicViewComplex(View):
    """Basic view.

    The `x` below stands for any other chemical element different from CHONS.

    Protein channels (in this order):
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    Ligand channels:
        carbon, hydrogen, oxygen, nitrogen, sulfur, x*.
    """

    def get_num_channels(self):
        return sum((6, 0, 0))

    def get_channels_names(self):
        chs = ["carbon", "hydrogen", "oxygen", "nitrogen", "sulfur", "other"]
        return (
            [f"{ch}_complex" for ch in chs]
        )

    def get_molecular_complex_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms."""

        channels = {
            0: ["C"],
            1: ["H"],
            2: ["O"],
            3: ["N"],
            4: ["S"],
            5: ["C", "H", "O", "N", "S"],
        }
        nchs = len(channels)

        # get a list of bools representing each atom in each channel
        symbs = molecular_complex.element_symbols
        chs = np.asarray([np.isin(symbs, channels[c]) for c in range(nchs)])

        # invert bools in last channel, since it represents any atom except CHONS
        np.invert(chs[-1], out=chs[-1])

        return torch.from_numpy(chs)

    def get_ligand_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for ligand atoms."""
        return None

    def get_protein_channels(self, molecular_complex: MolecularComplex) -> torch.Tensor:
        """Set of channels for protein atoms."""
        return None

if __name__ == "__main__":
    protein_file = "tests/data/6rnt_protein.pdb"
    ligand_file = "tests/data/6rnt_ligand.pdb"
    mol = MolecularComplex(protein_file, ligand_file)

    custom_view = BasicViewComplex()
    channels = custom_view(mol)
    print(channels.shape)
    print(channels)
