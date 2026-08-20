import torch

from docktgrid.view import View
from docktgrid.molecule import MolecularComplex

from .Customptable import ptable

class NewViewLigProt(View):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.period_idx = 0
        self.protons_idx = 1
        self.valence_s_idx = 2
        self.valence_p_idx = 3
        self.valence_d_idx = 4
        self.valence_f_idx = 5
        self.props = [ 'period', 'protons', 'valence_s', 'valence_p', 'valence_d', 'valence_f' ]

    def get_num_channels(self):
        return sum((0, 6, 6))

    def get_channels_names(self):
        return (
            [f"{ch}_protein" for ch in self.props]
            + [f"{ch}_ligand" for ch in self.props]
        )

    @staticmethod
    def _get_atomic_numbers(molecular_complex: MolecularComplex):
        return torch.tensor(
            [ptable[a.title()]["num"] for a in molecular_complex.element_symbols],
            dtype=torch.int,
        )

    @staticmethod
    def _get_atomic_periods(molecular_complex: MolecularComplex):
        return torch.tensor(
            [ptable[a.title()]["period"] for a in molecular_complex.element_symbols],
            dtype=torch.int,
        )
    
    @staticmethod
    def _get_valence_electrons(molecular_complex: MolecularComplex):
        valence_list = []
        for a in molecular_complex.element_symbols:
            ptable_entry = ptable[a.title()]
            valence_electrons = [
                ptable_entry["valence_s"],
                ptable_entry["valence_p"],
                ptable_entry["valence_d"],
                ptable_entry["valence_f"],
            ]
            valence_list.append(valence_electrons)
        return torch.tensor(valence_list, dtype=torch.int)

    def get_molecular_complex_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        return None  # not used
    
    def get_protein_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms in the protein only."""
        n_atoms_total = molecular_complex.n_atoms
        n_atoms = n_atoms_total - molecular_complex.n_atoms_ligand

        # create channels with shape (n_channels, n_atoms) to match convention used elsewhere
        n_channels = len(self.props)
        channels = torch.zeros((n_channels, n_atoms_total), dtype=torch.int)

        # load atom properties (as numpy scalars for easy indexing in loop)
        periods = self._get_atomic_periods(molecular_complex).cpu().numpy()
        protons = self._get_atomic_numbers(molecular_complex).cpu().numpy()
        valence_s, valence_p, valence_d, valence_f = (
            self._get_valence_electrons(molecular_complex).cpu().numpy().T
        )

        # fill channels: channel index first, atom index second
        # vectorized assignment for protein atoms (first n_atoms entries)
        channels[self.period_idx, :n_atoms] = torch.tensor(
            periods[:n_atoms], dtype=channels.dtype
        )
        channels[self.protons_idx, :n_atoms] = torch.tensor(
            protons[:n_atoms], dtype=channels.dtype
        )
        channels[self.valence_s_idx, :n_atoms] = torch.tensor(
            valence_s[:n_atoms], dtype=channels.dtype
        )
        channels[self.valence_p_idx, :n_atoms] = torch.tensor(
            valence_p[:n_atoms], dtype=channels.dtype
        )
        channels[self.valence_d_idx, :n_atoms] = torch.tensor(
            valence_d[:n_atoms], dtype=channels.dtype
        )
        channels[self.valence_f_idx, :n_atoms] = torch.tensor(
            valence_f[:n_atoms], dtype=channels.dtype
        )
        return channels

    def get_ligand_channels(
        self, molecular_complex: MolecularComplex
    ) -> torch.Tensor:
        """Set of channels for all atoms in the ligand only."""
        n_atoms_total = molecular_complex.n_atoms
        n_atoms = molecular_complex.n_atoms_ligand

        # create channels with shape (n_channels, n_atoms) to match convention used elsewhere
        n_channels = len(self.props)
        channels = torch.zeros((n_channels, n_atoms_total), dtype=torch.int)

        # load atom properties (as numpy scalars for easy indexing in loop)
        periods = self._get_atomic_periods(molecular_complex).cpu().numpy()
        protons = self._get_atomic_numbers(molecular_complex).cpu().numpy()
        valence_s, valence_p, valence_d, valence_f = (
            self._get_valence_electrons(molecular_complex).cpu().numpy().T
        )

        # fill channels: channel index first, atom index second
        # vectorized assignment for ligand atoms (last n_atoms entries)
        channels[self.period_idx, -n_atoms:] = torch.tensor(
            periods[-n_atoms:], dtype=channels.dtype
        )
        channels[self.protons_idx, -n_atoms:] = torch.tensor(
            protons[-n_atoms:], dtype=channels.dtype
        )
        channels[self.valence_s_idx, -n_atoms:] = torch.tensor(
            valence_s[-n_atoms:], dtype=channels.dtype
        )
        channels[self.valence_p_idx, -n_atoms:] = torch.tensor(
            valence_p[-n_atoms:], dtype=channels.dtype
        )
        channels[self.valence_d_idx, -n_atoms:] = torch.tensor(
            valence_d[-n_atoms:], dtype=channels.dtype
        )
        channels[self.valence_f_idx, -n_atoms:] = torch.tensor(
            valence_f[-n_atoms:], dtype=channels.dtype
        )
        return channels
    
if __name__ == "__main__":
    protein_file = "tests/data/6rnt_protein.pdb"
    ligand_file = "tests/data/6rnt_ligand.pdb"
    mol = MolecularComplex(protein_file, ligand_file)

    custom_view = NewViewLigProt()
    channels = custom_view(mol)
    print(channels.shape)
    print(channels)
