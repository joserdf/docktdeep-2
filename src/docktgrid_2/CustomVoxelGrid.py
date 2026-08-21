import torch

from docktgrid.config import DTYPE
from docktgrid import VoxelGrid


class CustomVoxelGrid(VoxelGrid):
    """VoxelGrid com occupancies extras e device de voxelizacao configuravel.

    O docktgrid original aloca tudo em ``docktgrid.config.DEVICE`` (cuda quando
    disponivel). Isso quebra os workers do DataLoader sob start method 'fork':
    ``Cannot re-initialize CUDA in forked subprocess``. Como grid points e
    coords da molecula ja sao CPU, voxelizar em CPU no worker e mover o batch
    para a GPU no laco de treino e o caminho natural -- e paraleliza pelos
    num_workers em vez de serializar numa unica GPU.
    """

    def __init__(self, *args, device="cpu", **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device(device)

    def voxelize(self, molecule, out=None, channels=None, requires_grad=False):
        """Igual ao da classe base, mas alocando em ``self.device``.

        Nao da para delegar via ``super().voxelize(out=...)``: a base passa o
        ``out`` recebido por ``torch.as_tensor(out, DTYPE, DEVICE)``, o que o
        moveria de volta para a GPU.
        """
        if out is None:
            out = torch.zeros(
                self.shape, dtype=DTYPE, device=self.device, requires_grad=requires_grad
            )
        else:
            out = torch.as_tensor(out, dtype=DTYPE, device=self.device)
            if out.shape != self.shape:
                raise ValueError(f"`out` shape must be == {self.shape}, got {out.shape}")

        if channels is None:
            channels = self.get_channels_mask(molecule)
        else:
            cshape = (self.num_channels, molecule.n_atoms)
            if channels.shape != cshape:
                raise ValueError(f"`channels` shape must be == {cshape}, got {channels.shape}")
        channels = channels.to(self.device)

        self.occupancy_func(molecule, out, channels)
        return out.view(self.shape)

    def get_occupancy_func(self, occ):
        if occ == 'gaussian':
            return self._voxelize_gaussian
        elif occ == 'gaussian_amax':
            return self._voxelize_gaussian_amax
        elif occ == 'vdw2':
            return self._voxelize_vdw2
        elif occ == 'vdw2_matmul':
            return self._voxelize_vdw2_matmul
        else:
            return super().get_occupancy_func(occ)

    @torch.no_grad()
    def _voxelize_vdw(self, molecule, out, channels) -> None:
        """Igual ao da base, mas em ``self.device``.

        A base manda coords/grid/raios para ``docktgrid.config.DEVICE`` (cuda)
        enquanto o nosso ``out`` esta em ``self.device``, e o kernel jit falha
        com device mismatch. Esta e a unica occupancy que nao sobrescrevemos
        por outro motivo, entao precisa da copia.
        """
        points = self.grid.points
        center = molecule.ligand_center
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        out = out.view(channels.shape[0], grid[x].shape[0])

        self._calc_vdw_occupancies(
            out,
            channels,
            molecule.coords[x].to(self.device),
            molecule.coords[y].to(self.device),
            molecule.coords[z].to(self.device),
            grid[x].to(self.device),
            grid[y].to(self.device),
            grid[z].to(self.device),
            molecule.vdw_radii.to(self.device),
        )
        
    @torch.no_grad()
    def _voxelize_gaussian(self, molecule, out, channels) -> None:
        points = self.grid.points
        center = molecule.ligand_center
        # translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        # channels is now a tensor of property values per channel/atom
        self._calc_gaussian_occupancies(
            out,
            channels.to(self.device),
            molecule.coords[x].to(self.device),
            molecule.coords[y].to(self.device),
            molecule.coords[z].to(self.device),
            grid[x].to(self.device),
            grid[y].to(self.device),
            grid[z].to(self.device),
            molecule.vdw_radii.to(self.device),
        )
    
    @staticmethod
    @torch.jit.script
    def _calc_gaussian_occupancies(
        out: torch.Tensor,           # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,      # property values per channel/atom, shape (n_channels, n_atoms)
        ax: torch.Tensor,            # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,            # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,            # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,            # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,            # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,            # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,          # vdw radii of atoms, shape (n_atoms,)
    ):
        # Compute squared distances (n_points, n_atoms) without sqrt for speed
        dx = ax - px
        dy = ay - py
        dz = az - pz
        
        dist2 = dx * dx + dy * dy + dz * dz

        # Protect against zero vdW radii and prepare widths (1, n_atoms)
        vdw = vdws.clamp_min(1e-6).unsqueeze(0)
        vdw2 = vdw * vdw

        # gaussian occupancy per atom at each grid point (n_points, n_atoms)
        occs = torch.exp(-dist2 / vdw2)
        # change dtype of channels to match occs for matmul
        channels = channels.to(occs.dtype)

        torch.matmul(channels, occs.t(), out=out)
        out.clamp_min_(0.0)

    @torch.no_grad()
    def _voxelize_gaussian_amax(self, molecule, out, channels) -> None:
        points = self.grid.points
        center = molecule.ligand_center
        # translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        # channels is now a tensor of property values per channel/atom
        self._calc_gaussian_amax_occupancies(
            out,
            channels.to(self.device),
            molecule.coords[x].to(self.device),
            molecule.coords[y].to(self.device),
            molecule.coords[z].to(self.device),
            grid[x].to(self.device),
            grid[y].to(self.device),
            grid[z].to(self.device),
            molecule.vdw_radii.to(self.device),
        )
    
    @staticmethod
    @torch.jit.script
    def _calc_gaussian_amax_occupancies(
        out: torch.Tensor,           # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,      # property values per channel/atom, shape (n_channels, n_atoms)
        ax: torch.Tensor,            # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,            # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,            # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,            # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,            # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,            # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,          # vdw radii of atoms, shape (n_atoms,)
    ):
        # Compute squared distances (n_points, n_atoms) without sqrt for speed
        dx = ax - px
        dy = ay - py
        dz = az - pz
        
        dist2 = dx * dx + dy * dy + dz * dz

        # Protect against zero vdW radii and prepare widths (1, n_atoms)
        vdw = vdws.clamp_min(1e-6).unsqueeze(0)
        vdw2 = vdw * vdw

        # gaussian occupancy per atom at each grid point (n_points, n_atoms)
        occs = torch.exp(-dist2 / vdw2)
        # change dtype of channels to match occs for matmul
        channels = channels.to(occs.dtype)

        weighted = channels.unsqueeze(1) * occs.unsqueeze(0)  # (n_channels, n_points, n_atoms)
        n_at = weighted.size(2)

        if n_at == 0:
            out.zero_()
        else:
            # weighted: (n_channels, n_points, n_atoms)
            torch.amax(weighted, dim=2, out=out)
        out.clamp_min_(0.0)


    @torch.no_grad()
    def _voxelize_vdw2(self, molecule, out, channels) -> None:
        points = self.grid.points
        center = molecule.ligand_center
        # translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        # channels is now a tensor of property values per channel/atom
        self._calc_vdw2_occupancies(
            out,
            channels.to(self.device),
            molecule.coords[x].to(self.device),
            molecule.coords[y].to(self.device),
            molecule.coords[z].to(self.device),
            grid[x].to(self.device),
            grid[y].to(self.device),
            grid[z].to(self.device),
            molecule.vdw_radii.to(self.device),
        )
    
    @staticmethod
    @torch.jit.script
    def _calc_vdw2_occupancies(
        out: torch.Tensor,           # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,      # property values per channel/atom, shape (n_channels, n_atoms)
        ax: torch.Tensor,            # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,            # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,            # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,            # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,            # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,            # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,          # vdw radii of atoms, shape (n_atoms,)
    ):
        # Compute squared distances (n_points, n_atoms) and avoid sqrt
        dx = ax - px
        dy = ay - py
        dz = az - pz

        dist2 = dx * dx + dy * dy + dz * dz
        dist2 = dist2.clamp_min(1e-12)  # avoid division by zero

        # Use (vdw / dist)^12 = vdw^12 / dist^12 = vdw^12 / (dist2^6)
        vdw12 = vdws.pow(12).unsqueeze(0)  # (1, n_atoms)
        dist12 = dist2.pow(6)               # (n_points, n_atoms)
        ratio12 = vdw12 / dist12
        occs = 1.0 - torch.exp(-ratio12)   # (n_points, n_atoms)

        # Multiply occupancies by channel values instead of boolean masking:
        # channels: (n_channels, n_atoms)
        # occs:     (n_points,  n_atoms)
        channels = channels.to(occs.dtype)
        weighted = channels.unsqueeze(1) * occs.unsqueeze(0)  # (n_channels, n_points, n_atoms)
        n_at = weighted.size(2)

        if n_at == 0:
            out.zero_()
        else:
            # weighted: (n_channels, n_points, n_atoms)
            torch.amax(weighted, dim=2, out=out)
        out.clamp_min_(0.0)

    @torch.no_grad()
    def _voxelize_vdw2_matmul(self, molecule, out, channels) -> None:
        points = self.grid.points
        center = molecule.ligand_center
        # translate grid points and reshape for proper broadcasting
        grid = [(u + v).unsqueeze(-1) for u, v in zip(points, center)]

        x, y, z = 0, 1, 2
        # reshape to n_channels, n_points
        out = out.view(channels.shape[0], grid[x].shape[0])

        # channels is now a tensor of property values per channel/atom
        self._calc_vdw2_matmul_occupancies(
            out,
            channels.to(self.device),
            molecule.coords[x].to(self.device),
            molecule.coords[y].to(self.device),
            molecule.coords[z].to(self.device),
            grid[x].to(self.device),
            grid[y].to(self.device),
            grid[z].to(self.device),
            molecule.vdw_radii.to(self.device),
        )
    
    @staticmethod
    @torch.jit.script
    def _calc_vdw2_matmul_occupancies(
        out: torch.Tensor,           # output tensor, shape (n_channels, n_points)
        channels: torch.Tensor,      # property values per channel/atom, shape (n_channels, n_atoms)
        ax: torch.Tensor,            # x coords of atoms, shape (n_atoms,)
        ay: torch.Tensor,            # y coords of atoms, shape (n_atoms,)
        az: torch.Tensor,            # z coords of atoms, shape (n_atoms,)
        px: torch.Tensor,            # x coords of grid points, shape (n_points, 1)
        py: torch.Tensor,            # y coords of grid points, shape (n_points, 1)
        pz: torch.Tensor,            # z coords of grid points, shape (n_points, 1)
        vdws: torch.Tensor,          # vdw radii of atoms, shape (n_atoms,)
    ):
        # Compute squared distances (n_points, n_atoms) and avoid sqrt
        dx = ax - px
        dy = ay - py
        dz = az - pz

        dist2 = dx * dx + dy * dy + dz * dz
        dist2 = dist2.clamp_min(1e-12)  # avoid division by zero

        # Use (vdw / dist)^12 = vdw^12 / dist^12 = vdw^12 / (dist2^6)
        vdw12 = vdws.pow(12).unsqueeze(0)  # (1, n_atoms)
        dist12 = dist2.pow(6)               # (n_points, n_atoms)
        ratio12 = vdw12 / dist12
        occs = 1.0 - torch.exp(-ratio12)   # (n_points, n_atoms)

        # Multiply occupancies by channel values instead of boolean masking:
        # channels: (n_channels, n_atoms)
        # occs:     (n_points,  n_atoms)
        channels = channels.to(occs.dtype)
        torch.matmul(channels, occs.t(), out=out)
        out.clamp_min_(0.0)

if __name__ == "__main__":
    import time

    from docktgrid_2.NewViewLigProt import NewViewLigProt
    from docktgrid.molecule import MolecularComplex

    molecule = MolecularComplex(
        "6rnt_protein.pdb", "6rnt_ligand.pdb", path="tests/data/"
    )

    occs = ["vdw", "vdw2", "vdw2_matmul", "gaussian", "gaussian_amax"]

    num_tests = 10000

    for occ in occs:
        vox = CustomVoxelGrid([NewViewLigProt()], 1.0, [24.0, 24.0, 24.0], occupancy=occ)

        times = []
        for _ in range(num_tests):
            stime = time.time_ns()
            grid = vox.voxelize(molecule)
            etime = time.time_ns()
            times.append((etime - stime) / 1e6)
        avg_time = sum(times) / num_tests
        std_time = (sum((t - avg_time) ** 2 for t in times) / num_tests) ** 0.5
        print(f"<{occ} voxelization avg time over {num_tests} runs: {avg_time:.1f}ms ± {std_time:.1f}ms>", end=" ", flush=True)
        print(grid.shape)

    #torch.save(grid, "test_voxel_grid.pt")
    #grid_loaded = torch.load("test_voxel_grid.pt")
    #print(torch.allclose(grid, grid_loaded))