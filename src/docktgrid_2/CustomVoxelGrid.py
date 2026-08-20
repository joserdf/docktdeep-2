import torch

from docktgrid.config import DEVICE
from docktgrid import VoxelGrid


class CustomVoxelGrid(VoxelGrid):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            channels.to(DEVICE),
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
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
            channels.to(DEVICE),
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
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
            channels.to(DEVICE),
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
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
            channels.to(DEVICE),
            molecule.coords[x].to(DEVICE),
            molecule.coords[y].to(DEVICE),
            molecule.coords[z].to(DEVICE),
            grid[x].to(DEVICE),
            grid[y].to(DEVICE),
            grid[z].to(DEVICE),
            molecule.vdw_radii.to(DEVICE),
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