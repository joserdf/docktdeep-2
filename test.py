from src.docktgrid_2.CustomView import CustomView
from src.docktgrid_2.CustomVoxelGrid import CustomVoxelGrid

views = [CustomView()]

voxel_grid = CustomVoxelGrid(
    vox_size=1.0,
    box_dims=(24.0, 24.0, 24.0),
    views=views,
    occupancy="gaussian",
)

print(voxel_grid.shape)
