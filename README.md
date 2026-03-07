# DockTDeep

Preprint: **"Data-centric training enables meaningful interaction learning in protein–ligand binding affinity prediction."** [ChemRXiv.](https://chemrxiv.org/engage/chemrxiv/article-details/68a52850728bf9025e40d9e4)

## 💾 Installation

> [!TIP]
> Always use a virtual environment to manage dependencies.
> 
> ```bash
> python -m venv .venv
> source .venv/bin/activate
> ```

### Using pip

Quick setup for inference. Install the package directly from PyPI:

```bash
pip install docktdeep
```

### Using Docker

For a containerized setup that requires no Python environment:

```bash
# clone the repository (needed for the model checkpoint)
git clone https://github.com/gmmsb-lncc/docktdeep.git
cd docktdeep

# build the image and install the wrapper script
./install.sh
```

This installs a `docktdeep` command system-wide. Run it from any directory containing your data files:

```bash
docktdeep --proteins protein.pdb --ligands ligand.mol2 --output-csv results.csv

# with GPU support
docktdeep --gpu --proteins protein.pdb --ligands ligand.mol2 --output-csv results.csv
```

> [!NOTE]
> The wrapper automatically mounts the current working directory into the container.
> All input files must be in the current directory and output files will be written there.


## 🚀 Quick start

### Basic usage

Predict binding affinities for protein-ligand pairs _(predictions are given in kcal/mol)_.

```bash
# single protein-ligand pair
docktdeep predict --proteins protein.pdb --ligands ligand.pdb --output-csv results.csv

# multiple pairs
docktdeep predict \
    --proteins protein1.pdb protein2.pdb \
    --ligands ligand1.pdb ligand2.pdb \
    --output-csv results.csv \
    --max-batch-size 16

# single protein with multiple ligands (protein auto-replicated)
docktdeep predict \
    --proteins protein.pdb \
    --ligands ligand1.mol2 ligand2.mol2 ligand3.mol2 \
    --output-csv results.csv

# multi-mol2 file (e.g., docking output with multiple poses)
docktdeep predict \
    --proteins protein.pdb \
    --ligands docked_poses.mol2 \
    --output-csv results.csv

# options available in help
docktdeep predict --help
```

> [!TIP]
> When using a single protein with multiple ligands, the protein is automatically replicated — no need to repeat the protein path.
> Multi-mol2 files (common output from docking programs like AutoDock Vina or GOLD) are automatically split into individual molecules.

> [!TIP]
> Use shell globbing patterns to process multiple files efficiently.
> ```bash
> # using regex expansion
> docktdeep predict \
>    --proteins $(ls path/to/proteins/*_protein.pdb) \
>    --ligands $(ls path/to/ligands/*_ligand.pdb)
>
> # another example using find command for more complex patterns
> docktdeep predict \
>    --proteins $(find /data/complexes -name "*_protein_prep.pdb" | sort) \
>    --ligands $(find /data/complexes -name "*_ligand_rnum.pdb" | sort)
> ```


## ⚙️ Development setup

For development and training custom models:

```bash
# clone the repository
git clone https://github.com/gmmsb-lncc/docktdeep.git
cd docktdeep

# create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# install deps
python -m pip install -r requirements.txt

# run tests to verify installation
python -m pytest tests/
```

### Training models

Initialize a new aim repository for tracking experiments:

```bash
aim init

# to start the aim server
aim server
```

To see all available training options:

```bash
python train.py --help
```


Train a model with optimized hyperparameters:

```bash
python train.py \
    --model Baseline \
    --experiment experiment-name \
    --depthwise-convs \
    --adaptive-pooling \
    --optim AdamW \
    --max-epochs 1500 \
    --batch-size 64 \
    --lr 0.00087469 \
    --beta1 0.25693012 \
    --eps 0.00032933 \
    --dropout 0.25348994 \
    --wdecay 0.0000169 \
    --molecular-dropout 0.06 \
    --molecular-dropout-unit complex \
    --random-rotation \
    --dataframe-path path/to/dataframe.csv \
    --root-dir path/to/data/PDBbind2020 \
    --ligand-path-pattern "{c}/{c}_ligand_rnum.pdb" \
    --protein-path-pattern "{c}/{c}_protein_prep.pdb" \
    --split-column random_split
```



## 📝 Citation

If you use DockTDeep in your research, please cite:

```bibtex
@article{dasilva2025docktdeep,
  title={Data-centric training enables meaningful interaction learning in protein--ligand binding affinity prediction},
  author={da Silva, Matheus M. P. and Vidal, Lincon and Guedes, Isabella and de Magalh{\~a}es, Camila and Cust{\'o}dio, F{\'a}bio and Dardenne, Laurent},
  year={2025}
}
```

### Related
- **DockTGrid: a python package for generating deep learning-ready voxel grids of molecular complexes.** [GitHub](https://github.com/gmmsb-lncc/docktgrid).
