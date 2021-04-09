conda create -n lab python=3.8 -y
source activate lab
conda install jupyterlab=3 nglview -c conda-forge -y
conda install openbabel rdkit ase click
pip install "pymatgen>=2021.1,<2022"
pip install git+https://github.com/skearnes/rdkit-utils.git
