conda create -n lab python=3.8 -y
source activate lab
conda install jupyterlab=3 nglview -c conda-forge -y
conda install openbabel rdkit click -c conda-forge
pip install "pymatgen>=2021.1,<2022" EFGs backports.cached_property
pip install git+https://github.com/skearnes/rdkit-utils.git
