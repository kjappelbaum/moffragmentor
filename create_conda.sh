pythonversion=3.8
nglviewversion=2.7.7
labversion=2.1.5
ipywidgetsversion=7.5.1

conda create -n moffragmentor python=$pythonversion -y
source activate moffragmentor
conda install ipywidgets==$ipywidgetsversion -c conda-forge -y
conda install nglview==$nglviewversion -c conda-forge -y
conda install jupyterlab=$labversion  -y -c conda-forge
conda install openbabel rdkit ase click
pip install "pymatgen>=2021.1,<2022"
jupyter-labextension install @jupyter-widgets/jupyterlab-manager
jupyter-labextension install nglview-js-widgets@$nglviewversion
