conda create -n moffragmentor python=3.8 -y
source activate moffragmentor
conda install jupyterlab=3 nglview openbabel -c conda-forge -y
pip install -e .
