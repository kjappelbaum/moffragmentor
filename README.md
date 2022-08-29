
<p align="center">
  <img src="https://github.com/kjappelbaum/moffragmentor/raw/main/docs/source/_static/logo.png" height="100">
</p>
<p align="center">
    <a href="https://github.com/kjappelbaum/moffragmentor/actions?query=workflow%3Apython_package">
        <img alt="Tests" src="https://github.com/kjappelbaum/moffragmentor/actions/workflows/python_package.yml/badge.svg" />
    </a>
    <a href="https://pypi.org/project/moffragmentor">
        <img alt="PyPI" src="https://img.shields.io/pypi/v/moffragmentor" />
    </a>
    <a href="https://pypi.org/project/moffragmentor">
        <img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/moffragmentor" />
    </a>
    <a href="https://github.com/kjappelbaum/moffragmentor/blob/main/LICENSE">
        <img alt="PyPI - License" src="https://img.shields.io/pypi/l/moffragmentor" />
    </a>
    <a href='https://moffragmentor.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/moffragmentor/badge/?version=latest' alt='Documentation Status' />
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
</p>

## 💪 Getting Started

```python
mof = MOF.from_cif('tests/test_files/HKUST-1.cif')
```

Fragment the MOF

```python
fragments = mof.fragment()
```

If you are in a Jupyter notebook you can visualize the components.

```python
fragments.linkers[0].show_molecule()
```

```python
fragments.nodes[0].show_molecule()
```

You can also search PubChem for the building blocks 

```python
fragments.linkers[0].search_pubchem()
```

To get the [RCSR code](http://rcsr.anu.edu.au/nets) run


```python
fragments.net_embedding.rcsr_code
```

## 🚀 Installation

To install the latest release run 

```bash 
pip install moffragmentor
```

To install the development version run 

```bash
pip install git+https://github.com/kjappelbaum/moffragmentor.git
```

You need to have `openbabel` installed which you can install with `conda install -c conda-forge openbabel`. 

As a utility, you can also run `bash create_conda.sh` to create a conda environment with all dependencies.
Note that you might want to change the name of the name of the environment (defaults to `moffragmentor`).


## moffragmentor in the MOF decomposition tool landscape

While moffragmentor has some unique features, it might not be the right tool for your task. Due to some design choices (having a pymatgen representation of all fragments and the net) it is relatively slow in the current implementation (though there are obvious ways for fixing this). 

Therefore, we encourage you to also consider alternative tools: 

- [mofid](https://github.com/snurr-group/mofid)
- [mBUD](https://github.com/gdev7/mBUD)



## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/kjappelbaum/moffragmentor/blob/main/CONTRIBUTING.rst) for more information on getting involved.


### ⚖️ License

The code in this package is licensed under the MIT License.


### 💰 Funding

The research was supported by the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme ([grant agreement 666983, MaGic](https://cordis.europa.eu/project/id/666983)), by the [NCCR-MARVEL](https://www.nccr-marvel.ch/), funded by the Swiss National Science Foundation, and by the Swiss National Science Foundation (SNSF) under Grant 200021_172759.
