# moffragmentor

> Split a MOF into linkers and nodes (and generator input files for a MOF assembly tool)



## Install

`pip install git+https://github.com/kjappelbaum/moffragmentor.git`

You need to have `openbabel` installed which you can install with `conda install -c conda-forge openbabel`. You will also need the RDKit which can be installed with `conda install -c conda-forge rdkit`.

You can also run `bash create_conda.sh`. Note that you might want to change the name of the name of the environment (defaults to `moffragmentor`).

## How to use

```python
mof = MOF.from_cif('test_files/hkust1.cif')
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

To get the [RCSR code](http://rcsr.anu.edu.au/nets) run


```python
fragments.net_embedding.rcsr_code
```

To get some features for the building blocks, you can use

```python
fragments.linkers[0].descriptors
```
