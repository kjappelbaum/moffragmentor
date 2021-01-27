# moffragmentor
> Split a MOF into linkers and nodes (and generator input files for a MOF assembly tool)


## Install

`pip install moffragmentor`

## How to use

```python
mof = MOF.from_cif('test_files/hkust1.cif')
```

```python
linkers, nodes = mof.fragment()
```

    /Users/kevinmaikjablonka/opt/miniconda3/envs/pymoffragmentor/lib/python3.8/site-packages/pymatgen/core/structure.py:759: UserWarning: Not all sites have property binding. Missing values are set to None.
      warnings.warn(


```python
linkers[0].show_molecule()
```

```python
nodes[0].show_molecule()
```
