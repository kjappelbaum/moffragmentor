# -*- coding: utf-8 -*-
"""Describing the organic building blocks, i.e., linkers"""
from copy import deepcopy

from pymatgen.core import Molecule
from rdkit.Chem import AllChem

from .sbu import SBU


class Linker(SBU):
    pass


def _get_edge_dict_from_rdkit_mol(mol):
    edges = {}
    for bond in mol.GetBonds():
        edges[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = None
    return edges


def _make_mol_from_rdkit_mol(mol):
    """Takes the first conformer"""
    molecule = deepcopy(mol)
    AllChem.EmbedMolecule(molecule)  # pylint:disable=no-member
    # not clear to me why pylint complains
    conf = molecule.GetConformers()
    positions = conf[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Molecule(symbols, positions)
