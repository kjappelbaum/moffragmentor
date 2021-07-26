# -*- coding: utf-8 -*-
from copy import deepcopy

from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem

from .sbu import SBU


class Linker(SBU):
    @classmethod
    def from_carboxy_mol(cls, mol):
        carboxy = Chem.MolFromSmarts("[O]C(=O)")
        matches = mol.GetSubstructMatches(carboxy)
        connection_sites = []
        for tpl in matches:
            connection_sites.append(tpl[0])
            connection_sites.append(tpl[2])

        edges = _get_edge_dict_from_rdkit_mol(mol)
        pmg_mol = _make_mol_from_rdkit_mol(mol)
        molecule_graph = MoleculeGraph.with_edges(pmg_mol, edges)

        return cls(pmg_mol, molecule_graph, connection_sites)


def _get_edge_dict_from_rdkit_mol(mol):
    edges = {}
    for bond in mol.GetBonds():
        edges[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = None
    return edges


def _make_mol_from_rdkit_mol(mol):
    """Takes the first conformer"""
    molecule = deepcopy(mol)
    AllChem.EmbedMolecule(molecule)
    conf = molecule.GetConformers()
    positions = conf[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Molecule(symbols, positions)
