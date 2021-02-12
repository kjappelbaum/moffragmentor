# -*- coding: utf-8 -*-
__all__ = ["SBU", "Node", "Linker"]

import datetime
from copy import deepcopy
from typing import List

import nglview
import numpy as np
from openbabel import pybel as pb
from pymatgen import Molecule, Structure
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist

from .descriptors import (
    chemistry_descriptors,
    distance_descriptors,
    get_lsop,
    rdkit_descriptors,
)
from .utils import pickle_dump, write_cif


def get_max_sep(coordinates):
    if len(coordinates) > 1:
        distances = pdist(coordinates)
        return np.max(distances)
    else:
        return 5


class SBU:
    def __init__(
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        connection_indices: List[int],
    ):
        self.molecule = molecule
        self._ob_mol = None
        self._smiles = None
        self._rdkit_mol = None
        self.molecule_graph = molecule_graph
        self.connection_indices = connection_indices
        self._descriptors = None
        self.meta = {}

    def set_meta(self, key, value):
        self.meta[key] = value

    def dump(self, path):
        pickle_dump(self, path)

    @property
    def rdkit_mol(self):
        if self._rdkit_mol is not None:
            return self._rdkit_mol
        else:
            self._rdkit_mol = Chem.MolFromSmiles(self.smiles)
            return self.rdkit_mol

    @property
    def openbabel_mol(self):
        if self._ob_mol is not None:
            return self._ob_mol
        else:
            return self.get_openbabel_mol()

    @classmethod
    def from_labled_molecule(cls, mol, mg, meta={}):
        connection_indices = get_binding_indices(mol)
        sbu = cls(mol, mg, connection_indices)
        sbu.meta = meta
        return sbu

    def get_openbabel_mol(self):
        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        self._ob_mol = pm
        return pm

    def show_molecule(self):
        return nglview.show_pymatgen(self.molecule)

    @property
    def smiles(self):
        mol = self.openbabel_mol
        self._smiles = mol.write("can").strip()
        return self._smiles

    def _get_boxed_structure(self):
        max_size = get_max_sep(self.molecule.cart_coords)
        s = self.molecule.get_boxed_structure(
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            reorder=False,
        )
        return s

    def _get_descriptors(self):
        s = self._get_connected_sites_structure()

        descriptors_lsop = get_lsop(s)
        descriptors_rdkit = rdkit_descriptors(self.rdkit_mol)
        descriptors_chemistry = chemistry_descriptors(s)
        descriptors_distance = distance_descriptors(s)

        return {
            **descriptors_lsop,
            **descriptors_rdkit,
            **descriptors_chemistry,
            **descriptors_distance,
        }

    def get_descriptors(self):
        if not self._descriptors:
            self._descriptors = self._get_descriptors()
        return self._descriptors

    def _get_connected_sites_structure(self):
        sites = []
        s = self._get_boxed_structure()
        for i in self.connection_indices:
            sites.append(s[i])
        return Structure.from_sites(sites)

    def _get_tobacco_string(self):
        s = self._get_boxed_structure()
        return write_cif(
            s, self.molecule_graph, self.connection_indices, molecule=self.molecule
        )

    def write_tobacco_file(self, filename=None):
        """To create a database of building blocks it is practical to be able to
        write Tobacco input file.
        We need to only place the X for sites with property binding=True
        """

        cif_string = self._get_tobacco_string()
        if filename is None:
            return cif_string


class Node(SBU):
    pass


def _get_edge_dict_from_rdkit_mol(mol):
    edges = {}
    for bond in mol.GetBonds():
        edges[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = None
    return edges


def _make_mol_from_rdkit_mol(mol):
    """Takes the first conformer"""
    m = deepcopy(mol)
    AllChem.EmbedMolecule(m)
    conf = m.GetConformers()
    positions = conf[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Molecule(symbols, positions)


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
        mg = MoleculeGraph.with_edges(pmg_mol, edges)

        return cls(pmg_mol, mg, connection_sites)


def get_binding_indices(mol):
    indices = []
    for i, site in enumerate(mol):
        try:
            if site.properties["binding"] == True:
                indices.append(i)
        except KeyError:
            pass

    return indices
