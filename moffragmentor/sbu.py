# -*- coding: utf-8 -*-
__all__ = [
    "SBU",
    "SBUCollection",
    "Node",
    "Linker",
    "LinkerCollection",
    "NodeCollection",
]

import datetime
from copy import deepcopy
from typing import Dict, List

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
from .utils import pickle_dump, unwrap, write_cif


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
        branching_indices: List[int],
        binding_indices: List[int],
        mapping_from_original_indices: Dict[int, int],
    ):
        self.molecule = molecule
        self._ob_mol = None
        self._smiles = None
        self._rdkit_mol = None
        self.molecule_graph = molecule_graph
        self.branching_indices: List = branching_indices
        self._original_branching_indices = None
        self._original_binding_indices = None
        self.binding_indices: List = binding_indices
        self._descriptors = None
        self.meta = {}
        self.mapping_from_original_indices = mapping_from_original_indices
        self.mapping_to_original_indices = {
            v: k for k, v in self.mapping_from_original_indices.items()
        }

    def __len__(self):
        return len(self.molecule)

    def __str__(self):
        return self.smiles

    def set_meta(self, key, value):
        self.meta[key] = value

    def dump(self, path):
        pickle_dump(self, path)

    @property
    def cart_coords(self):
        return self.molecule.cart_coords

    @property
    def coordination(self):
        return len(self.branching_indices)

    @property
    def original_branching_indices(self):
        if self._original_branching_indices is None:
            _original_branching_indices = {
                self.mapping_to_original_indices[i] for i in self.branching_indices
            }

            self._original_branching_indices = _original_branching_indices

        return self._original_branching_indices

    @property
    def original_binding_indices(self):
        if self._original_binding_indices is None:
            _original_binding_indices = {
                self.mapping_to_original_indices[i] for i in self.branching_indices
            }
            self._original_binding_indices = _original_binding_indices

        return self._original_binding_indices

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
    def from_labled_molecule(cls, mol, mg, mapping_from_original_indices, meta={}):
        branching_indices = get_indices_with_property(mol)
        binding_indices = get_indices_with_property(mol, "binding")
        sbu = cls(
            mol, mg, branching_indices, binding_indices, mapping_from_original_indices
        )
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
            s,
            self.molecule_graph,
            self.connection_indices,
            molecule=self.molecule,
            write_bonding_mode=True,
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


class SBUCollection:
    def __init__(self, sbus: List[SBU]):
        self.sbus = sbus
        self._sbu_types = None
        self._composition = None
        self._unique_sbus = None

    def __len__(self):
        return len(self.sbus)

    def __getitem__(self, index):
        return self.sbus[index]

    def __next__(self):
        for sbu in self.sbus:
            yield sbu

    @property
    def sbu_types(self):
        if not self._sbu_types:
            self._get_unique()
        return self._sbu_types

    @property
    def coordination_numbers(self):
        return [sbu.coordination for sbu in self.sbus]

    @property
    def unique_sbus(self):
        if not self._sbu_types:
            self._get_unique()
        return self._unique_sbus

    def _get_unique(self):
        all_strings = [str(sbu) for sbu in self.sbus]
        unique_strings = set(all_strings)
        unique_mapping = []
        for string in all_strings:
            for i, unique_string in enumerate(unique_strings):
                if string == unique_string:
                    unique_mapping.append(i)
                    break
        self._unique_sbus = unique_strings
        self._sbu_types = unique_mapping


class LinkerCollection(SBUCollection):
    @property
    def composition(self):
        if self._composition is None:
            self._composition = ["".join(["L", i]) for i in self.sbu_types]


class NodeCollection(SBUCollection):
    @property
    def composition(self):
        if self._composition is None:
            self._composition = ["".join(["N", i]) for i in self.sbu_types]


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


def get_indices_with_property(mol, property: str = "branching"):
    indices = []
    for i, site in enumerate(mol):
        try:
            if site.properties[property] == True:
                indices.append(i)
        except KeyError:
            pass

    return indices
