# -*- coding: utf-8 -*-
__all__ = [
    "SBU",
    "SBUCollection",
    "Node",
    "Linker",
    "LinkerCollection",
    "NodeCollection",
]

from collections import Counter
from copy import deepcopy
from typing import Collection, Dict, List, Set

import nglview
import numpy as np
from openbabel import pybel as pb
from pymatgen import Molecule, Structure
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
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
from .fragmentor.splitter import get_subgraphs_as_molecules
from .utils import (
    _not_relevant_structure_indices,
    connected_mol_from_indices,
    get_edge_dict,
    pickle_dump,
    unwrap,
    write_cif,
)


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
        center: np.ndarray,
        branching_indices: List[int],
        binding_indices: List[int],
        original_indices: Collection[int],
    ):
        self.molecule = molecule
        self.center = center
        self._ob_mol = None
        self._smiles = None
        self._rdkit_mol = None
        self.molecule_graph = molecule_graph
        self._original_branching_indices = branching_indices
        self._original_binding_indices = binding_indices
        self._descriptors = None
        self.meta = {}
        self.mapping_from_original_indices = dict(
            zip(original_indices, range(len(molecule)))
        )
        self.mapping_to_original_indices = dict(
            zip(range(len(molecule)), original_indices)
        )
        self._indices = original_indices

    @property
    def indices(self):
        return self._indices

    def __len__(self):
        return len(self.molecule)

    def __str__(self):
        return self.smiles

    def set_meta(self, key, value):
        self.meta[key] = value

    def dump(self, path):
        pickle_dump(self, path)

    @property
    def composition(self):
        return self.molecule.composition.alphabetical_formula

    @property
    def cart_coords(self):
        return self.molecule.cart_coords

    @property
    def coordination(self):
        return len(self.original_branching_indices)

    @property
    def original_branching_indices(self):
        return self._original_branching_indices

    @property
    def original_binding_indices(self):
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

    def get_openbabel_mol(self):
        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        self._ob_mol = pm
        return pm

    def show_molecule(self):
        return nglview.show_pymatgen(self.molecule)

    def to(self, fmt: str, filename: str):
        return self.molecule.to(fmt, filename)

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
    @classmethod
    def from_mof_and_indices(
        cls,
        mof,
        node_indices: Set[int],
        branching_indices: Set[int],
        binding_indices: Set[int],
    ):
        graph_ = deepcopy(mof.structure_graph)
        to_delete = _not_relevant_structure_indices(mof.structure, node_indices)
        graph_.remove_nodes(to_delete)

        mol = connected_mol_from_indices(mof, node_indices)
        graph = MoleculeGraph.with_edges(mol, get_edge_dict(graph_))
        # # Todo: we can make this more efficient by skipping the expansion to the supercell and directly extracting the subgraphs
        # mol, graph, idx, centers = get_subgraphs_as_molecules(graph_)
        # assert len(mol) == 1

        center = np.mean(mol.cart_coords, axis=0)

        node = cls(
            mol,
            graph,
            center,
            branching_indices & node_indices,
            binding_indices & node_indices,
            node_indices,
        )
        return node


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


class SBUCollection:
    def __init__(self, sbus: List[SBU]):
        self.sbus = sbus
        self._sbu_types = None
        self._composition = None
        self._unique_sbus = None
        self._centers = [sbu.center for sbu in self.sbus]
        self._indices = [sbu.indices for sbu in self.sbus]

        assert len(self._indices) == len(self.sbus)
        assert len(self._indices) == len(self.sbus)

    def __len__(self):
        return len(self.sbus)

    def __getitem__(self, index):
        return self.sbus[index]

    def __next__(self):
        for sbu in self.sbus:
            yield sbu

    @property
    def indices(self):
        return self._indices

    @property
    def centers(self):
        return self._centers

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

    def _get_composition(self):
        if self._composition is None:
            composition = []
            for mol in self.sbus:
                composition.append(str(mol.composition))
            composition_counter = Counter(composition)
            self._composition = dict(composition_counter)

        return self._composition

    @property
    def composition(self):
        return self._get_composition()


class LinkerCollection(SBUCollection):
    @property
    def building_block_composition(self):
        if self._composition is None:
            self._composition = ["".join(["L", i]) for i in self.sbu_types]
        return self._composition


class NodeCollection(SBUCollection):
    @property
    def building_block_composition(self):
        if self._composition is None:
            self._composition = ["".join(["N", i]) for i in self.sbu_types]
        return self._composition


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


def get_indices_with_property(mol, property: str = "branching"):
    indices = []
    for i, site in enumerate(mol):
        try:
            if site.properties[property] == True:
                indices.append(i)
        except KeyError:
            pass

    return indices
