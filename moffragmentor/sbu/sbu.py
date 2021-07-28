# -*- coding: utf-8 -*-
"""Representation for a secondary building block"""
import warnings
from typing import Collection, List

import networkx as nx
import numpy as np
from backports.cached_property import cached_property
from openbabel import pybel as pb
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from scipy.spatial.distance import pdist

from ..descriptors import (
    chemistry_descriptors,
    distance_descriptors,
    get_lsop,
    rdkit_descriptors,
)
from ..utils import pickle_dump


def ob_mol_without_metals(obmol):
    import openbabel as ob

    mol = obmol.clone
    for atom in ob.OBMolAtomIter(mol.OBMol):
        if atom.IsMetal():
            mol.OBMol.DeleteAtom(atom)

    return mol


__all__ = ["SBU"]


def obmol_to_rdkit_mol(obmol):

    smiles = obmol.write("can").strip()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        warnings.warn("Attempting to remove metals to generate RDKit molecule")
        new_obmol = ob_mol_without_metals(obmol)
        smiles = new_obmol.write("can").strip()
        mol = Chem.MolFromSmiles(smiles, sanitize=True)

    return mol


class SBU:
    """Representation for a secondary building block.
    It also acts as container for site indices:

    - graph_branching_indices: are the branching indices according to the graph-based definition. They might not be part of the molecule.
    - closest_branching_index_in_molecule: those are always part of the molecule. In case the branching index is part of the molecule, they are equal to to the graph_branching_indices. Otherwise they are obtained as the closest vertex of the original branching vertex that is part of the molecule.
    - binding_indices: are the indices of the sites between the branching index and metal
    - original_indices: complete original set of indices that has been selected for this building blocks
    - persistent_non_metal_bridged: components that are connected via a bridge both in the MOF structure and building block molecule. No metal is part of the edge, i.e., bound solvents are not included in this set
    - terminal_in_mol_not_terminal_in_struct: indices that are terminal in the molecule but not terminal in the structure
    """

    def __init__(
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        center: np.ndarray,
        graph_branching_indices: Collection[int],
        closest_branching_index_in_molecule: Collection[int],
        binding_indices: Collection[int],
        original_indices: Collection[int],
        persistent_non_metal_bridged=None,
        terminal_in_mol_not_terminal_in_struct=None,
        graph_branching_coords=None,
    ):
        self.molecule = molecule
        self.center = center
        self._original_indices = original_indices
        self.molecule_graph = molecule_graph
        self._original_graph_branching_indices = graph_branching_indices
        self._original_closest_branching_index_in_molecule = (
            closest_branching_index_in_molecule
        )

        self._persistent_non_metal_bridged = persistent_non_metal_bridged
        self._terminal_in_mol_not_terminal_in_struct = (
            terminal_in_mol_not_terminal_in_struct
        )
        self.graph_branching_coords = graph_branching_coords

        self._original_binding_indices = binding_indices
        self._descriptors = None
        self.meta = {}
        self._nx_graph = None
        self.mapping_from_original_indices = dict(
            zip(original_indices, range(len(molecule)))
        )
        self.mapping_to_original_indices = dict(
            zip(range(len(molecule)), original_indices)
        )
        self._indices = original_indices

    def get_neighbor_indices(self, site: int) -> List[int]:
        """Get list of indices of neighboring sites"""
        return [site.index for site in self.molecule_graph.get_connected_sites(site)]

    # make this function so we can have different flavors
    def get_indices(self):
        return self._indices

    def __len__(self):
        return len(self.molecule)

    def __str__(self):
        return self.smiles

    def set_meta(self, key, value):
        self.meta[key] = value

    def dump(self, path):
        pickle_dump(self, path)

    def _get_nx_graph(self):
        if self._nx_graph is None:
            self._nx_graph = nx.Graph(self.molecule_graph.graph.to_undirected())
        return self._nx_graph

    @property
    def nx_graph(self):
        return self._get_nx_graph()

    @property
    def composition(self):
        return self.molecule.composition.alphabetical_formula

    @property
    def cart_coords(self):
        return self.molecule.cart_coords

    @property
    def coordination(self):
        return len(self.original_graph_branching_indices)

    @property
    def original_graph_branching_indices(self):
        return self._original_graph_branching_indices

    @property
    def graph_branching_indices(self):
        return [
            self.mapping_from_original_indices[i]
            for i in self.original_graph_branching_indices
        ]

    @cached_property
    def branching_coords(self):
        if self.graph_branching_coords is not None:
            return self.graph_branching_coords
        # ToDo: add here also a try capture for the case that the graph branching indices are not part of the molecule
        else:
            return self.cart_coords[self.graph_branching_indices]

    @cached_property
    def connecting_indices(self):
        return [
            self.mapping_from_original_indices[i]
            for i in self._original_closest_branching_index_in_molecule
        ]

    @property
    def original_binding_indices(self):
        return self._original_binding_indices

    @cached_property
    def rdkit_mol(self):
        return obmol_to_rdkit_mol(self.openbabel_mol)

    @cached_property
    def openbabel_mol(self):
        return self.get_openbabel_mol()

    def get_openbabel_mol(self):
        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        self._ob_mol = pm
        return pm

    def show_molecule(self):
        import nglview

        return nglview.show_pymatgen(self.molecule)

    def to(self, fmt: str, filename: str):
        return self.molecule.to(fmt, filename)

    @cached_property
    def smiles(self):
        mol = self.openbabel_mol
        smiles = mol.write("can").strip()
        return smiles

    def _get_boxed_structure(self):
        max_size = _get_max_sep(self.molecule.cart_coords)
        s = self.molecule.get_boxed_structure(
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            reorder=False,
        )
        return s

    def _get_connected_sites_structure(self):
        sites = []
        s = self._get_boxed_structure()
        for i in self.connecting_indices:
            sites.append(s[i])
        return Structure.from_sites(sites)

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

    @cached_property
    def descriptors(self):
        return self._get_descriptors()


def _get_max_sep(coordinates):
    if len(coordinates) > 1:
        distances = pdist(coordinates)
        return np.max(distances)
    else:
        return 5
