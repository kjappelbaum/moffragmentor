# -*- coding: utf-8 -*-
from typing import Collection, List

import networkx as nx
import numpy as np
from openbabel import pybel as pb
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from scipy.spatial.distance import pdist

from ..utils import pickle_dump, write_cif

__all__ = ["SBU"]


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
        self._original_indices = original_indices
        self.molecule_graph = molecule_graph
        self._original_branching_indices = branching_indices
        self._original_binding_indices = binding_indices
        self._descriptors = None
        self.meta = {}
        self._nx_graph = None
        self._netlsd_heat = None
        self._netlsd_wave = None
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
        return len(self.original_branching_indices)

    @property
    def original_branching_indices(self):
        return self._original_branching_indices

    @property
    def branching_indices(self):
        return [
            self.mapping_from_original_indices[i]
            for i in self.original_branching_indices
        ]

    @property
    def branching_coords(self):
        return self.cart_coords[self.branching_indices]

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
        import nglview

        return nglview.show_pymatgen(self.molecule)

    def to(self, fmt: str, filename: str):
        return self.molecule.to(fmt, filename)

    @property
    def smiles(self):
        mol = self.openbabel_mol
        self._smiles = mol.write("can").strip()
        return self._smiles

    def _get_boxed_structure(self):
        max_size = _get_max_sep(self.molecule.cart_coords)
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


def _get_max_sep(coordinates):
    if len(coordinates) > 1:
        distances = pdist(coordinates)
        return np.max(distances)
    else:
        return 5
