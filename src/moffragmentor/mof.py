# -*- coding: utf-8 -*-
"""Defining the main representation of a MOF"""
import os
from collections import defaultdict
from typing import List, Union, Optional

import networkx as nx
import numpy as np
import yaml
from backports.cached_property import cached_property
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CutOffDictNN
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .descriptors.sbu_dimensionality import get_structure_graph_dimensionality
from .fragmentor import run_fragmentation
from .utils import IStructure, pickle_dump, write_cif

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

with open(
    os.path.join(THIS_DIR, "utils", "data", "tuned_vesta.yml"), "r", encoding="utf8"
) as handle:
    VESTA_CUTOFFS = yaml.load(handle, Loader=yaml.UnsafeLoader)

VestaCutoffDictNN = CutOffDictNN(cut_off_dict=VESTA_CUTOFFS)

__all__ = ["MOF"]


class MOF:  # pylint:disable=too-many-instance-attributes, too-many-public-methods
    """Main representation for a MOF structure"""

    def __init__(self, structure: Structure, structure_graph: StructureGraph):
        self._structure = structure
        self._structure_graph = structure_graph
        self._node_indices = None
        self._linker_indices = None
        self._bridges = None
        self._topology = None
        self._connecting_node_indices = None
        self._solvent_indices = None
        self._branching_indices = None
        self._nx_graph = None
        nx.set_node_attributes(
            self._structure_graph.graph,
            name="idx",
            values=dict(zip(range(len(structure_graph)), range(len(structure_graph)))),
        )

    def _reset(self):
        self._node_indices = None
        self._linker_indices = None
        self._topology = None
        self._connecting_node_indices = None
        self._solvent_indices = None
        self._bridges = None
        self._nx_graph = None

    def dump(self, path):
        """Dump this object as pickle file"""
        pickle_dump(self, path)

    def __len__(self) -> str:
        return len(self.structure)

    # we should not be able to write this
    @property
    def structure(self):
        return self._structure

    # we should not be able to write this
    @property
    def structure_graph(self):
        return self._structure_graph

    @cached_property
    def dimensionality(self):
        return get_structure_graph_dimensionality(self.structure_graph)

    @property
    def lattice(self) -> Lattice:
        return self.structure.lattice

    @property
    def composition(self) -> str:
        return self.structure.composition.alphabetical_formula

    @property
    def cart_coords(self) -> np.ndarray:
        return self.structure.cart_coords

    @cached_property
    def frac_coords(self) -> np.ndarray:
        return self.structure.frac_coords

    @classmethod
    def from_cif(cls, cif: Union[str, os.PathLike], symprec: float=0.5, angle_tolerance: float=10):
        # using the IStructure avoids bugs where somehow the structure changes
        structure = IStructure.from_file(cif)
        spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        structure = spga.get_conventional_standard_structure()
        structure = IStructure.from_sites(structure)
        structure_graph = StructureGraph.with_local_env_strategy(structure, VestaCutoffDictNN)
        return cls(structure, structure_graph)

    @classmethod 
    def from_structure(cls, structure: Structure, symprec: Optional[float]=0.5, angle_tolerance: Optional[float]=10):
        if (symprec is not None) and (angle_tolerance is not None):
            spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            structure = spga.get_conventional_standard_structure()
        structure = IStructure.from_sites(structure)
        structure_graph = StructureGraph.with_local_env_strategy(structure, VestaCutoffDictNN) 
        return cls(structure, structure_graph)
 

    def _is_terminal(self, index):
        return len(self.get_neighbor_indices(index)) == 1

    @cached_property
    def terminal_indices(self):
        return [i for i in range(len(self.structure)) if self._is_terminal(i)]

    def _get_nx_graph(self):
        if self._nx_graph is None:
            self._nx_graph = nx.Graph(self.structure_graph.graph.to_undirected())
        return self._nx_graph

    def _leads_to_terminal(self, edge):
        sorted_edge = sorted(edge)
        try:
            bridge_edge = self.bridges[sorted_edge[0]]
            return sorted_edge[1] in bridge_edge
        except KeyError:
            return False

    @cached_property
    def nx_graph(self):
        """Structure graph as networkx graph object"""
        return self._get_nx_graph()

    def _generate_bridges(self):
        if self._bridges is None:
            bridges = list(nx.bridges(self.nx_graph))

            bridges_dict = defaultdict(list)
            for key, value in bridges:
                bridges_dict[key].append(value)

            self._bridges = dict(bridges_dict)
        return self._bridges

    @cached_property
    def bridges(self) -> dict:
        """Bridges are edges in a graph that, if deleted,
        increase the number of connected components"""
        return self._generate_bridges()

    @cached_property
    def adjaceny_matrix(self):
        return nx.adjacency_matrix(self.structure_graph.graph)

    def show_adjacency_matrix(self, highlight_metals=False):
        """Plot structure graph as adjaceny matrix"""
        import matplotlib.pylab as plt  # pylint:disable=import-outside-toplevel

        matrix = self.adjaceny_matrix.todense()
        if highlight_metals:
            cols = np.nonzero(matrix[self.metal_indices, :])
            rows = np.nonzero(matrix[:, self.metal_indices])
            matrix[self.metal_indices, cols] = 2
            matrix[rows, self.metal_indices] = 2
        plt.imshow(self.adjaceny_matrix.todense(), cmap="Greys_r")

    @cached_property
    def metal_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if species.is_metal]

    @cached_property
    def h_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if str(species) == "H"]

    def get_neighbor_indices(self, site: int) -> List[int]:
        """Get list of indices of neighboring sites"""
        return [site.index for site in self.structure_graph.get_connected_sites(site)]

    def get_symbol_of_site(self, site: int) -> str:
        """Get elemental symbol of site indexed site"""
        return str(self.structure[site].specie)

    def show_structure(self):
        """Visualize structure using nglview"""
        import nglview  # pylint:disable=import-outside-toplevel

        return nglview.show_pymatgen(self.structure)

    def _fragment(self):
        fragmentation_result = run_fragmentation(self)
        return fragmentation_result

    def fragment(self):
        """Splits the MOF into building blocks (linkers, nodes, bound,
        undbound solvent, net embedding of those building blocks)"""
        return self._fragment()

    def _get_cif_text(self) -> str:
        return write_cif(self.structure, self.structure_graph, [])

    def write_cif(self, filename) -> None:
        """Writes the structure to a CIF file"""
        with open(filename, "w", encoding="utf8") as file_handle:
            file_handle.write(self._get_cif_text())
