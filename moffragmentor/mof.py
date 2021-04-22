# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pylab as plt
import networkx as nx
import nglview
import numpy as np
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    CutOffDictNN,
    JmolNN,
    MinimumDistanceNN,
    VoronoiNN,
)
from pymatgen.core import Structure

from .fragmentor import run_fragmentation
from .sbu import Linker, Node
from .utils import pickle_dump, write_cif

VestaCutoffDictNN = CutOffDictNN.from_preset("vesta_2019")

__all__ = ["MOF"]


class MOF:
    def __init__(self, structure: Structure, structure_graph: StructureGraph):
        self.structure = structure
        self.structure_graph = structure_graph
        self._node_indices = None
        self._linker_indices = None
        self._bridges = None
        self._topology = None
        self._connecting_node_indices = None
        self._solvent_indices = None
        self._branching_indices = None
        self._nx_graph = None
        self._netlsd_heat = None
        self._netlsd_wave = None
        # ToDo: Maybe add the binding/branching attributes back to the graph
        nx.set_node_attributes(
            self.structure_graph.graph,
            name="idx",
            values=dict(zip(range(len(structure_graph)), range(len(structure_graph)))),
        )

    def _reset(self):
        self._node_indices = None
        self._linker_indices = None
        self.nodes = []
        self.linker = []
        self._topology = None
        self._connecting_node_indices = None
        self._solvent_indices = None
        self._bridges = None
        self._nx_graph = None
        self._netlsd_heat = None
        self._netlsd_wave = None

    def dump(self, path):
        """Dump this object as pickle file"""
        pickle_dump(self, path)

    def __len__(self):
        return len(self.structure)

    def _get_netlsd_heat(self):
        import netlsd

        if self._netlsd_heat is None:
            self._netlsd_heat = netlsd.heat(self.nx_graph)
        return self._netlsd_heat

    def _get_netlsd_wave(self):
        import netlsd

        if self._netlsd_wave is None:
            self._netlsd_wave = netlsd.wave(self.nx_graph)
        return self._netlsd_wave

    @property
    def netlsd_heat(self):
        return self._get_netlsd_heat()

    @property
    def netlsd_wave(self):
        return self._get_netlsd_wave()

    def heat_compare(self, other):
        import netlsd

        return netlsd.compare(self.netlsd_heat, other.netlsd_heat)

    def wave_compare(self, other):
        import netlsd

        return netlsd.compare(self.netlsd_wave, other.netlsd_wave)

    @property
    def lattice(self):
        return self.structure.lattice

    @property
    def composition(self):
        return self.structure.composition.alphabetical_formula

    @property
    def cart_coords(self):
        return self.structure.cart_coords

    @property
    def frac_coords(self):
        return self.structure.frac_coords

    @classmethod
    def from_cif(cls, cif: Union[str, os.PathLike]):
        s = Structure.from_file(cif)
        sg = StructureGraph.with_local_env_strategy(s, VestaCutoffDictNN)
        return cls(s, sg)

    def _is_branch_point(self, index: int, allow_metal: bool = False) -> bool:
        """The branch point definition is key for splitting MOFs
        into linker and nodes. Branch points are here defined as points
        that have at least three connections that do not lead to a tree or
        leaf node.

        Args:
            index (int): index of site that is to be probed
            allow_metal (bool, optional): If True it does not perform this check for metals (and just return False). Defaults to False.

        Returns:
            bool: True if this is a branching index
        """
        valid_connections = 0
        connected_sites = self.get_neighbor_indices(index)
        non_metal_connections = 0
        if len(connected_sites) < 3:
            return False

        if not allow_metal:
            if index in self.metal_indices:
                return False

        for connected_site in connected_sites:
            if (not self._leads_to_terminal((index, connected_site))) and (
                not self._is_terminal(connected_site)
            ):
                valid_connections += 1
                if not connected_site in self.metal_indices:
                    non_metal_connections += 1

        return (valid_connections >= 3) and (non_metal_connections >= 2)

    def _is_terminal(self, index):
        return len(self.get_neighbor_indices(index)) == 1

    def _get_nx_graph(self):
        if self._nx_graph is None:
            self._nx_graph = nx.Graph(self.structure_graph.graph.to_undirected())
        return self._nx_graph

    def _leads_to_terminal(self, edge):
        sorted_edge = sorted(edge)
        try:
            r = self.bridges[sorted_edge[0]]
            return sorted_edge[1] in r
        except KeyError:
            return False

    @property
    def nx_graph(self):
        return self._get_nx_graph()

    def _generate_bridges(self):
        if self._bridges is None:
            bridges = list(nx.bridges(self.nx_graph))

            bridges_dict = defaultdict(list)
            for key, value in bridges:
                bridges_dict[key].append(value)

            self._bridges = dict(bridges_dict)
        return self._bridges

    @property
    def bridges(self):
        return self._generate_bridges()

    @property
    def adjaceny_matrix(self):
        return nx.adjacency_matrix(self.structure_graph.graph)

    def show_adjacency_matrix(self, highlight_metals=False):
        matrix = self.adjaceny_matrix.todense()
        if highlight_metals:
            cols = np.nonzero(matrix[self.metal_indices, :])
            rows = np.nonzero(matrix[:, self.metal_indices])
            matrix[self.metal_indices, cols] = 2
            matrix[rows, self.metal_indices] = 2
        plt.imshow(self.adjaceny_matrix.todense(), cmap="Greys_r")

    @property
    def metal_indices(self) -> List[int]:
        return [
            i for i, species in enumerate(self.structure.species) if species.is_metal
        ]

    @property
    def h_indices(self) -> List[int]:
        return [
            i for i, species in enumerate(self.structure.species) if str(species) == "H"
        ]

    def get_neighbor_indices(self, site: int) -> List[int]:
        return [site.index for site in self.structure_graph.get_connected_sites(site)]

    def get_symbol_of_site(self, site: int) -> str:
        return str(self.structure[site].specie)

    @property
    def node_indices(self) -> set:
        """Returns node indices from cache if they already have been determined, otherwise calculates them"""
        if self._node_indices is None:
            self._get_node_indices()

        return self._node_indices

    def show_structure(self):
        return nglview.show_pymatgen(self.structure)

    def _fragment(self):
        fragmentation_result = run_fragmentation(self)
        return fragmentation_result

    def fragment(self):

        return self._fragment()

    def _get_cif_text(self):
        return write_cif(self.structure, self.structure_graph, [])

    def write_cif(self, filename):
        with open(filename, "w") as f:
            f.write(self._get_cif_text())
