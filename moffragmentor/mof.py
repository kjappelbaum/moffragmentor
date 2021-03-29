# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pylab as plt
import networkx as nx
import nglview
import numpy as np
from pymatgen import Structure
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    CutOffDictNN,
    JmolNN,
    MinimumDistanceNN,
    VoronoiNN,
)

from .fragmentor.filter import is_valid_linker, is_valid_node
from .fragmentor.locator import find_node_clusters
from .fragmentor.splitter import get_subgraphs_as_molecules
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
        self.nodes = []
        self._bridges = None
        self.linker = []
        self._topology = None
        self._connecting_node_indices = None
        self.meta = {}
        self._filter_solvent = True
        self.fragmentation_method = "all_node"
        self._solvent_indices = None
        self._branching_indices = None
        self._clear_properties()
        self._nx_graph = None

    def _clear_properties(self):
        for site in range(len(self.structure)):
            self.structure[site].properties = {"binding": False, "branching": False}

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
        self._clear_properties()

    def dump(self, path):
        """Dump this object as pickle file"""
        pickle_dump(self, path)

    def set_meta(self, key, value):
        """Set metadata"""
        self.meta[key] = value

    def __len__(self):
        return len(self.structure)

    @property
    def lattice(self):
        return self.structure.lattice

    @property
    def cart_coords(self):
        return self.structure.cart_coords

    @property
    def frac_coords(self):
        return self.structure.frac_coords

    def set_use_solvent_filter(self, use_solvent_filter):
        assert isinstance(use_solvent_filter, bool)
        self._filter_solvent = use_solvent_filter

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

    @property
    def linker_indices(self) -> set:
        if self._linker_indices is None:
            node_indices = self.node_indices
            self._linker_indices = set(range(len(self.structure)))
            self._linker_indices -= node_indices
            self._linker_indices -= set(sum(self._solvent_indices, []))
        return self._linker_indices

    def _label_site(self, site: int, label: str = "branching", key: bool = True):
        """adds property site to a structure"""
        self.structure[site].properties[label] = key

    # def _label_structure(self):
    #     """Label node and linker atoms that are connected"""
    #     for branching_atom in self._connecting_node_indices:
    #         neighbor_indices = self.get_neighbor_indices(branching_atom)
    #         for neighbor_idx in neighbor_indices:
    #             if neighbor_idx in self.linker_indices:
    #                 self._label_site(branching_atom)
    #                 self._label_site(neighbor_idx)

    def _label_branching_sites(self):
        for branching_index in self._branching_indices:
            self._label_site(branching_index, "branching", True)

    def _label_binding_sites(self):
        for binding_index in self._binding_indices:
            self._label_site(binding_index, "binding", True)

    def _label_structure(self):
        if self._binding_indices is None or self._branching_indices is None:
            self._get_node_indices()

        self._label_branching_sites()
        self._label_binding_sites()

    def _fragment(self) -> Tuple[List[Linker], List[Node]]:
        node_indices = self.node_indices
        self._label_structure()
        sg1 = deepcopy(self.structure_graph)
        linkers = []
        nodes = []
        node_structure = Structure.from_sites([self.structure[i] for i in node_indices])
        sg0 = StructureGraph.with_local_env_strategy(
            node_structure, MinimumDistanceNN(0.5)
        )

        node_molecules, node_graphs, node_indices = get_subgraphs_as_molecules(sg0)
        for node_molecule, node_graph, node_idx in zip(
            node_molecules, node_graphs, node_indices
        ):
            valid_node, to_add = is_valid_node(self, node_idx)
            if valid_node:
                nodes.append(
                    Node.from_labled_molecule(node_molecule, node_graph, self.meta)
                )
            if to_add:
                self._linker_indices.update(to_add)
                for idx in to_add:
                    self._node_indices.remove(idx)

        sg1.remove_nodes(list(self.node_indices))
        linker_molecules, linker_graphs, linker_indices = get_subgraphs_as_molecules(
            sg1
        )
        for linker_molecule, linker_graph, linker_idx in zip(
            linker_molecules, linker_graphs, linker_indices
        ):
            if is_valid_linker(linker_molecule):
                linkers.append(
                    Linker.from_labled_molecule(
                        linker_molecule, linker_graph, self.meta
                    )
                )
            else:
                # Something went wrong, we should raise a warning
                ...

        self.nodes = nodes
        self.linkers = linkers
        return linkers, nodes

    def fragment(
        self, fragmentation_method="all_node", filter_out_solvent: bool = True
    ):
        self.fragmentation_method = fragmentation_method
        self.set_use_solvent_filter(filter_out_solvent)

        return self._fragment()

    def _get_cif_text(self):
        return write_cif(self.structure, self.structure_graph, [])

    def write_cif(self, filename):
        with open(filename, "w") as f:
            f.write(self._get_cif_text())

    def _get_node_indices(self):
        # if self.fragmentation_method == "all_node":
        #     result = fragment_all_node(self, self._filter_solvent)
        # elif self.fragmentation_method == "oxo":
        #     result = fragment_oxo_node(self, self._filter_solvent)

        # self._node_indices = result["node_indices"]
        # self._solvent_connections = result["solvent_connections"]
        # self._solvent_indices = result["solvent_indices"]
        # self._connecting_node_indices = result["connecting_node_indices"]
        nodes, bs, connecting_paths = find_node_clusters(self)
        self._node_indices = nodes
        self._branching_indices = bs
        self._binding_indices = connecting_paths
