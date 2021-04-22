# -*- coding: utf-8 -*-
from typing import Dict

import numpy as np
from pymatgen.core import Lattice, Structure

from .sbu import LinkerCollection, NodeCollection
from .utils import is_tool
from .utils.errors import JavaNotFoundError
from .utils.periodic_graph import (
    _draw_net_structure_graph,
    _get_pmg_structure_graph_for_net,
    _simplify_structure_graph,
)
from .utils.systre import _get_systre_input_from_pmg_structure_graph, run_systre

__all__ = ["NetEmbedding"]


class NetEmbedding:
    """In all composition/coordinates arrays we have linkers first"""

    def __init__(
        self,
        linker_collection: LinkerCollection,
        node_collection: NodeCollection,
        edge_dict: Dict[int, list],
        lattice: Lattice,
    ):
        """
        A NetEmbedding instance is defined by a collection of linkers and nodes
        and their connection on a lattice

        Args:
            linker_collection (LinkerCollection): Iterable object in which every item is a Linker,
                the order is important as the edge dict uses indices to refer to specific linker
            node_collection (NodeCollection): Iterable object in which every item is a metal node,
                the order is important as the keys of the edge dict are node indices, refering to
                the node collection
            edge_dict (Dict[int, list]): Defining the connection between metal nodes and linkers.
                The main keys are the node indices. The value is a list of tuples that contain the
                linker indices, the jimages, and the centers
            lattice (Lattice): A net is a periodic graph. Hence we need information about
                the periodicity, i.e., the lattice. For this we use a pymatgen Lattice object.
                We use it to convert between cartesian and fractional coordinates and
                to get the lattice constants. Usually, it can simply be accessed from a pymatgen
                Structure `s` via `s.lattice`.
        """
        self.node_collection = node_collection
        self.linker_collection = linker_collection
        self.linker_centers = linker_collection.centers
        self.node_centers = node_collection.centers
        self.edge_dict = edge_dict

        self._edges = None
        self._lattice = lattice
        self._cart_coords = None
        self._composition = None
        self._rcsr_code = None
        self._space_group = None
        self._structure_graph = _get_pmg_structure_graph_for_net(self)

    def __len__(self):
        return len(self.node_collection) + len(self.selected_linkers)

    @property
    def edges(self):
        if self._edges is None:
            self._find_edges()
        return self._edges

    def _find_edges(self):
        edges = set()
        for i, node in enumerate(self.node_collection):
            for j, linker in enumerate(self.linker_collection):
                if (
                    len(
                        linker.original_branching_indices
                        & node.original_branching_indices
                    )
                    > 0
                ):
                    edge_tuple = (j, i + len(self.linker_collection))
                    edges.add(edge_tuple)
        self._edges = edges

    def get_distance(self, i, j):
        return np.linalg.norm(self.cart_coords[i] - self.cart_coords[j])

    def get_frac_distance(self, i, j):
        return np.linalg.norm(self.frac_coords[i] - self.frac_coords[j])

    @property
    def lattice(self):
        return self._lattice

    @property
    def coordination_numbers(self):
        return (
            self.linker_collection.coordination_numbers
            + self.node_collection.coordination_numbers
        )

    @property
    def cart_coords(self):
        if self._cart_coords is None:
            self._cart_coords = self._get_coordinates()
        return self._cart_coords

    def _get_coordinates(self):
        coordinates = []

        for i, _ in enumerate(self.linker_collection):
            coordinates.append(self.linker_centers[i])

        for i, _ in enumerate(self.node_collection):
            coordinates.append(self.node_centers[i])

        return np.array(coordinates).reshape(-1, 3)

    def _get_dummy_structure(self):
        linker_symbols = ["O" for i, _ in enumerate(self.linker_collection)]
        node_symbols = ["Si" for _ in self.node_collection]
        coordinates = self._get_coordinates()
        frac_coords = self.lattice.get_fractional_coords(coordinates)
        return Structure(self.lattice, linker_symbols + node_symbols, frac_coords)

    def show_dummy_structure(self):
        import nglview

        return nglview.show_pymatgen(self._get_dummy_structure())

    def plot_net(self):
        return _draw_net_structure_graph(self)

    @property
    def structure_graph(self):
        return self._structure_graph

    @property
    def frac_coords(self):
        return self.lattice.get_fractional_coords(self.cart_coords)

    @property
    def composition(self):
        if self._composition is None:
            self._get_composition()
        return self._composition

    @property
    def kinds(self):
        if self._self_kinds is None:
            self._get_kinds()

        return self._kinds

    def _get_composition(self):
        return self.linker_collection.composition + self.node_collection.composition

    def _get_kinds(self):
        node_indices = [
            i + len(self.linker_collection.unique_sbus)
            for i in self.node_collection.unique_sbus
        ]
        self._kinds = self.linker_collection.sbu_types + node_indices

    def _write_systre_file(self, simplify: bool = True):
        if simplify:
            new_structure_graph = _simplify_structure_graph(self.structure_graph)
        else:
            new_structure_graph = self.structure_graph
        filestring = _get_systre_input_from_pmg_structure_graph(
            new_structure_graph, self.lattice
        )
        return filestring

    @property
    def rcsr_code(self):
        if self._rcsr_code is None:
            self._run_systre()
        return self._rcsr_code

    @property
    def space_group(self):
        if self._space_group is None:
            self._run_systre()
        return self._space_group

    def _run_systre(self):
        self._rcsr_code = ""
        self._space_group = ""

        if not is_tool("java"):
            raise JavaNotFoundError(
                "To determine the topology of the net, `java` must be in the PATH."
            )
        systre_output = run_systre(self._write_systre_file())
        self._rcsr_code = systre_output["rcsr_code"]
        self._space_group = systre_output["space_group"]
