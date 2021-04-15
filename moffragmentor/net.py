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
)
from .utils.systre import run_systre

__all__ = ["NetEmbedding"]


class NetEmbedding:
    """In all composition/coordinates arrays we have linkers first"""

    def __init__(
        self,
        linker_collection: LinkerCollection,
        node_collection: NodeCollection,
        edge_dict: Dict[int, dict],
        lattice: Lattice,
    ):
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

    def _write_systre_file(self):
        symmetry_group = "   GROUP P1"

        cell_line = f"   CELL {self.lattice.a} {self.lattice.b} {self.lattice.c} {self.lattice.alpha} {self.lattice.beta} {self.lattice.gamma}"
        atom_lines = []
        counter = 0
        for coordination, coordinate in zip(
            self.coordination_numbers, self.frac_coords
        ):
            atom_lines.append(
                f"   NODE {counter} {coordination} {coordinate[0]:.4f} {coordinate[1]:.4f} {coordinate[2]:.4f}"
            )
            counter += 1

        edge_lines = []
        missing_nodes = []
        missing_node_indices = []
        for edge in self.edges:
            frac_coords_a = self.frac_coords[edge[0]]
            frac_coords_b = self.frac_coords[edge[1]]

            _, images = self.lattice.get_distance_and_image(
                frac_coords_a, frac_coords_b
            )
            frac_coords_b += images
            if sum(images) != 0:
                missing_nodes.append(frac_coords_b)
                missing_node_indices.append(edge[1])

            edge_lines.append(
                f"   EDGE   {frac_coords_a[0]:.4f} {frac_coords_a[1]:.4f} {frac_coords_a[2]:.4f} {frac_coords_b[0]:.4f} {frac_coords_b[1]:.4f} {frac_coords_b[2]:.4f}"
            )

        file_lines = (
            ["CRYSTAL", "   NAME", symmetry_group, cell_line]
            + atom_lines
            + edge_lines
            + ["END"]
        )

        filestring = "\n".join(file_lines)

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
