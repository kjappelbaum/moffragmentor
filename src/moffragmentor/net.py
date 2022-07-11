# -*- coding: utf-8 -*-
"""Defines Python representations for nets and voltage.

The VoltageEdge class implementation is based on https://github.com/qai222/CrystalGraph,
which is licensed under the MIT license provided here:

MIT License

Copyright (c) 2021 Alex Ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from collections import OrderedDict, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from backports import cached_property
from pymatgen.core import IMolecule, Lattice, Molecule
from pymatgen.util.coord import pbc_diff

from moffragmentor.sbu import SBU
from moffragmentor.sbu.linkercollection import LinkerCollection
from moffragmentor.sbu.nodecollection import NodeCollection

from .utils.systre import run_systre


def is_3d_parallel(v1: np.array, v2: np.array, eps: float = 1e-5) -> bool:
    cp = np.cross(v1, v2)
    return np.allclose(cp, np.zeros(3), atol=eps)


class VoltageEdge:
    """Representation of an edge in a labeled quotient graph (LQG).

    This class simplifies the construction of LQGs (via the equiality operator).
    """

    def __init__(self, vector: np.ndarray, n1: int, n2: int, n1_image: tuple, n2_image: tuple):
        """Initialize a VoltageEdge.

        Args:
            vector (np.ndarray): The vector of the edge
                (difference between coordinates of nodes).
            n1 (int): The index of the first node.
            n2 (int): The index of the second node.
            n1_image (tuple): The image of the first node.
            n2_image (tuple): The image of the second node.
        """
        self.vector = vector
        self.n1 = n1
        self.n2 = n2
        self.n1_image = n1_image if n1_image is not None else (0, 0, 0)
        self.n2_image = n2_image if n2_image is not None else (0, 0, 0)
        self.terminals = frozenset([self.n1, self.n2])

    def __repr__(self):
        """Return a string representation of the VoltageEdge."""
        terms = (self.n1, self.n2)
        imags = (self.n1_image, self.n2_image)
        a, b = ((x, t) for x, t in sorted(zip(imags, terms), key=lambda x: x[1]))
        return "Edge: {}, voltage: {}, vector: {}".format((a, b), self.voltage, self.vector)

    @property
    def length(self) -> float:
        """Length of the edge."""
        return np.linalg.norm(self.vector)

    @property
    def voltage(self) -> Tuple[int, int, int]:
        """Voltage is the tuple describing the direction of the edge.

        In simple words, it represents the translation operation.

        Returns:
            Tuple[int, int, int]: The voltage of the edge.
        """
        terms = (self.n1, self.n2)
        imags = (self.n1_image, self.n2_image)
        a_image, b_image = (x for x, _ in sorted(zip(imags, terms), key=lambda x: x[1]))
        return tuple(a_image[i] - b_image[i] for i in range(3))

    def __eq__(self, other):
        """Check if two VoltageEdges are equal."""
        eps = 1e-5
        is_parallel = is_3d_parallel(self.vector, other.vector, eps=eps)
        if not is_parallel:
            return False
        is_sameterminal = self.terminals == other.terminals
        if not is_sameterminal:
            return False
        is_eqlength = abs(self.length - other.length) < eps
        if not is_eqlength:
            return False
        is_eqvoltage = is_3d_parallel(
            self.voltage, other.voltage
        ) or self.voltage == other.voltage == (0, 0, 0)
        if not is_eqvoltage:
            return False
        return True


class NetNode:
    def __init__(
        self,
        coords: np.array,
        branching_coord_in_cell: Optional[Iterable[np.array]] = None,
        all_branching_coord: Optional[Iterable[np.array]] = None,
        molecule: Optional[Union[Molecule, IMolecule]] = None,
        flavor: Optional[str] = None,
    ):
        """Initialize a NetNode.

        Args:
            coords (np.array): Coordinates of the node (center of mass).
            branching_coord_in_cell (Optional[Iterable[np.array]], optional):
                Coordinates of the branching indices that are in the original unit cell.
                Defaults to None.
            all_branching_coord (Optional[Iterable[np.array]], optional):
                Coordinates of all branching sites. Defaults to None.
            molecule (Optional[Union[Molecule, IMolecule]], optional):
                Pymatgen molecule object. Defaults to None.
            flavor (Optional[str], optional):
                String describing the net type, e.g. "linker", "metalcluster. Defaults to None.
        """
        self._coords = coords
        self._molecule = molecule
        self._flavor = flavor
        self._branching_coord_in_cell = branching_coord_in_cell
        self._all_branching_coord = all_branching_coord

    def __eq__(self, other: "NetNode") -> bool:  # noqa: F821 - forward reference
        """Check if two NetNodes are equal.

        Two NetNodes are equal if they have the same coordinates.

        Args:
            other (NetNode): The other NetNode to compare to.

        Returns:
            bool: True if the two NetNodes are equal, False otherwise.
        """
        return self._coords == other._coords

    def is_periodic_image(
        self, other: "NetNode", lattice: Lattice, tolerance: float = 1e-8
    ):  # noqa: F821 - forward reference
        return is_periodic_image(self._coords, other._coords, lattice, tolerance)

    def periodic_image(self, other: "NetNode", lattice: Lattice):  # noqa: F821 - forward reference
        if not self.is_periodic_image(other, lattice):
            raise ValueError("Nodes are not periodic images")
        frac_self = lattice.get_fractional_coords(self._coords)
        frac_other = lattice.get_fractional_coords(other._coords)
        image = frac_other / frac_self
        return image.astype(int)

    def __repr__(self) -> str:
        """Return a string representation for a NetNode."""
        return "NetNode: {:.2f} {:.2f} {:.2f} {}".format(
            self._coords[0], self._coords[1], self._coords[2], self._flavor
        )


def is_periodic_image(a: np.array, b: np.array, lattice: Lattice, tolerance: float = 1e-8):
    a_frac_coords = lattice.get_fractional_coords(a)
    b_frac_coords = lattice.get_fractional_coords(b)
    frac_diff = pbc_diff(a_frac_coords, b_frac_coords)
    return np.allclose(frac_diff, [0, 0, 0], atol=tolerance)


class Net:
    def __init__(self, nodes: Dict[str, Tuple[NetNode, int]], edges: Iterable[VoltageEdge]):
        """Initialize a Net object.

        Args:
            nodes (Dict[str, Tuple[NetNode, int]]): A dictionary of NetNode objects.
            edges (Iterable[VoltageEdge]): An iterable of VoltageEdge objects.
        """
        self.nodes = nodes
        self.edges = edges

        self._remove_2c()

    @cached_property
    def rcsr_code(self):
        """Return the RCSR code of the net."""
        return run_systre(self.get_systre_string())["rcsr_code"]

    def _construct_nx_multigraph(self) -> nx.MultiGraph:
        graph = nx.MultiGraph()
        for edge in self.edges:
            graph.add_edge(edge.n1, edge.n2, voltage=edge.voltage, vector=edge.vector)
        return graph

    def get_systre_string(self) -> str:
        """Return a PERIODIC_GRAPGH systre representation of the net."""
        lines = []
        template = "   {start} {end} {voltage_0} {voltage_1} {voltage_2}"
        header = "PERIODIC_GRAPH\nID moffragmentor\nEDGES"
        for edge in self.edges:
            lines.append(
                template.format(
                    start=edge.n1,
                    end=edge.n2,
                    voltage_0=edge.voltage[0],
                    voltage_1=edge.voltage[1],
                    voltage_2=edge.voltage[2],
                )
            )
        tail = "END"
        return "\n".join([header, "\n".join(lines), tail])

    def get_cgd(self, lattice) -> str:
        """Return a CGD systre representation of the net."""
        header = "CRYSTAL\nNAME\nGROUP P1"
        cell_line = "CELL {cell.a} {cell.b} {cell.c} {cell.alpha} {cell.beta} {cell.gamma}".format(
            cell=lattice
        )
        tail = "END"
        lines = []
        node_template = "NODE {i} {cn} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}"
        edge_template = "EDGE {coords_a[0]:.2f} {coords_a[1]:.2f} {coords_a[2]:.2f} {coords_b[0]:.2f} {coords_b[1]:.2f} {coords_b[2]:.2f}"  # noqa: E501
        g = self._construct_nx_multigraph()
        node_list = [n for n, i in self.nodes.values()]
        for i, node in enumerate(node_list):
            lines.append(
                node_template.format(
                    i=i, cn=g.degree(i), coords=lattice.get_fractional_coords(node._coords)
                )
            )
        for edge in self.edges:
            b_coord = lattice.get_fractional_coords(node_list[edge.n2]._coords)
            b_coord = b_coord + edge.n2_image
            a_coord = lattice.get_fractional_coords(node_list[edge.n1]._coords)
            lines.append(edge_template.format(coords_a=a_coord, coords_b=b_coord))
        return "\n".join([header, cell_line, "\n".join(lines), tail])

    @property
    def _edge_dict(self) -> Dict[str, List[VoltageEdge]]:
        edge_dict = defaultdict(list)
        for edge in self.edges:
            edge_dict[edge.n1].append(edge)
            edge_dict[edge.n2].append(edge)
        return edge_dict

    def _remove_2c(self) -> None:
        """Remove all vertices that have degree 2.

        Those "vertices" are not really nodes but edge centers.
        This method overrides the self.nodes dictionary
        and the self.edges list.
        """
        # not sure that i really want to re-index the nodes
        nodes_to_remove = []
        edges_to_reconnect = []
        # we make new lists to also re-index
        new_nodes = OrderedDict()
        new_edges = []

        # First, we use the edge dict to find the 2c centers
        # we drop those nodes and their edges, but reconnect the
        # vertices to which they were bound to
        # i.e. for every removed 2c, we remove 1 vertex, 2 edges
        # and add 1 vertex.

        edge_dict = self._edge_dict
        for k, v in edge_dict.items():
            if len(v) == 2:
                nodes_to_remove.append(k)
                edges_to_reconnect.append((k, v))

        # given that we know now which vertices are 2c we copy all the good o
        for node_key, (node, index) in self.nodes.items():
            if node_key not in nodes_to_remove:
                new_nodes[node_key] = (node, index)

        # now, prune the edges that are involved with those nodes
        for edge in self.edges:
            if edge.n1 in nodes_to_remove or edge.n2 in nodes_to_remove:
                continue
            new_edges.append(edge)

        # now, we need to reconnect the edges
        for key, edge_list in edges_to_reconnect:
            original_index_that_is_now_dropped = key
            terminal_1 = (
                edge_list[0].n1
                if edge_list[0].n1 != original_index_that_is_now_dropped
                else edge_list[0].n2
            )
            terminal_2 = (
                edge_list[1].n1
                if edge_list[1].n1 != original_index_that_is_now_dropped
                else edge_list[1].n2
            )
            total_vector = edge_list[0].vector + edge_list[1].vector

            terminal_1_image = (
                edge_list[0].n1_image
                if edge_list[0].n1 == original_index_that_is_now_dropped
                else edge_list[0].n2_image
            )
            terminal_2_image = (
                edge_list[1].n1_image
                if edge_list[1].n1 == original_index_that_is_now_dropped
                else edge_list[1].n2_image
            )

            new_egde = VoltageEdge(
                total_vector, terminal_1, terminal_2, terminal_1_image, terminal_2_image
            )
            new_edges.append(new_egde)

            self.nodes = new_nodes
            self.edges = new_edges


def add_node_to_collection(
    node: NetNode, lattice: Lattice, collection: Dict[str, Tuple[NetNode, int]]
) -> Tuple[bool, np.ndarray, int]:
    """Add a node to a collection.

    Return a tuple of (is_new, image, index).
    We find image using the call .periodic_image(node, lattice).

    Args:
        node (NetNode): The node to add to the collection.
        lattice (Lattice): The lattice to use for finding the periodic image.
        collection (Dict[str, Tuple[NetNode, int]]): The collection to add the node to.]=

    Returns:
        Tuple[bool, np.ndarray, int]: A tuple of (is_new, image, index).
    """
    image = None
    for n, i in collection.values():
        if node.is_periodic_image(n, lattice):
            image = node.periodic_image(n, lattice)
            return False, image, i

    is_new = True
    collection[str(node)] = (node, len(collection))

    return is_new, image, collection[str(node)][1]


def contains_edge(edge: VoltageEdge, collection: Iterable[VoltageEdge]) -> bool:
    """Return True if the edge is in the collection."""
    for e in collection:
        if e == edge:
            return True

    return False


def branching_index_match(linker, metal_cluster, lattice: Lattice, tolerance: float = 1e-8):
    """Check if the branching index of the linker is the same as of the metal cluster."""
    for branching_coordinate in metal_cluster.branching_coords:

        frac_a = lattice.get_fractional_coords(branching_coordinate)
        for coord in linker.branching_coords:
            frac_b = lattice.get_fractional_coords(coord)
            # wrap the coordinates to the unit cell
            # frac_b = frac_b- np.floor(frac_b)
            distance, image = lattice.get_distance_and_image(frac_a, frac_b)

            if distance < tolerance:
                return True, image
    return False, None


def which_periodic_images_of_images(
    a: np.array, b: np.array, lattice: Lattice, tolerance: float = 1e-8
):
    """Return the multiplier by which b must be multiplied to be a"""
    potential_replicas = [
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
        (1, 1, 0),
        (1, -1, 0),
        (-1, 1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (1, 0, -1),
        (-1, 0, 1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, 1, -1),
        (0, -1, 1),
        (0, -1, -1),
    ]

    a_frac_coords = lattice.get_fractional_coords(a)
    b_frac_coords = lattice.get_fractional_coords(b)

    found_replicas = []

    # TODO: potentially speed up with numba
    for replica_a in potential_replicas:
        for replica_b in potential_replicas:
            if np.allclose(
                a_frac_coords + np.array(replica_a),
                b_frac_coords + np.array(replica_b),
                atol=tolerance,
            ):
                found_replicas.append((replica_a, replica_b))

    return len(found_replicas) > 0, found_replicas


def in_cell(node: NetNode, lattice: Lattice) -> Tuple[bool, List[np.array]]:
    """
    Check if a node is in the unit cell.

    It does so by checking if any branching coordinate is in the unit cell.

    Args:
        node (NetNode): The node to check.
        lattice (Lattice): The lattice to use for checking.

    Returns:
        Tuple[bool, List[np.array]]: A tuple of (is_in_cell, coordinates in cell).
    """
    branching_coords_in_cell = []
    for branching_coord in node.branching_coords:
        branching_coord = lattice.get_fractional_coords(branching_coord)
        if np.all(branching_coord <= 1) & np.all(branching_coord >= 0):
            branching_coords_in_cell.append(branching_coord)
    if len(branching_coords_in_cell) == 0:
        return False, branching_coords_in_cell
    return True, branching_coords_in_cell


def has_edge(metal_cluster: SBU, linker: SBU, lattice: Lattice) -> Tuple[bool, List[np.array]]:
    """Check if the linker has an edge to the metal cluster.

    It does so by checking if there is any connecting between the branching indices
    - either in the cell or between periodic replica.

    Args:
        metal_cluster (SBU): The metal cluster to check.
        linker (SBU): The linker to check.
        lattice (Lattice): The lattice to use for checking.

    Returns:
        Tuple[bool, List[np.array]]: A tuple of (is_connected, image).
    """
    images = []
    replica_sums = []
    for metal_coord in metal_cluster._all_branching_coord:
        for linker_coord in linker._all_branching_coord:
            match, images_ = which_periodic_images_of_images(metal_coord, linker_coord, lattice)
            if match:
                for image_a, image_b in images_:
                    images.append((image_a, image_b))
                    replica_sums.append(np.abs(image_a).sum() + np.abs(image_b).sum())
    return len(images) > 0, images


def build_net(metal_clusters: NodeCollection, linkers: LinkerCollection, lattice: Lattice) -> Net:
    """Given a metal cluster and linker collection from the fragmentation, build a net.

    Args:
        metal_clusters (NodeCollection): The metal clusters from the fragmentation.
        linkers (LinkerCollection): The linkers from the fragmentation.
        lattice (Lattice): The lattice of the MOF (e.g. MOF.lattice).

    Returns:
        Net: An object representing the combinatorial net.
    """
    found_metal_nodes = OrderedDict()
    found_linker_nodes = OrderedDict()
    found_edges = []

    for metal_cluster in metal_clusters:

        found_in_cell, coordinates = in_cell(metal_cluster, lattice)
        if not found_in_cell:
            continue
        # there seems to be a mismatch between branching coords and the centers in some cases.
        # ToDo: fix this at the source of the problem.
        center = lattice.get_fractional_coords(metal_cluster.center)
        center = center - np.floor(center)
        center = lattice.get_cartesian_coords(center)
        net_node = NetNode(
            center,
            molecule=metal_cluster,
            branching_coord_in_cell=coordinates,
            all_branching_coord=metal_cluster.branching_coords,
            flavor="metalcluster",
        )

        _, metal_image, metal_index = add_node_to_collection(net_node, lattice, found_metal_nodes)

    for linker in linkers:

        found_in_cell, linker_coordinates = in_cell(linker, lattice)
        if not found_in_cell:
            continue
        center = lattice.get_fractional_coords(linker.center)
        center = center - np.floor(center)
        center = lattice.get_cartesian_coords(center)
        net_node = NetNode(
            center,
            molecule=linker,
            branching_coord_in_cell=linker_coordinates,
            all_branching_coord=linker.branching_coords,
            flavor="linker",
        )
        _, metal_image, metal_index = add_node_to_collection(net_node, lattice, found_linker_nodes)

    linker_index_offset = len(found_metal_nodes)
    for metal_node, metal_index in found_metal_nodes.values():
        for linker_node, linker_index in found_linker_nodes.values():
            at_least_one_edge, images = has_edge(metal_node, linker_node, lattice)
            if at_least_one_edge:
                #   def __init__(self, vector: np.ndarray, n1: int, n2: int, n1_image: tuple, n2_image: tuple):
                for image_a, image_b in images:
                    if image_a == (0, 0, 0) or image_b == (0, 0, 0):
                        metal_center = lattice.get_fractional_coords(metal_node._coords) + image_a
                        linker_center = lattice.get_fractional_coords(linker_node._coords) + image_b
                        edge = VoltageEdge(
                            linker_center - metal_center,
                            metal_index,
                            linker_index + linker_index_offset,
                            image_a,
                            image_b,
                        )
                        if not contains_edge(edge, found_edges):
                            found_edges.append(edge)

    for linker_node, linker_index in found_linker_nodes.values():
        for metal_node, metal_index in found_metal_nodes.values():
            at_least_one_edge, images = has_edge(linker_node, metal_node, lattice)
            if at_least_one_edge:
                for image_a, image_b in images:
                    if image_a == (0, 0, 0) or image_b == (0, 0, 0):
                        metal_center = lattice.get_fractional_coords(metal_node._coords) + image_b
                        linker_center = lattice.get_fractional_coords(linker_node._coords) + image_a
                        edge = VoltageEdge(
                            linker_center - metal_center,
                            metal_index,
                            linker_index + linker_index_offset,
                            image_a,
                            image_b,
                        )

                        if not contains_edge(edge, found_edges):
                            found_edges.append(edge)

    found_metal_nodes.update(found_linker_nodes)
    return Net(found_metal_nodes, found_edges)
