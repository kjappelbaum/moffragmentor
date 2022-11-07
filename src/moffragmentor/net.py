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
from backports.cached_property import cached_property
from loguru import logger
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import IMolecule, Lattice, Molecule, Structure
from pymatgen.util.coord import pbc_diff

from moffragmentor.sbu import SBU
from moffragmentor.sbu.linkercollection import LinkerCollection
from moffragmentor.sbu.nodecollection import NodeCollection

from .utils.systre import _get_systre_input_from_pmg_structure_graph, run_systre


def is_3d_parallel(v1: np.array, v2: np.array, eps: float = 1e-5) -> bool:
    if np.allclose(v1, 0) and not np.allclose(v2, 0):
        return False
    if np.allclose(v2, 0) and not np.allclose(v1, 0):
        return False
    cp = np.cross(v1, v2)
    return np.allclose(cp, np.zeros(3), atol=eps)


def get_image_if_on_edge(frac_coords):
    # exactly one --> return the -1 image
    images = np.zeros(3, dtype=int)
    is_exactly_one = np.argwhere(np.abs(frac_coords - 1) < 1e-4)
    if is_exactly_one.size > 0:
        images[is_exactly_one] = -1
    is_exactly_zero = np.argwhere(np.abs(frac_coords) < 1e-4)
    if is_exactly_zero.size > 0:
        images[is_exactly_zero] = 1
    return images


def sanitize_graph(structure_graph):
    # if there is a node with only one edge, it is probably because the binding partner is at an edge or corner of the structure
    # that is, at least one fractional coordinate is 0 or 1
    # in this case, we add a bond to the image of this site
    new_edges = []
    for i, _ in enumerate(structure_graph.structure):
        connected_sites = structure_graph.get_connected_sites(i)
        if len(connected_sites) == 1:
            frac_coords = connected_sites[0].site.frac_coords
            images = get_image_if_on_edge(frac_coords)
            if np.any(images != 0):
                new_edges.append((i, connected_sites[0].index, images))

    for edge in new_edges:
        structure_graph.add_edge(edge[0], edge[1], to_jimage=edge[2])


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
        """
        Return `True` if the two nodes are periodic images of each other.

        Args:
            other ("NetNode"): NetNode
            lattice (Lattice): The lattice of the crystal.
            tolerance (float): The tolerance for the distance between the two nodes.

        Returns:
            A boolean value.
        """
        return is_periodic_image(self._coords, other._coords, lattice, tolerance)

    def periodic_image(self, other: "NetNode", lattice: Lattice):  # noqa: F821 - forward reference
        """`periodic_image` returns the periodic image of `other` w.r.t to `self`.

        Args:
            other ("NetNode"): NetNode
            lattice (Lattice): The lattice of the crystal.

        Returns:
            np.array: The periodic image of the other node in the lattice.

        Raises:
            ValueError: If the two nodes are not periodic images of each other.
        """
        if not self.is_periodic_image(other, lattice):
            raise ValueError("Nodes are not periodic images")
        frac_self = lattice.get_fractional_coords(self._coords)
        frac_other = lattice.get_fractional_coords(other._coords)
        return which_periodic_image(frac_self, frac_other, lattice)[-1][0][1]

    def __repr__(self) -> str:
        """Return a string representation for a NetNode."""
        return "NetNode: {:.2f} {:.2f} {:.2f} {}".format(
            self._coords[0], self._coords[1], self._coords[2], self._flavor
        )


def is_periodic_image(
    a: np.array, b: np.array, lattice: Lattice, tolerance: float = 1e-8, is_frac: bool = False
) -> bool:
    """
    Return `True` if `a` and `b` are periodic images of each other.

    Args:
        a (np.array): np.array, b: np.array: The two points you want to compare.
        b (np.array): np.array
        lattice (Lattice): The lattice of the structure.
        tolerance (float): The tolerance for the difference between
            the fractional coordinates of the two points.
        is_frac (bool): Whether the points are in fractional coordinates.

    Returns:
        bool: The function is_periodic_image returns a boolean value.
    """
    if not is_frac:
        a_frac_coords = lattice.get_fractional_coords(a)
        b_frac_coords = lattice.get_fractional_coords(b)
    else:
        a_frac_coords = a
        b_frac_coords = b
    frac_diff = pbc_diff(a_frac_coords, b_frac_coords)
    return np.allclose(frac_diff, [0, 0, 0], atol=tolerance)


class Net:
    def __init__(
        self,
        nodes: Dict[str, Tuple[NetNode, int]],
        edges: Iterable[VoltageEdge],
        lattice: Lattice,
    ):
        """Initialize a Net object.

        Args:
            nodes (Dict[str, Tuple[NetNode, int]]): A dictionary of NetNode objects.
            edges (Iterable[VoltageEdge]): An iterable of VoltageEdge objects.
            lattice (Lattice): The lattice of the crystal.
        """
        self.nodes = nodes
        self.edges = edges
        self.lattice = lattice

    def get_dummy_structure(self) -> Structure:
        coords = []
        symbols = []
        for node, _ in self.nodes.values():
            if "metal" in node._flavor:
                coords.append(node._coords)
                symbols.append("Si")
            else:
                coords.append(node._coords)
                symbols.append("O")

        return Structure(
            lattice=self.lattice, species=symbols, coords=coords, coords_are_cartesian=True
        )

    def get_pmg_structure_graph(
        self, simplify: bool = True, sanitize: bool = False
    ) -> StructureGraph:
        """Return a StructureGraph object from the Net object

        Args:
            simplify (bool): Whether to simplify the graph. Defaults to True.
                If True, 2c vertices are removed.
            sanitize (bool): Whether to sanitize the graph. Defaults to False.
                If True, add edge to periodic image for sites with only one neighbor
                if this neighbor is at the egde of the unit cell.

        Returns:
            StructureGraph: The StructureGraph object.
        """
        s = self.get_dummy_structure()
        edge_dict = {}

        for edge in self.edges:
            edge_dict[
                (
                    edge.n1,
                    edge.n2,
                    (edge.n1_image[0], edge.n1_image[1], edge.n1_image[2]),
                    (edge.n2_image[0], edge.n2_image[1], edge.n2_image[2]),
                )
            ] = None

        structure_graph = StructureGraph.with_edges(s, edge_dict)
        if sanitize:
            sanitize_graph(structure_graph)
        if simplify:
            structure_graph = _simplify_structure_graph(structure_graph)
        return structure_graph

    @cached_property
    def _systre_results(self) -> dict:
        """Return the RCSR code of the net."""
        return run_systre(
            _get_systre_input_from_pmg_structure_graph(self.get_pmg_structure_graph(), self.lattice)
        )

    @property
    def rcsr_code(self) -> str:
        """Return the RCSR code of the net."""
        try:
            return self._systre_results["rcsr_code"]
        except TypeError:
            return ""

    @property
    def p(self) -> int:
        return len(self._systre_results["relaxed_node_positions"])

    @property
    def q(self) -> int:
        return len(self._systre_results["relaxed_edge_positions"])

    def _construct_nx_multigraph(self) -> nx.MultiGraph:
        graph = nx.MultiGraph()
        for edge in self.edges:
            graph.add_edge(edge.n1, edge.n2, voltage=edge.voltage, vector=edge.vector)
        return graph

    @property
    def _cns(self) -> Dict[str, int]:
        """Return a list of the number of nodes in each edge."""
        return {k: len(v) for k, v in self._edge_dict.items()}

    @property
    def _edge_dict(self) -> Dict[str, List[VoltageEdge]]:
        edge_dict = defaultdict(list)
        for edge in self.edges:
            edge_dict[edge.n1].append(edge)
            edge_dict[edge.n2].append(edge)
        return edge_dict


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
    return any(e == edge for e in collection)


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


def which_periodic_image(
    a: np.ndarray, b: np.ndarray, lattice: Lattice, tolerance: float = 1e-2, is_frac: bool = True
) -> Tuple[bool, Optional[np.array], Optional[List[Tuple]]]:
    """
    Find which periodic image of a is closest to b.

    Uses pymatgens get_distance_and_image.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.
        lattice (Lattice): The lattice to use for finding the periodic image.
        tolerance (float): The tolerance for finding the periodic image.
        is_frac (bool): Whether the vectors are in fractional coordinates.

    Returns:
        Tuple[bool, Optional[np.array], Optional[List[Tuple]]]: A tuple of (is_new, image, image_list).
    """
    if not is_frac:
        a = lattice.get_fractional_coords(a)
        b = lattice.get_fractional_coords(b)
    if is_periodic_image(a, b, lattice, tolerance, is_frac=True):
        _, image = lattice.get_distance_and_image(a, b)
        return True, a, [(np.array([0, 0, 0]), -image), (image, np.array([0, 0, 0]))]
    return False, None, None


def in_cell(node: "SBU", lattice: Lattice) -> Tuple[bool, List[np.array]]:
    """
    Check if a node is in the unit cell.

    It does so by checking if any branching coordinate is in the unit cell.

    Args:
        node (SBU): The node to check.
        lattice (Lattice): The lattice to use for checking.

    Returns:
        Tuple[bool, List[np.array]]: A tuple of (is_in_cell, coordinates in cell).
    """
    branching_coords_in_cell = []
    for branching_coord in node.branching_coords:
        branching_coord = lattice.get_fractional_coords(branching_coord)
        if np.all(branching_coord < 1) & np.all(branching_coord > 0):
            branching_coords_in_cell.append(branching_coord)
    if len(branching_coords_in_cell) == 0:
        return False, branching_coords_in_cell
    return True, branching_coords_in_cell


def com_in_cell(node: "SBU", lattice: Lattice):
    com = node.molecule.center_of_mass
    com_frac = lattice.get_fractional_coords(com)
    if np.all(com_frac < 1) & np.all(com_frac >= 0):
        return True
    return False


def has_edge(
    metal_cluster: NetNode, linker: NetNode, lattice: Lattice
) -> Tuple[bool, List[np.array]]:
    """Check if the linker has an edge to the metal cluster.

    It does so by checking if there is any connecting between the branching indices
    - either in the cell or between periodic replica.

    Args:
        metal_cluster (NetNode): The metal cluster to check.
        linker (NetNode): The linker to check.
        lattice (Lattice): The lattice to use for checking.

    Returns:
        Tuple[bool, List[np.array]]: A tuple of (is_connected, image).
    """
    images = []

    for metal_coord in metal_cluster._all_branching_coord:
        for linker_coord in linker._all_branching_coord:
            match, coord, images_ = which_periodic_image(
                metal_coord, linker_coord, lattice, is_frac=False
            )
            if match:
                for image_a, image_b in images_[:1]:
                    images.append((coord, image_a, image_b))

    return len(images) > 0, images


def build_net(
    metal_clusters: NodeCollection,
    linkers: LinkerCollection,
    lattice: Lattice,
) -> Net:
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
        # if not found_in_cell:
        #     continue
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

        _, _, metal_index = add_node_to_collection(net_node, lattice, found_metal_nodes)

    for linker in linkers:

        found_in_cell, linker_coordinates = in_cell(linker, lattice)
        # if not found_in_cell:
        #    continue
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
        _, _, metal_index = add_node_to_collection(net_node, lattice, found_linker_nodes)

    linker_index_offset = len(found_metal_nodes)

    logger.debug(f"Found {len(found_metal_nodes)} metal nodes out of {len(metal_clusters)}")
    logger.debug(f"Found {len(found_linker_nodes)} linker nodes out of {len(linkers)}")

    egde_candiates = defaultdict(list)
    for metal_node, metal_index in found_metal_nodes.values():
        for linker_node, linker_index in found_linker_nodes.values():
            at_least_one_edge, images = has_edge(metal_node, linker_node, lattice)
            if at_least_one_edge:
                for coord, image_a, image_b in images:
                    metal_center = lattice.get_fractional_coords(metal_node._coords) + image_a
                    linker_center = lattice.get_fractional_coords(linker_node._coords) + image_b
                    edge = VoltageEdge(
                        linker_center - metal_center,
                        metal_index,
                        linker_index + linker_index_offset,
                        image_a,
                        image_b,
                    )
                    egde_candiates[
                        (round(coord[0], 2), round(coord[1], 2), round(coord[2], 2))
                    ].append((edge, np.abs(image_b).sum()))

    for linker_node, linker_index in found_linker_nodes.values():
        for metal_node, metal_index in found_metal_nodes.values():
            at_least_one_edge, images = has_edge(linker_node, metal_node, lattice)
            if at_least_one_edge:
                for coord, image_a, image_b in images:
                    metal_center = lattice.get_fractional_coords(metal_node._coords) + image_b
                    linker_center = lattice.get_fractional_coords(linker_node._coords) + image_a
                    edge = VoltageEdge(
                        linker_center - metal_center,
                        metal_index,
                        linker_index + linker_index_offset,
                        image_a,
                        image_b,
                    )
                    egde_candiates[
                        (round(coord[0], 2), round(coord[1], 2), round(coord[2], 2))
                    ].append((edge, np.abs(image_b).sum()))

    edge_selection = []
    for _branching_coord, edges in egde_candiates.items():
        # branching_coord = np.array(branching_coord)
        # if np.all(branching_coord < 1) & np.all(
        #     branching_coord > 0
        # ):
        if len(edges) == 1:
            edge_selection.append(edges[0][0])
        else:
            # sort ascending by second element in tuple in the list
            edge_selection.append(sorted(edges, key=lambda x: x[1])[0][0])

    for edge in edge_selection:
        if not contains_edge(edge, found_edges):
            found_edges.append(edge)

    found_metal_nodes.update(found_linker_nodes)
    return Net(found_metal_nodes, found_edges, lattice)


def _simplify_structure_graph(structure_graph: StructureGraph) -> StructureGraph:
    """Simplifies a structure graph by removing two-connected nodes.

    We will place an edge between the nodes that were connected
    by the two-connected node.

    The function does not touch the input graph (it creates a deep copy).
    Using the deep copy simplifies the implementation a lot as we can
    add the edges in a first loop where we check for two-connected nodes
    and then remove the nodes. This avoids the need for dealing with indices
    that might change when one creates a new graph.

    Args:
        structure_graph (StructureGraph): Input structure graph.
            Usually this is a "net graph". That is, a structure graph for
            a structure in which the atoms are MOF SBUs

    Returns:
        StructureGraph: simplified structure graph, where we removed the
            two-connected nodes.
    """
    graph_copy = structure_graph.__copy__()
    to_remove = []
    added_edges = set()

    # in the first iteration we just add the edge
    # and collect the nodes to delete
    for i, _ in enumerate(structure_graph.structure):
        if structure_graph.get_coordination_of_site(i) == 2:
            if str(structure_graph.structure[i].specie) != "Si":
                indices = []
                images = []
                for neighbor in structure_graph.get_connected_sites(i):
                    indices.append(neighbor.index)
                    images.append(neighbor.jimage)
                    try:
                        graph_copy.break_edge(i, neighbor.index, neighbor.jimage)
                    except ValueError:
                        try:
                            graph_copy.break_edge(
                                neighbor.index,
                                i,
                                (-neighbor.jimage[0], -neighbor.jimage[1], -neighbor.jimage[2]),
                            )
                        except ValueError:
                            logger.warning(f"Edge {i, neighbor.index} cannot be broken")
                sorted_images = [x for _, x in sorted(zip(indices, images))]
                edge_tuple = (tuple(sorted(indices)), tuple(sorted_images))
                # in principle, this check should not be needed ...
                if edge_tuple not in added_edges:
                    added_edges.add(edge_tuple)
                    graph_copy.add_edge(indices[0], indices[1], images[0], images[1])

                to_remove.append(i)
            else:
                logger.warning(
                    "Metal cluster with low coodination number detected.\
                    Results might be incorrect."
                )

    # after we added all the edges, we can remove the nodes
    graph_copy.remove_nodes(to_remove)
    return graph_copy
