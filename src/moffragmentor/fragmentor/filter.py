# -*- coding: utf-8 -*-
"""This module contains functions that perform filtering on
indices or fragments obtained from the other fragmentation modules"""

import networkx as nx
import numpy as np
from loguru import logger
from scipy.spatial.qhull import Delaunay, QhullError  # pylint:disable=no-name-in-module

from ..utils import unwrap


def bridges_across_cell(mof, indices) -> bool:
    """Check if a molecule of indices bridges across the cell"""
    bridges = {}

    for index in indices:
        for neighbor_site in mof.structure_graph.get_connected_sites(index):
            if not neighbor_site.index in bridges:
                bridges[neighbor_site.index] = neighbor_site.jimage
            else:
                if (bridges[neighbor_site.index] != neighbor_site.jimage) & (neighbor_site.index in mof.metal_indices):
                    return True

    return False

def point_in_mol_coords(point, points, lattice):
    new_coords = unwrap(np.append(points, [point], axis=0), lattice)
    return in_hull(new_coords[-1], new_coords[:-1]) or in_hull(points[-1], points[:-1])

def in_hull(pointcloud, hull):
    """
    Test if points in `p` are in `hull`b

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed

    https://stackoverflow.com/a/16898636
    """

    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
    except QhullError:

        if len(pointcloud) < 5:
            logger.warning("Too few points to compute Delaunay triangulation")
            return False

        hull = Delaunay(hull, qhull_options="QJ")


    return hull.find_simplex(pointcloud) >= 0


def _filter_branch_points(
    branch_indices: list, metal_indices: list, graph: nx.Graph
) -> list:
    """In a MOF structure there might be many sites with
    more than three neighbors that do not lead to a tree or
    leaf node. The relevant branching indices are those that
    are not between other ones.
    That is, we want to filter out branching indices for which the shortest
    path to a metal goes via another branching index.

    Args:
        branch_indices (list): candidate list of branching indices
        metal_indices (list): metal indices
        graph (nx.Graph): graph on which the nodes can be access using the
            items on the branch_indices and metal_indices lists

    Returns:
        list filtered branching indices
    """
    filtered_indices = []
    for branch_index in branch_indices:
        shortest_path = _shortest_path_to_metal(branch_index, metal_indices, graph)
        if not _has_branch_index_in_path(shortest_path, branch_indices):
            filtered_indices.append(branch_index)
    return filtered_indices


def _has_branch_index_in_path(path, branch_indices):
    for metal_index in branch_indices:
        if metal_index in path:
            return True
    return False


def _shortest_path_to_metal(branch_index, metal_indices, graph):
    paths = []

    for metal_index in metal_indices:
        path = nx.shortest_path(  # pylint:disable=unexpected-keyword-arg, no-value-for-parameter
            graph, source=branch_index, target=metal_index
        )
        paths.append(path)

    shortest_path = min(paths, key=len)

    return shortest_path[1:]
