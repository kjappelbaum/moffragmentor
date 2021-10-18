# -*- coding: utf-8 -*-
"""Functions to describe nets encoded as pymatgen structures"""
import pathlib
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import LocalStructOrderParams, MinimumDistanceNN
from pymatgen.core import Lattice, Structure

from . import ALL_LSOP

MIN_DISTANCE_NN = MinimumDistanceNN()
LSOP = LocalStructOrderParams(types=ALL_LSOP)


def get_basic_net_features(structure: Structure) -> dict:
    """Return basic characteristics of a structure.
    The structure represents the net.

    Args:
        structure (Structure): pymatgen Structure
            representing the net

    Returns:
        dict: global features characterising
            the net/structure
    """
    return {
        "density": structure.density,
        "volume": structure.volume,
        "sites": len(structure),
    }


def get_neighbor_indices(structure_graph: StructureGraph, index: int) -> List[int]:
    return [site.index for site in structure_graph.get_connected_sites(index)]


def get_lsop_for_site(structure: Structure, index: int, neighbors: List[int]) -> list:
    """Return local structure order parameters for site index.

    Args:
        structure (Structure): pymatgen Structure
        index (int): index of the site for which we compute the
            descriptors
        neighbors (List[int]): Neighbors of site that are considered
            for the calculation of the order parameter

    Returns:
        list: local structure order parameters
    """
    descriptors = LSOP.get_order_parameters(structure, index, indices_neighs=neighbors)
    return descriptors


def cgd_to_structure(  # pylint:disable=too-many-locals
    filename: Union[str, pathlib.Path],
    node_symbol: str = "C",
    edge_center_symbol: str = "O",
) -> Tuple[str, Structure]:
    """Code stolen from PORMAKE.
    Parse a cgd file and return a pymatgen Structure

    Args:
        filename (Union[str, pathlib.Path]): path to the cgd file
        node_symbol (str, optional): Element in the pymatgen Structure
            that will be used for node. Defaults to "C".
        edge_center_symbol (str, optional): Element in pymatgen Structure
            that will be used for edge center. Defaults to "O".

    Returns:
        Tuple[str, Structure]: RCSR code of net, pymatgen Structure
    """
    with open(filename, "r", encoding="utf8") as f:
        # Neglect "CRYSTAL" and "END"
        lines = f.readlines()[1:-1]
    lines = [line for line in lines if not line.startswith("#")]

    # Get topology name.
    name = lines[0].split()[1]
    # Get spacegroup.
    spacegroup = lines[1].split()[1]

    # Get cell paremeters and expand cell lengths by 10.
    cellpar = np.array(lines[2].split()[1:], dtype=np.float32)

    # Parse node information.
    node_positions = []
    coordination_numbers = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "NODE":
            continue

        coordination_number = int(tokens[2])
        pos = [float(r) for r in tokens[3:]]
        node_positions.append(pos)
        coordination_numbers.append(coordination_number)

    node_positions = np.array(node_positions)
    # coordination_numbers = np.array(coordination_numbers)

    # Parse edge information.
    edge_center_positions = []
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE":
            continue

        pos_i = np.array([float(r) for r in tokens[1:4]])
        pos_j = np.array([float(r) for r in tokens[4:]])

        edge_center_pos = 0.5 * (pos_i + pos_j)
        edge_center_positions.append(edge_center_pos)

    # New feature. Read EDGE_CENTER.
    for line in lines[3:]:
        tokens = line.split()

        if tokens[0] != "EDGE_CENTER":
            continue

        edge_center_pos = np.array([float(r) for r in tokens[1:]])
        edge_center_positions.append(edge_center_pos)

    edge_center_positions = np.array(edge_center_positions)

    # Carbon for nodes, oxygen for edges.
    n_nodes = node_positions.shape[0]
    n_edges = edge_center_positions.shape[0]
    species = np.concatenate(
        [
            np.full(shape=n_nodes, fill_value=node_symbol),
            np.full(shape=n_edges, fill_value=edge_center_symbol),
        ]
    )

    coords = np.concatenate([node_positions, edge_center_positions], axis=0)

    # Pymatget can handle : indicator in spacegroup.
    # Mark symmetrically equivalent sites.
    node_types = [i for i, _ in enumerate(node_positions)]
    edge_types = [-(i + 1) for i, _ in enumerate(edge_center_positions)]
    site_properties = {
        "type": node_types + edge_types,
        "cn": coordination_numbers + [2 for _ in edge_center_positions],
    }

    # I don't know why pymatgen can't parse this spacegroup.
    if spacegroup == "Cmca":
        spacegroup = "Cmce"

    structure = Structure.from_spacegroup(
        sg=spacegroup,
        lattice=Lattice.from_parameters(*cellpar),
        species=species,
        coords=coords,
        site_properties=site_properties,
    )

    return name, structure, node_types, edge_types


def get_distance_descriptors(
    structure: Structure, site: int, neighbors: List[int]
) -> dict:
    """Return distance descriptors for site index.

    Args:
        structure (Structure): pymatgen Structure
        index (int): index of the site for which we compute the
            descriptors
        neighbors (List[int]): Neighbors of site that are considered
            for the calculation of the order parameter

    Returns:
        list: distance descriptors
    """
    try:
        distances = []
        for neighbor in neighbors:
            distances.append(structure.get_distance(site, neighbor))
        return {
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "mean_distance": np.mean(distances),
            "median_distance": np.mean(distances),
            "std_distance": np.std(distances),
        }
    except Exception:  # pylint:disable=broad-except
        return {
            "min_distance": 0,
            "max_distance": 0,
            "mean_distance": 0,
            "median_distance": 0,
            "std_distance": 0,
        }


def get_bb_info(structure: Structure) -> dict:
    """Compile vertex/edge features for a
    net represented as pymatgen Structure

    Args:
        structure (Structure): net representation
            as pymatgen Structure

    Returns:
        dict: information about vertices/edges
            and descriptors
    """
    types = []
    cns = []
    type_dict = defaultdict(list)
    bb_types = {"nodes": set(), "linkers": set()}
    lsop_features = {}
    distance_features = {}

    structure_graph = StructureGraph.with_local_env_strategy(structure, MIN_DISTANCE_NN)

    for i, site in enumerate(structure):
        p = site.as_dict()["properties"]
        t = p["type"]
        types.append(t)
        cns.append(p["cn"])
        type_dict[p["type"]].append(i)
        if t > 0:
            bb_types["nodes"].add(t)
        else:
            bb_types["linkers"].add(t)

    for k, v in type_dict.items():
        neighbors = get_neighbor_indices(structure_graph, v[0])
        lsop_features[k] = get_lsop_for_site(structure, v[0], neighbors)
        distance_features[k] = get_distance_descriptors(structure, v[0], neighbors)

    return {
        "types": types,
        "cns": cns,
        "type_indices": type_dict,
        "bb_types": bb_types,
        "lsop_features": lsop_features,
        "distance_features": distance_features,
    }


def get_net_descriptors(netfile: Union[str, pathlib.Path]) -> dict:
    """Create descriptors for net encoded in a cgd file

    Args:
        netfile (Union[str, pathlib.Path]): Path to cgd file

    Returns:
        dict: dictionary with global, local descriptors and
            basic building block information
    """
    rcsr_code, structure, node_types, edge_types = cgd_to_structure(netfile)
    global_descriptors = get_basic_net_features(structure)
    bb_descriptors = get_bb_info(structure)
    descriptors = {**global_descriptors, **bb_descriptors}
    descriptors["rcsr_code"] = rcsr_code
    descriptors["node_types"] = node_types
    descriptors["edge_types"] = edge_types

    return descriptors
