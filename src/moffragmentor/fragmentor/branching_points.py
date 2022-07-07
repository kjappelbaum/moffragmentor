"""
Routines for finding branching points in a structure graph of a MOF.

Note that those routines do not work for other reticular materials 
as they assume the presence of a metal.
"""
from typing import List 
from more_itertools import pairwise
import numpy as np
import networkx as nx 
from pymatgen.core import Structure
from moffragmentor.utils import _not_relevant_structure_indices

from moffragmentor.fragmentor.molfromgraph import get_subgraphs_as_molecules

def get_distances_to_metal(mof: "MOF", site: int) -> List[float]:
    """For a given site, return the distances to all metals in the MOF."""
    distances = []
    for i in mof.metal_indices:
        distances.append(mof.structure.get_distance(site, i))
    return distances


def has_bridge_in_path(mof: "MOF", path: List[int])->bool:   
    """Return True if the path contains a bridge."""
    for edge in pairwise(path):
        if mof._leads_to_terminal(edge):
            return True
    return False



def get_two_edge_paths_from_site(mof: "MOF", site: int) -> List[List[int]]: 
    """
    Return all two edge paths from a site.
    
    Example:
        >>> mof = MOF(...)
        >>> get_two_edge_paths_from_site(mof, 0)
        [[0, 1, 2], [0, 3, 4], [0, 5, 6]]
    """
    paths = []
    for i in mof.get_neighbor_indices(site):
        for j in mof.get_neighbor_indices(i):
            if j != site:
                paths.append([site, i, j])

    return paths


def has_metal_in_path(mof: "MOF", path: List[int])->bool:
    """Return True if the path contains a metal.""" 
    for site in path:
        if site in mof.metal_indices:
            return True
    return False


def has_non_bridge_path_with_metal(mof: "MOF", site: int)->bool:
    """Return True if the MOF has a non-bridge path with a meta.l"""
    for path in get_two_edge_paths_from_site(mof, site):
        if not has_bridge_in_path(mof, path) and has_metal_in_path(mof, path):
            return True
    return False



def _is_branch_point(mof: "MOF", index: int, allow_metal: bool = False) -> bool:
    """The branch point definition is key for splitting MOFs
    into linker and nodes. Branch points are here defined as points
    that have at least three connections that do not lead to a tree or
    leaf node.

    Args:
        mof: MOF object
        index (int): index of site that is to be probed
        allow_metal (bool): If True it does not perform
            this check for metals (and just return False). Defaults to False.

    Returns:
        bool: True if this is a branching index
    """
    connected_sites = mof.get_neighbor_indices(index)

    if len(connected_sites) < 3:
        return False

    if not allow_metal:
        if index in mof.metal_indices:
            return False

    # lets store all the info in a numpy array 
    sites = []
    for connected_site in connected_sites:
        leads_to_terminal =  mof._leads_to_terminal((index, connected_site))
        is_terminal =  mof._is_terminal(connected_site)
        in_metal_indices = connected_site in mof.metal_indices
        non_bridge_metal = has_non_bridge_path_with_metal(mof, index)
        sites.append([leads_to_terminal, is_terminal, in_metal_indices, non_bridge_metal])
    sites = np.array(sites).astype(bool)
    non_terminal_metal_connections = np.sum(~sites[:,0] & ~sites[:,1] &  sites[:,3])
    non_terminal_non_metal_connections = np.sum(~sites[:,0] & ~sites[:,1] & ~sites[:,2])
    terminal_metal_connections = np.sum((sites[:,0] | sites[:,1]) &  sites[:,2])        
  
    if terminal_metal_connections > 0:
        return False 
    if (non_terminal_non_metal_connections >= 2) & (non_terminal_metal_connections > 0):
        return True

    return False


def filter_branch_points(mof: "MOF", branching_indices) -> List[int]:
    """
    Return a list of all branching points in the MOF.
    """
 
    # now, check if there are connected branching points
    # we need to clean that up as it does not make sense for splitting and the clustering 
    # algorithm will not work (i believe).

    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    to_delete = _not_relevant_structure_indices(mof.structure, branching_indices)
    graph_.remove_nodes(to_delete)
    mols, graphs, idx, centers, coordinates = get_subgraphs_as_molecules(graph_, return_unique=False)

    verified_indices = []

    for mol, graph, index in zip(mols, graphs, idx):
        if len(index) == 1:
            verified_indices.extend(index)
        else: 
            rank = rank_by_metal_distance(index, mof)
            if rank is not None:
                verified_indices.append(rank)
            else: 
                verified_indices.append(cluster_nodes(graph, index, mof))
    
    return verified_indices

def has_bond_to_metal(mof: "MOF", index: int) -> bool:
    """Return True if the site has a bond to a metal."""
    for i in mof.get_neighbor_indices(index):
        if i in mof.metal_indices:
            return True
    return False

def rank_by_metal_distance(idx, mof):
    """
    Rank the indices by the distance to the metal.
    """
    idx = np.array(idx)
    # ToDo: should use the path length
    # in this case, it simply means if there is one 
    # direct single bond 
    distances = -np.array([has_bond_to_metal(mof, i) for i in idx]).astype(int)
    # this works by default in ascending order as we need it
    order = np.argsort(distances)
    sorted_distances = distances[order]
    if sorted_distances[0] < sorted_distances[1]:
        return idx[order][0]
    else: 
        return None

def cluster_nodes(graph: nx.Graph, original_indices: List[int], mof: "MOF") -> int: 
    g = nx.Graph(graph).to_undirected()
    terminal_nodes = []
    for node in g.nodes:
        if g.degree[node] == 1:
            terminal_nodes.append(node)
        if g.degree[node] > 2:
            raise ValueError('Unsupported connectivity of branching points.')
    
    if len(terminal_nodes) >2:
        raise ValueError('Unsupported connectivity of branching points.')
    
    path = nx.shortest_path(g, terminal_nodes[0], terminal_nodes[1])
    # if odd return the middle node
    if len(path) % 2 == 1:
        return original_indices[path[int(len(path)/2)]]
    
    # else we currently have no good solution 
    # perhaps the best would be to insert a dummy node in the middle
    # but for now we will perhaps simply return the the node closest to a metal 
    # (in the Euclidean embedding)) 
    min_distance = np.inf
    min_node = None
    for node in path:
        distances = get_distances_to_metal(mof, node)
        if np.min(distances) < min_distance:
            min_distance = np.min(distances)
            min_node = node
    return original_indices[min_node]
        
