# -*- coding: utf-8 -*-
"""Methods for the fragmentation of MOFs"""
from collections import namedtuple

from moffragmentor.fragmentor.filter import in_hull

from ..net import NetEmbedding
from ..utils import _get_metal_sublist
from .filter import point_in_mol_coords
from .linkerlocator import create_linker_collection
from .nodelocator import NodelocationResult, create_node_collection, find_node_clusters
from .solventlocator import (
    get_all_bound_solvent_molecules,
    get_floating_solvent_molecules,
)

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    ["nodes", "linkers", "bound_solvent", "unbound_solvent", "net_embedding"],
)


def run_fragmentation(mof) -> FragmentationResult:
    unbound_solvent = get_floating_solvent_molecules(mof)
    # Find nodes
    node_result = find_node_clusters(mof)
    node_collection = create_node_collection(mof, node_result)
    # Find bound solvent
    bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
    # Filter the linkers (valid linkers have at least two branch points)
    linker_collection, edge_dict = create_linker_collection(
        mof, node_result, node_collection, unbound_solvent
    )

    # filter for metal in linker case:
    ok_node = []
    need_rerun = False

    for i, node in enumerate(node_result.nodes):
        metal_in_node = _get_metal_sublist(node, mof.metal_indices)
        node_ok = True
        # ToDo: check and think if this can handle the general case
        # it should, at least if we only look at the metals
        if len(metal_in_node) == 1:
            for linker in linker_collection:
                if point_in_mol_coords(
                    mof.cart_coords[metal_in_node[0]],
                    mof.cart_coords[linker._original_indices],
                    mof.lattice,
                ):
                    print(f"{i} in hull")
                    if (
                        len(
                            set(mof.get_neighbor_indices(metal_in_node[0]))
                            & set(linker._original_indices)
                        )
                        > 1
                    ):
                        need_rerun = True
                        node_ok = False
                        break
            if node_ok:
                ok_node.append(i)
        else:
            ok_node.append(i)

    if need_rerun:
        selected_nodes = [node_result.nodes[i] for i in ok_node]
        node_result = NodelocationResult(
            selected_nodes, node_result.branching_indices, node_result.connecting_paths
        )
        node_collection = create_node_collection(mof, node_result)
        # Find bound solvent
        bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
        # Filter the linkers (valid linkers have at least two branch points)
        linker_collection, edge_dict = create_linker_collection(
            mof, node_result, node_collection, unbound_solvent
        )

    # Now, get the net
    net_embedding = NetEmbedding(
        linker_collection, node_collection, edge_dict, mof.lattice
    )
    fragmentation_results = FragmentationResult(
        node_collection,
        linker_collection,
        bound_solvent,
        unbound_solvent,
        net_embedding,
    )

    return fragmentation_results
