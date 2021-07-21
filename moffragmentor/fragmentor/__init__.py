# -*- coding: utf-8 -*-
"""Methods for the fragmentation of MOFs"""
from collections import namedtuple

from ..net import NetEmbedding
from .locator import (create_linker_collection, create_node_collection,
                      find_node_clusters, get_all_bound_solvent_molecules)
from .splitter import get_floating_solvent_molecules

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
