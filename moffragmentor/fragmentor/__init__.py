# -*- coding: utf-8 -*-
from collections import namedtuple

from ..net import NetEmbedding
from .locator import (
    create_linker_collection,
    create_node_collection,
    find_node_clusters,
    get_all_bound_solvent_molecules,
)
from .splitter import get_floating_solvent_molecules

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    [
        "nodes",
        "linkers",
        "bound_solvent",
        "unbound_solvent",
        "net_embedding",
        "all_linkers",
    ],
)


def run_fragmentation(mof) -> FragmentationResult:
    unbound_solvent = get_floating_solvent_molecules(mof)
    # Find nodes
    node_result = find_node_clusters(mof)
    node_collection = create_node_collection(mof, node_result)
    # Find bound solvent
    bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
    # Filter the linkers (valid linkers have at least two branch points)
    linker_collection, all_linkers, selected_linkers = create_linker_collection(
        mof, node_result, node_collection, unbound_solvent
    )
    # Now, get the net
    net_embedding = NetEmbedding(
        all_linkers, node_collection, selected_linkers, mof.lattice
    )
    fragmentation_results = FragmentationResult(
        node_collection,
        linker_collection,
        bound_solvent,
        unbound_solvent,
        net_embedding,
        all_linkers,
    )

    return fragmentation_results
