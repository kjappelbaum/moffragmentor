# -*- coding: utf-8 -*-
from collections import namedtuple

from .locator import (
    create_linker_collection,
    create_node_collection,
    find_node_clusters,
)
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

    # Filter the linkers (valid linkers have at least two branch points)
    linker_collection = create_linker_collection(mof, node_result, unbound_solvent)
    # Now, get the net

    fragmentation_results = FragmentationResult(
        node_collection, linker_collection, unbound_solvent
    )
