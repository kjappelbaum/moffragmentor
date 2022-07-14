# -*- coding: utf-8 -*-
"""Methods for the fragmentation of MOFs"""
from collections import namedtuple

from loguru import logger

from .filter import point_in_mol_coords
from .linkerlocator import create_linker_collection
from .nodelocator import NodelocationResult, create_node_collection, find_node_clusters
from .solventlocator import get_all_bound_solvent_molecules, get_floating_solvent_molecules
from ..net import build_net
from ..utils import _get_metal_sublist
from ..utils.periodic_graph import is_periodic

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    ["nodes", "linkers", "bound_solvent", "unbound_solvent", "net_embedding"],
)


def run_fragmentation(mof) -> FragmentationResult:  # pylint: disable=too-many-locals
    """Take a MOF and split it into building blocks."""
    unbound_solvent = get_floating_solvent_molecules(mof)
    need_rerun = True
    forbidden_indices = []
    while need_rerun:
        not_node = []
        
        # Find nodes
        node_result = find_node_clusters(mof, unbound_solvent.indices, forbidden_indices=forbidden_indices)
        node_collection = create_node_collection(mof, node_result)
        # Find bound solvent
        bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
        # Filter the linkers (valid linkers have at least two branch points)
        linker_collection = create_linker_collection(mof, node_result, node_collection, unbound_solvent)

        logger.info("Checking for metal in linker")
        # ToDo: factor this out into its own function
        for i, node in enumerate(node_result.nodes):  # pylint:disable=too-many-nested-blocks
            metal_in_node = _get_metal_sublist(node, mof.metal_indices)
            # ToDo: check and think if this can handle the general case
            # it should, at least if we only look at the metals
            if len(metal_in_node) == 1:
                for _, linker in enumerate(linker_collection):
                    if point_in_mol_coords(
                        mof.cart_coords[metal_in_node[0]],
                        mof.cart_coords[linker._original_indices],  # pylint:disable=protected-access
                        mof.lattice,
                    ):
                        logger.info("Metal in linker found, current node: {}, indices: {}".format(node, node_collection[i]._original_indices))
                        if (
                            len(
                                set(mof.get_neighbor_indices(metal_in_node[0]))
                                & set(linker._original_indices)  # pylint:disable=protected-access
                            )
                            > 1
                        ):
                            if is_periodic(
                                mof,
                                metal_in_node
                                + linker._original_indices,  # pylint:disable=protected-access
                            ):
                                need_rerun = True
                                not_node.append(i)
                                break
       
        for node in not_node:
            forbidden_indices.extend(node_collection[node]._original_indices) 

        if len(not_node) == 0:
            need_rerun = False
            break
     
    logger.info("Constructing the embedding")
    # Now, get the net
    net_embedding = build_net(linker_collection, node_collection, mof.lattice)
    fragmentation_results = FragmentationResult(
        node_collection,
        linker_collection,
        bound_solvent,
        unbound_solvent,
        net_embedding,
    )

    return fragmentation_results
