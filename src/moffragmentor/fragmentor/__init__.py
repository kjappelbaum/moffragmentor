# -*- coding: utf-8 -*-
"""Methods for the fragmentation of MOFs"""
from collections import namedtuple

from loguru import logger
from skspatial.objects import Points

from moffragmentor.sbu.linkercollection import LinkerCollection

from ._no_core_linker import generate_new_node_collection
from .linkerlocator import create_linker_collection
from .nodelocator import create_node_collection, find_node_clusters
from .solventlocator import get_all_bound_solvent_molecules, get_floating_solvent_molecules
from ..net import build_net
from ..utils import _get_metal_sublist

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    ["nodes", "linkers", "bound_solvent", "unbound_solvent", "capping_molecules", "net_embedding"],
)


def metal_and_branching_coplanar(node, mof, tol=0.1):
    branching_idx = list(node._original_graph_branching_indices)
    coords = mof.frac_coords[list(node._original_indices) + branching_idx]
    points = Points(coords)
    return points.are_coplanar(tol=tol)


def run_fragmentation(mof) -> FragmentationResult:  # pylint: disable=too-many-locals
    """Take a MOF and split it into building blocks."""
    logger.debug("Starting fragmentation with location of unbound solvent")
    unbound_solvent = get_floating_solvent_molecules(mof)
    need_rerun = True
    forbidden_indices = []
    counter = 0
    while need_rerun:
        not_node = []
        logger.debug(f"Fragmenting MOF for the {counter} time")
        # Find nodes
        node_result = find_node_clusters(
            mof, unbound_solvent.indices, forbidden_indices=forbidden_indices
        )
        node_collection = create_node_collection(mof, node_result)
        # Find bound solvent
        logger.debug("Locating bound solvent")
        bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
        logger.debug(f"Found bound solvent {len(bound_solvent.indices)>0}")
        # Filter the linkers (valid linkers have at least two branch points)
        logger.debug("Locating linkers")
        linker_collection = create_linker_collection(
            mof, node_result, node_collection, unbound_solvent, bound_solvent
        )

        logger.debug("Checking for metal in linker")
        # ToDo: factor this out into its own function
        for i, node in enumerate(node_result.nodes):  # pylint:disable=too-many-nested-blocks
            metal_in_node = _get_metal_sublist(node, mof.metal_indices)
            # ToDo: check and think if this can handle the general case
            # it should, at least if we only look at the metals
            if len(metal_in_node) == 1:
                logger.debug(
                    "metal_in_node",
                    i,
                    metal_in_node,
                    node_collection[i]._original_indices,
                    node_collection[i]._original_graph_branching_indices,
                )
                if metal_and_branching_coplanar(node_collection[i], mof):
                    logger.debug(
                        "Metal in linker found, current node: {}, indices: {}".format(
                            node,
                            node_collection[i]._original_indices,
                        )
                    )
                    need_rerun = True
                    not_node.append(i)

        for node in not_node:
            forbidden_indices.extend(list(node_collection[node]._original_indices))

        if len(not_node) == 0:
            need_rerun = False
            break
        if len(not_node) == len(node_result.nodes):
            logger.warning(
                "We have metal in plane with the organic part. \
                Which would indicate a prophyrin. \
                However, there is no other metal cluster, so we will treat it as metal cluster."
            )
            need_rerun = False
            break
        counter += 1

    logger.debug(
        "Check if we need to move the capping molecules from the linkercollection into their own collection"
    )
    is_linker = []
    is_capping = []
    for linker in linker_collection:
        if len(linker._original_graph_branching_indices) >= 2:
            is_linker.append(linker)
        else:
            if len(linker.molecule) > 1:
                is_capping.append(linker)
            else:
                logger.warning(
                    "Capping molecule with only one atom. Perhaps something funny happened in the fragmentation."
                )

    linker_collection = LinkerCollection(is_linker)
    capping_molecules = LinkerCollection(is_capping)

    # Now handle the case of the the frameworks that have linkers without core (e.g. H-COO)
    # we detect those by having 0 entries in the linker collection and all capping molecules overlapping with
    # with the node collection
    # in this case, we will pass the capping molecules to the net constructor
    use_capping_in_net = False
    if len(linker_collection) == 0:
        logger.warning("No linkers found")
        if all([set(c._original_indices) & node_collection.indices for c in capping_molecules]):
            logger.warning(
                "All capping molecules overlap with node collection. Will use them for net construction."
            )
            logger.debug("Constructing the embedding")
            use_capping_in_net = True

    if use_capping_in_net:
        logger.debug("Constructing the embedding")
        # However, I'd also need to split the node in this case
        new_node_collection = generate_new_node_collection(mof, node_result)
        net_embedding = build_net(new_node_collection, capping_molecules, mof.lattice)
    else:
        logger.debug("Constructing the embedding")
        # Now, get the net
        net_embedding = build_net(node_collection, linker_collection, mof.lattice)
    fragmentation_results = FragmentationResult(
        node_collection,
        linker_collection,
        bound_solvent,
        unbound_solvent,
        capping_molecules,
        net_embedding,
    )

    return fragmentation_results
