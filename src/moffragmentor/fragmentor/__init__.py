# -*- coding: utf-8 -*-
"""Methods for the fragmentation of MOFs"""
from collections import namedtuple

from loguru import logger

from moffragmentor.fragmentor._no_core_linker import generate_new_node_collection
from moffragmentor.fragmentor.linkerlocator import create_linker_collection
from moffragmentor.fragmentor.nodelocator import detect_porphyrin, find_nodes
from moffragmentor.fragmentor.solventlocator import (
    get_all_bound_solvent_molecules,
    get_floating_solvent_molecules,
)
from moffragmentor.net import build_net
from moffragmentor.sbu.linkercollection import LinkerCollection

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    [
        "nodes",
        "linkers",
        "bound_solvent",
        "unbound_solvent",
        "capping_molecules",
        "net_embedding",
        "has_1d_sbu",
    ],
)

_MAX_LOOPS = 2


def run_fragmentation(
    mof,
    check_dimensionality: bool = True,
    create_single_metal_bus: bool = False,
    break_organic_nodes_at_metal: bool = True,
) -> FragmentationResult:
    """Take a MOF and split it into building blocks."""
    logger.debug("Starting fragmentation with location of unbound solvent")
    unbound_solvent = get_floating_solvent_molecules(mof)
    need_rerun = True
    forbidden_indices = []
    counter = 0
    has_1d_sbu = None
    while need_rerun:
        try:
            not_node = []
            logger.debug(f"Fragmenting MOF for the {counter} time")
            node_result, node_collection = find_nodes(
                mof,
                unbound_solvent,
                forbidden_indices,
                create_single_metal_bus,
                check_dimensionality,
                break_organic_nodes_at_metal,
            )

            # Find bound solvent
            logger.debug("Locating bound solvent")
            bound_solvent = get_all_bound_solvent_molecules(mof, node_result.nodes)
            logger.debug(f"Found bound solvent {len(bound_solvent.indices)>0}")

            logger.debug("Locating linkers")
            linker_collection = create_linker_collection(
                mof, node_result, node_collection, unbound_solvent, bound_solvent
            )
            logger.debug(f"Found {len(linker_collection)} linkers")
            # if we have no linker we need to rerun the node detection
            if len(linker_collection) == 0:
                logger.debug("No linkers found, rerunning node detection")
                create_single_metal_bus = True
                need_rerun = True
            else:
                logger.debug("Checking for metal in linker")
                # ToDo: factor this out into its own function

                not_node = detect_porphyrin(node_collection, mof)

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

        except Exception as e:
            logger.exception(f"Error while fragmenting: {e}")
        finally:
            counter += 1
            if counter > _MAX_LOOPS:
                need_rerun = False
                raise ValueError(f"Could not fragment after {counter} attempts.")

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
    if len(linker_collection) == 0 and len(is_capping) > 0:
        logger.warning(
            "No linkers found, but capping molecules. Assigning capping molecules to linkers."
        )
        linker_collection = LinkerCollection(is_capping)
        is_capping = []

    capping_molecules = LinkerCollection(is_capping)

    # Now handle the case of the the frameworks that have linkers without core (e.g. H-COO)
    # we detect those by having 0 entries in the linker collection and all capping molecules overlapping with
    # with the node collection
    # in this case, we will pass the capping molecules to the net constructor
    use_capping_in_net = False
    if len(linker_collection) == 0:
        logger.warning("No linkers found")
        if all(set(c._original_indices) & node_collection.indices for c in capping_molecules):
            logger.warning(
                "All capping molecules overlap with node collection. Will use them for net construction."
            )
            logger.debug("Constructing the embedding")
            use_capping_in_net = True

    try:
        if use_capping_in_net:
            logger.debug("Constructing the embedding")
            # However, I'd also need to split the node in this case
            new_node_collection = generate_new_node_collection(mof, node_result)

            # ToDo: need to add some code as follows
            # Need to create also "netnodes" before that
            # egde_candiates = defaultdict(list)
            # for i, netnode_i in enumerate(netnodes):
            #     for j, netnode_j in enumerate(netnodes):
            #         at_least_one_edge, images = has_edge(netnode_i, netnode_j, lattice)
            #         if at_least_one_edge:
            #             for coord, image_a, image_b in images:
            #                 if i == j and all(image_a == (0,0,0)) and all(image_b == (0,0,0)):
            #                     continue
            #                 metal_center = lattice.get_fractional_coords(netnode_i._coords) + image_a
            #                 linker_center = lattice.get_fractional_coords(netnode_j._coords) + image_b
            #                 edge = VoltageEdge(
            #                     linker_center - metal_center,
            #                     i,
            #                     j,
            #                     image_a,
            #                     image_b,
            #                 )
            #                 egde_candiates[
            #                     (round(coord[0], 2), round(coord[1], 2), round(coord[2], 2))
            #                 ].append((edge, np.abs(image_b).sum()))

            net_embedding = build_net(new_node_collection, capping_molecules, mof.lattice)
        else:
            logger.debug("Constructing the embedding")
            # Now, get the net
            net_embedding = build_net(node_collection, linker_collection, mof.lattice)
    except Exception as e:
        logger.exception(f"Error {e} in net construction")
        net_embedding = None
    fragmentation_results = FragmentationResult(
        node_collection,
        linker_collection,
        bound_solvent,
        unbound_solvent,
        capping_molecules,
        net_embedding,
        has_1d_sbu,
    )

    return fragmentation_results
