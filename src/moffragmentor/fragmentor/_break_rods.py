from loguru import logger

from moffragmentor.descriptors.sbu_dimensionality import get_sbu_dimensionality
from moffragmentor.fragmentor.nodelocator import NodelocationResult


def break_rod_node(mof, indices):
    metal_subset = {i for i in indices if i in mof.metal_indices}
    if not isinstance(metal_subset, int):
        return [set([i]) for i in metal_subset]
    else:
        return [set([metal_subset])]


def break_rod_nodes(mof, node_result):
    """Break rod nodes into smaller pieces."""
    new_nodes = []
    for node in node_result.nodes:
        if get_sbu_dimensionality(mof, node) == 1:
            logger.debug("Found 1- or 2-dimensional node. Will break into isolated metals.")
            new_nodes.extend(break_rod_node(mof, node))
        else:
            new_nodes.append(node)
    new_node_result = NodelocationResult(
        new_nodes,
        node_result.branching_indices,
        node_result.connecting_paths,
        node_result.binding_indices,
        node_result.to_terminal_from_branching,
    )
    return new_node_result
