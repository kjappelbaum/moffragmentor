# -*- coding: utf-8 -*-
from moffragmentor import MOF
from moffragmentor.fragmentor.filter import (  # _creates_new_connected_components,; _filter_isolated_node_candidates,
    _filter_branch_points,
)


def test__filter_branching_points(get_dicarboxy_biphenyl_graph):
    """Branching points are only those for which the path to a node does not cross another branching point"""
    g, metal, branching = get_dicarboxy_biphenyl_graph
    filtered_indices = _filter_branch_points(branching, metal, g)
    assert len(filtered_indices) == 2
    assert "n2" in filtered_indices
    assert "n15" in filtered_indices


# def test__creates_new_connected_components(porphyrin_mof_structure_and_graph):
#     """Removing the Zn in the porphyrin MOF should not create new connected components."""
#     s, sg = porphyrin_mof_structure_and_graph
#     mof = MOF(s, sg)
#     assert not _creates_new_connected_components([0], sg, mof.terminal_indices)
#     assert not _creates_new_connected_components([1], sg, mof.terminal_indices)
#     assert not _creates_new_connected_components([60], sg, mof.terminal_indices)
#     assert not _creates_new_connected_components([61], sg, mof.terminal_indices)
#     assert _creates_new_connected_components([60, 61], sg, mof.terminal_indices)


# def test__filter_isolated_node_candidates(porphyrin_mof_structure_and_graph):
#     """for the porphyrin MOF, we should discard the Zn in the prophyrins"""
#     s, sg = porphyrin_mof_structure_and_graph
#     mof = MOF(s, sg)
#     site_candidates = [[0], [1], [60], [61]]  # Zn, Zn, Ru, Ru
#     filtered_nodes, original_indices = _filter_isolated_node_candidates(
#         site_candidates, sg, mof.terminal_indices
#     )
#     assert filtered_nodes == [[60], [61]]
#     assert original_indices == [2, 3]
