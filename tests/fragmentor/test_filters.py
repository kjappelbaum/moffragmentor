# -*- coding: utf-8 -*-

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
