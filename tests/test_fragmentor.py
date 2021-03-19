# -*- coding: utf-8 -*-
from moffragmentor.fragmentor.filter import (
    _creates_new_connected_components,
    _filter_branch_points,
    _filter_isolated_node_candidates,
)


def test__filter_branching_points(get_dicarboxy_biphenyl_graph):

    g, metal, branching = get_dicarboxy_biphenyl_graph
    filtered_indices = _filter_branch_points(branching, metal, g)
    assert len(filtered_indices) == 2
    assert "n2" in filtered_indices
    assert "n15" in filtered_indices


def test__creates_new_connected_components(porphyrin_mof_structure_and_graph):
    _, sg = porphyrin_mof_structure_and_graph
    assert not _creates_new_connected_components(0, sg.graph)
    assert not _creates_new_connected_components(1, sg.graph)
    assert _creates_new_connected_components(60, sg.graph)
    assert _creates_new_connected_components(61, sg.graph)


def test__filter_isolated_node_candidates(porphyrin_mof_structure_and_graph):
    # for the porphyrin MOF, we should discard the Zn in the prophyrins
    s, sg = porphyrin_mof_structure_and_graph
    site_candidates = [0, 1, 60, 61]  # Zn, Zn, Ru, Ru
    filtered_nodes = _filter_isolated_node_candidates(site_candidates, sg.graph)
    assert filtered_nodes == [60, 61]
