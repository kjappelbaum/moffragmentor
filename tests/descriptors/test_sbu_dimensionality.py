# -*- coding: utf-8 -*-
"""Test calculation of the SBU dimensionality"""
from moffragmentor.descriptors.sbu_dimensionality import get_structure_graph_dimensionality


def test_get_structure_graph_dimensionality(get_0d_node_graph, get_1d_node_graph):
    """The function we test here does not do anything special
    other than calling pymatgen.
    This test is rather for documentation
    and to check if the function works as a MOF chemist
    would expect"""
    assert get_structure_graph_dimensionality(get_0d_node_graph) == 0
    assert get_structure_graph_dimensionality(get_1d_node_graph) == 1
