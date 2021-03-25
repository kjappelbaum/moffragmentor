# -*- coding: utf-8 -*-
from collections import Counter

from moffragmentor.fragmentor.locator import find_node_clusters


def test_find_cu_i_ii_btc_clusters(get_cuiiibtc_mof):
    """Make sure that we do find the two correct node types in Cu(I/II)-BTC"""
    mof = get_cuiiibtc_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 6  # four paddlewheels and two macrocylus
    nodes_sizes = [len(node) for node in node_location_result.nodes]
    assert len(set(nodes_sizes)) == 2
    size_counters = Counter(nodes_sizes)
    assert sorted(list(size_counters.values())) == [2, 4]
    assert sorted(list(size_counters.keys())) == [16, 20]


def test_find_porphyrin_mof_clusters(get_porphryin_mof):
    """Looking for the porphyrin MOF clusters, here we should only find two nodes in the unit cell"""
    mof = get_porphryin_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 2
    assert len(node_location_result.branching_indices) == 8 * 2


def test_find_hypothetical_mof_clusters(get_hypothetical_mof):
    """Check if we can find the correct nodes in hypothetical MOFs"""
    mof = get_hypothetical_mof
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 16
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert node_lengths[0] == 24


def test_find_p_linker_floating_mof_clusters(get_p_linker_with_floating):
    """Check if we can deal with non-carboxy ligand and floating solvent"""
    mof = get_p_linker_with_floating
    node_location_result = find_node_clusters(mof)
    assert len(node_location_result) == 3
    assert len(node_location_result.nodes) == 8
    node_lengths = [len(node) for node in node_location_result.nodes]
    assert len(set(node_lengths)) == 1
    assert (
        node_lengths[0] == 25
    )  # ToDo: think more carefully if we want to deal this way with the µ1-carboxy
