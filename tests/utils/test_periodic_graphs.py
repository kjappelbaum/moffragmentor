# -*- coding: utf-8 -*-
from pymatgen.analysis.graphs import StructureGraph

from moffragmentor.utils.periodic_graph import _get_supergraph_index_map


def test__get_supergraph_index_map():
    """Make sure that the index map we create makes sense, i.e., for the 3,3,3 map we have to mulitply with the factor 27"""
    map = _get_supergraph_index_map(624)
    assert isinstance(map, dict)
    assert len(map.values()) == 624
    assert isinstance(map[0], list)
    assert len(map[0]) == 27
    assert len(sum(map.values(), [])) == 624 * 27
