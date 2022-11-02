# -*- coding: utf-8 -*-
"""Test the branching point location"""
from moffragmentor.fragmentor.branching_points import (
    _is_branch_point,
    filter_branch_points,
    get_branch_points,
)


def test_get_branch_points(abaxin):
    """Test the branch point detection on a case that previously failed (#70)."""
    bp = get_branch_points(abaxin)
    assert set(bp) == {146, 147, 72, 73, 112, 113, 52, 53, 92, 93}


def test_is_branch_point(get_agn_mof):
    """In this case, all except for the Ag is a branch point candidate."""
    bp = [i for i, _ in enumerate(get_agn_mof.structure) if _is_branch_point(get_agn_mof, i)]
    assert len(bp) == 144

    filtered_bp = filter_branch_points(get_agn_mof, bp)
    assert len(filtered_bp) == 46
    for i in filtered_bp:
        assert get_agn_mof.structure[i].specie.symbol == "N"
