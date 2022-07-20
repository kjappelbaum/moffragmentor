# -*- coding: utf-8 -*-
"""Make sure that we have no regression
in the output of the fragmentation"""
import os

import pytest

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_FILE_DIR = os.path.join(THIS_DIR, "..", "test_files")


def fragment(cif):
    path = os.path.join(TEST_FILE_DIR, cif)
    mof = MOF.from_cif(path)
    parts = mof.fragment()
    return parts


@pytest.mark.slow
def test_fragmentation():
    """On a set of structures from the review and beyond."""
    vejhez_parts = fragment("VEJHEZ_clean.cif")
    assert vejhez_parts.nodes.smiles[0] == "[C]1O[Zn]2O[C]O[Zn](O1)O[C]O2"
    assert len(set(vejhez_parts.nodes.smiles)) == 1

    assert (
        vejhez_parts.linkers.smiles[0]
        == "[O]C(=O)c1cc(cc(c1)C1=C2C=CC3=[N]2[Zn@]24n5c1ccc5C(=C1[N]2=C(C=C1)C(=c1n4c(=C3c2cc(cc(c2)C(=O)[O])C(=O)[O])cc1)c1cc(cc(c1)C(=O)[O])C(=O)[O])c1cc(cc(c1)C(=O)[O])C(=O)[O])C(=O)[O]"  # noqa: E501
    )
    assert len(set(vejhez_parts.linkers.smiles)) == 1
    assert vejhez_parts.net_embedding.rcsr_code == "the"

    hkust_parts = fragment("HKUST-1.cif")
    assert hkust_parts.linkers[0].smiles == "[O]C(=O)c1cc(C([O])=O)cc(C([O])=O)c1"

    assert hkust_parts.nodes[0].smiles == "[C]1O[Cu]234O[C]O[Cu]2(O1)(O[C]O3)O[C]O4"

    two_dimensional_mof_parts = fragment("2dmof.cif")
    assert len(two_dimensional_mof_parts.linkers.unique_sbus) == 1
    assert len(two_dimensional_mof_parts.nodes.unique_sbus) == 1
