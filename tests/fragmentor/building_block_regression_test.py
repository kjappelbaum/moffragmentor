# -*- coding: utf-8 -*-
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
    """On a set of structures from the review and beyond.
    Not that we only give the parent net, not potential subnets.
    """
    vejhez_parts = fragment("VEJHEZ_clean.cif")
    assert (
        vejhez_parts.linkers.smiles[0]
        == "[O]C(=O)c1cc(cc(c1)C1=C2C=CC3=[N]2[Zn@]24n5c1ccc5C(=C1[N]2=C(C=C1)C(=c1n4c(=C3c2cc(cc(c2)C(=O)[O])C(=O)[O])cc1)c1cc(cc(c1)C(=O)[O])C(=O)[O])c1cc(cc(c1)C(=O)"
    )
    assert len(set(vejhez_parts.linkers.smiles)) == 1

    assert vejhez_parts.nodes.smiles[0] == "[C]1O[Zn]2O[C]O[Zn](O1)O[C]O2"
    assert len(set(vejhez_parts.nodes.smiles)) == 1

    hkust_parts = fragment("HKUST-1.cif")
    assert hkust_parts.linkers[0].smiles == "[O]C(=O)c1cc(cc(c1)C(=O)[O])C(=O)[O]"

    assert hkust_parts.nodes[0].smiles == "[C]1O[Cu]234[Cu](O1)(O[C]O4)(O[C]O3)O[C]O2"
