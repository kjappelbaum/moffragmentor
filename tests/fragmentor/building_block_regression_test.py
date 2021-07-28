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
