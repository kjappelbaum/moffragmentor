# -*- coding: utf-8 -*-
import os

import pytest

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_FILE_DIR = os.path.join(THIS_DIR, "..", "test_files")


def get_rscr_code(cif):
    path = os.path.join(TEST_FILE_DIR, cif)
    mof = MOF.from_cif(path)
    parts = mof.fragment()
    return parts.net_embedding.rcsr_code


@pytest.mark.slow
def test_net_detection():
    """On a set of structures from the review and beyond.
    Not that we only give the parent net, not potential subnets.
    """
    assert get_rscr_code("mof-5_cellopt.cif") == "pcu"
    assert get_rscr_code("HKUST-1.cif") == "tbo"
    # assert get_rscr_code("SDU-1.cif") == "tfe"
    # assert get_rscr_code("USF-3.cif") == "mmm"
    assert get_rscr_code("QEFWUV_clean.cif") == "csq"
    # only works after "standardization of crystal data" in VESTA
    assert get_rscr_code("LASYOU.cif") == "nbo"
    assert get_rscr_code("OFOCUI_clean.cif") == "nbo"
    # assert get_rscr_code("PUBSOV_clean.cif") == "ssb"
    assert get_rscr_code("WETPES_clean.cif") == "ssa"