# -*- coding: utf-8 -*-
import os

import pytest

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="module")
def get_cuiiibtc_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    return mof


@pytest.fixture(scope="module")
def get_hkust_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    return mof
