# -*- coding: utf-8 -*-
import os

import pytest
from pymatgen import Structure

from moffragmentor import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def check_mof_creation_from_cif(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert isinstance(mof, object)
    assert isinstance(mof.structure, Structure)


def test_is_terminal(get_cuiiibtc_mof):
    """An atom is terminal when it only has one neighbor"""
    mof = get_cuiiibtc_mof
    assert mof._is_terminal(32)
    assert not mof._is_terminal(129)
    assert not mof._is_terminal(176)
