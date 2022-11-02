"""Test SBU functionality."""
import pytest


def test_sbu(get_linker_object, get_node_object):
    linker = get_linker_object
    assert linker.composition == "C9 H3 O6"
    assert pytest.approx(linker.molar_mass) == 207.11651999999998

    node = get_node_object
    assert node.composition == "C4 Cu2 O8"
    assert pytest.approx(node.molar_mass) == 303.13
