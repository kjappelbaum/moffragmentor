"""Test SBU collection functionality."""
from moffragmentor.sbu.sbucollection import SBUCollection


def test_sbu_collection(get_linker_object):
    sbu_collection = SBUCollection([get_linker_object])
    assert len(sbu_collection.sbus) == 1
    assert sbu_collection.sbus[0].composition == "C9 H3 O6"

    sbu_collection = SBUCollection([get_linker_object, get_linker_object])
    assert len(sbu_collection.sbus) == 2
