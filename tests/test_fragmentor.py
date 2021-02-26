# -*- coding: utf-8 -*-
import os

from moffragmentor.fragmentor import (
    classify_neighbors,
    find_solvent_molecule_indices,
    fragment_all_node,
    fragment_oxo_node,
    has_path_to_any_other_metal,
    is_valid_node,
    has_two_metals_as_neighbor
)
from moffragmentor.mof import MOF

THIS_DIR = os.path.dirname(os.path.realpath(__file__))


def test_has_path_to_any_other_metal():
    """This tests if we can correctly find sites that are solvents
    in the sense that they only link to one metal and do not have a path to another metal
    """
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    assert not has_path_to_any_other_metal(mof, 176, 8)
    assert has_path_to_any_other_metal(mof, 129, 8)


def test_find_solvent_molecule_indices():
    """This tests if given the connecting atom of a solvent, we find all the other ones of that molecules"""
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    solvent_mol = find_solvent_molecule_indices(mof, 176, 8)
    assert len(solvent_mol) == 3
    assert 176 in solvent_mol

    water = mof.get_neighbor_indices(176) + [176]
    water.remove(8)

    assert sorted(water) == sorted(solvent_mol)


def test_classify_neighbors():
    """This tests if we can find all the solvents connected to our metals.
    In this example, we have 8 waters"""
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    solvent_connections = classify_neighbors(mof, mof.metal_indices)
    assert len(solvent_connections) == 3
    assert len(solvent_connections["solvent_indices"]) == 8
    assert len(solvent_connections["solvent_connections"]) == 8
    assert solvent_connections["solvent_connections"] == set(
        [176, 177, 178, 179, 180, 181, 182, 183]
    )


def test_fragment_all_node():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))

    fragmentation_all_nodes = fragment_all_node(mof)

    assert len(fragmentation_all_nodes) == 4
    assert len(fragmentation_all_nodes["solvent_connections"]) == 8
    assert len(fragmentation_all_nodes["solvent_indices"]) == 8

    assert fragmentation_all_nodes["node_indices"] == set(
        [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            58,
            61,
            64,
            67,
            70,
            73,
            76,
            79,
            82,
            85,
            88,
            91,
            94,
            97,
            100,
            103,
            104,
            107,
            110,
            113,
            116,
            119,
            122,
            125,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
            150,
            151,
            152,
            153,
            154,
            155,
            156,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            164,
            165,
            166,
            167,
            168,
            169,
            170,
            171,
            172,
            173,
            174,
            175,
        ]
    )

    assert len(fragmentation_all_nodes["connecting_node_indices"]) < len(fragmentation_all_nodes["node_indices"])

    for index in fragmentation_all_nodes["connecting_node_indices"]:
        assert str(mof.structure[index].specie) == 'C'
        assert index in fragmentation_all_nodes["node_indices"]

def test_fragment_oxo_node():
    """Test the clustering using the oxo convention"""
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    oxo_node_index_result = fragment_oxo_node(mof)
    assert len(oxo_node_index_result) == 4
    assert len(oxo_node_index_result["solvent_connections"]) == 8
    assert len(oxo_node_index_result["solvent_indices"]) == 8

    assert len(oxo_node_index_result["node_indices"]) == len(mof.metal_indices)
    assert len(oxo_node_index_result["connecting_node_indices"]) == len(mof.metal_indices)
    # check turning off the solvent filtering
    oxo_node_index_result = fragment_oxo_node(mof, filter_out_solvent=False)
    assert len(oxo_node_index_result) == 4
    assert len(oxo_node_index_result["solvent_connections"]) == 8
    assert len(oxo_node_index_result["solvent_indices"]) == 8

    assert len(oxo_node_index_result["node_indices"]) == len(mof.metal_indices) + 8 * 3
    assert len(oxo_node_index_result["connecting_node_indices"]) == len(mof.metal_indices)


def test_is_valid_node(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert is_valid_node(mof, mof.metal_indices)
    assert not is_valid_node(mof, [2])


def test_has_two_metals_as_neighbor(get_cuiiibtc_mof):
    mof = get_cuiiibtc_mof
    assert not has_two_metals_as_neighbor(mof, 0)
    assert has_two_metals_as_neighbor(mof, 2)