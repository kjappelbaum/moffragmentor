# -*- coding: utf-8 -*-
"""Test ranking of molecules according to similarity"""
from moffragmentor.utils.mol_compare import mcs_rank, tanimoto_rank

reference_smiles = "C1=CC=CC=C1"

query_smiles = ["C1=CC=CC=N1", "C1=CC=CC=P1", "CC1=CC=CC=C1"]


def test_tanimoto_rank():
    ranked_smiles = tanimoto_rank(reference_smiles, query_smiles)
    assert ranked_smiles[0][0] == "C1=CC=CC=N1"
    assert ranked_smiles[1][0] == "C1=CC=CC=P1"
    assert ranked_smiles[2][0] == "CC1=CC=CC=C1"


def test_mcs_rank():
    ranked_smiles = mcs_rank(reference_smiles, query_smiles)
    assert ranked_smiles[1][0] == "C1=CC=CC=N1"
    assert ranked_smiles[2][0] == "C1=CC=CC=P1"
    assert ranked_smiles[0][0] == "CC1=CC=CC=C1"
