# -*- coding: utf-8 -*-
"""Methods for describing the flexibility of molecules"""
from rdkit.Chem import (Descriptors, Descriptors3D, GraphDescriptors,
                        rdMolDescriptors)

from .nconf20 import calc_nconf20


def calculate_flexibility_descriptors(mol):
    nconf20 = calc_nconf20(mol)


def kier_molecular_flexibility(mol):
    """
    Calculation of Kier molecular flexibility index
    10.1002/qsar.19890080307
    """
    kappa1 = GraphDescriptors.Kappa1(mol)
    kappa2 = GraphDescriptors.Kappa2(mol)
    A = mol.GetNumHeavyAtoms()
    phi = kappa1 * kappa2 / (A + 0.0)
    return phi
