# -*- coding: utf-8 -*-
"""Methods for describing the flexibility of molecules"""
import numpy as np
from rdkit.Chem import GraphDescriptors

from .nconf20 import n_conf20


def calculate_flexibility_descriptors(mol):
    nconf20 = n_conf20(mol)
    try:
        kier = kier_molecular_flexibility(mol)
    except Exception:
        nconf20 = np.nan
    return {
        "nconf20": nconf20,
        "kier": kier,
    }


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
