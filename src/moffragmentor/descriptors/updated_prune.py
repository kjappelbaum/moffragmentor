# -*- coding: utf-8 -*-
# pylint:disable=no-member
"""Updated prune method from
https://github.com/skearnes/rdkit-utils/pull/21/commits/be6a3512919ae4bd9387dc25907cd622bab27cda"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def prune_conformers_update(self, mol):
    """
    Update the 'prune_conformers' function to get more speed.
    The idea came from Jean-Paul Ebejer and others' paper:
    *Freely Availabel Conformer Generation Methods: How Good Are They?*
    (1) generate n conformers in set C_gen.
    (2) Energy minimization is performed, and the conformer list is sorted
        by increasing energy value, recorded the lower energy conformer,
        and add it to C_keep.
    (3) For each conformer, c in C_gen, compute the rmsd between c and
        each conformer in C_keep.
        (a) If any rmsd value is smaller than a fixed threshold, dicard c
        (b) Otherwise add c to C_keep.
    I test: the result is the same as prune_conformers.
    """
    if self.rmsd_threshold < 0 or mol.GetNumConformers() <= 1:
        return mol

    energies = self.get_conformer_energies(mol)

    sort = np.argsort(energies)  # sort by increasing energy
    keep = []  # always keep lowest-energy conformer
    for i in sort:
        if len(keep) == 0:
            keep.append(i)
            continue

        if len(keep) >= self.max_conformers:
            break

        filter_rms = filter(
            lambda rms: rms >= self.rmsd_threshold,
            [AllChem.GetBestRMS(mol, mol, i, j) for j in keep],
        )
        if filter_rms:
            keep.append(i)
    new = Chem.Mol(mol)
    new.RemoveAllConformers()
    conf_ids = [conf.GetId() for conf in mol.GetConformers()]
    for i in keep:
        conf = mol.GetConformer(conf_ids[i])
        new.AddConformer(conf, assignId=True)
    return new
