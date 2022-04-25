# -*- coding: utf-8 -*-
"""Code taken from the SI for 10.1021/acs.jcim.6b00565"""
# pylint:disable=no-member
from collections import OrderedDict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def _generate_conformers(mol, num_confs):  # pylint:disable=too-many-locals
    # Add H atoms to skeleton
    molecule = Chem.AddHs(mol)
    conformer_integers = []
    # Embed and optimise the conformers
    conformers = AllChem.EmbedMultipleConfs(molecule, num_confs, pruneRmsThresh=0.5, numThreads=3)

    optimised_and_energies = AllChem.MMFFOptimizeMoleculeConfs(
        molecule, maxIters=600, numThreads=3, nonBondedThresh=100.0
    )
    energy_dict_with_key_was_id = {}
    final_conformers_to_use = {}
    # Only keep the conformers which were successfully fully optimised

    for conformer in conformers:
        optimised, energy = optimised_and_energies[conformer]
        if optimised == 0:
            energy_dict_with_key_was_id[conformer] = energy
            conformer_integers.append(conformer)
    # Keep the lowest energy conformer
    lowestenergy = min(energy_dict_with_key_was_id.values())
    for key, value in energy_dict_with_key_was_id.items():
        if value == lowestenergy:
            lowest_energy_conformer_id = key
    final_conformers_to_use[lowest_energy_conformer_id] = lowestenergy

    # Remove H atoms to speed up substructure matching
    molecule = AllChem.RemoveHs(molecule)
    # Find all substructure matches of the molecule with itself,
    # to account for symmetry
    matches = molecule.GetSubstructMatches(molecule, uniquify=False)
    maps = [list(enumerate(match)) for match in matches]
    # Loop over conformers other than the lowest energy one
    for conformer_id, _ in energy_dict_with_key_was_id.items():
        okay_to_add = True
        for finalconformer_id in final_conformers_to_use:
            rms = AllChem.GetBestRMS(molecule, molecule, finalconformer_id, conformer_id, maps)
            if rms < 1.0:
                okay_to_add = False
                break

        if okay_to_add:
            final_conformers_to_use[conformer_id] = energy_dict_with_key_was_id[conformer_id]

    sorted_dictionary = OrderedDict(sorted(final_conformers_to_use.items(), key=lambda t: t[1]))
    energies = list(sorted_dictionary.values())

    return energies


def calc_nconf20(energy_list):
    energy_descriptor = 0

    relative_energies = np.array(energy_list) - energy_list[0]

    for energy in relative_energies[1:]:
        if 0 <= energy < 20:
            energy_descriptor += 1

    return energy_descriptor


def n_conf20(mol):
    energy_list = _generate_conformers(mol, 50)
    descriptor = calc_nconf20(energy_list)
    return descriptor
