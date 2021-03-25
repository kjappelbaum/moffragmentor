# -*- coding: utf-8 -*-
"""This module contains classes that are used to organize non-SBU molecules such as floating or bound solvents"""
from collections import Counter
from typing import List

from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph


class NonSbuMolecule:
    """Class to handle solvent or other non-SBU molecules"""

    def __init__(
        self, molecule: Molecule, molecule_graph: MoleculeGraph, indices: List[int]
    ):
        self.molecule = molecule
        self.molecule_graph = molecule_graph
        self.indices = indices

    @property
    def composition(self):
        return self.molecule.composition.alphabetical_formula

    def __str__(self):
        return str(self.composition)

    def __len__(self):
        return len(self.molecule)


class NonSbuMoleculeCollection:
    """Class to handle collections of molecules, e.g. bound solvents and non-bound solvents"""

    def __init__(self, non_sbu_molecules=List[NonSbuMolecule]):
        self.molecules = non_sbu_molecules
        self._composition = None

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        return self.molecules[index]

    def __next__(self):
        for molecule in self.molecules:
            yield molecule

    def _get_composition(self):
        if self._composition is None:
            composition = []
            for mol in self.molecules:
                composition.append(str(mol.composition))
            composition_counter = Counter(composition)
            self._composition = dict(composition_counter)

        return self._composition

    @property
    def composition(self):
        return self._get_composition()
