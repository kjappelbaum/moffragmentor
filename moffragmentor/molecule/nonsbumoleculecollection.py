# -*- coding: utf-8 -*-
"""collections of molecules, e.g. bound solvents and non-bound solvents"""
from collections import Counter
from typing import List

from .nonsbumolecule import NonSbuMolecule


class NonSbuMoleculeCollection:
    """Class to handle collections of molecules,
    e.g. bound solvents and non-bound solvents"""

    def __init__(self, non_sbu_molecules=List[NonSbuMolecule]):
        self.molecules = non_sbu_molecules
        self._composition = None
        # currently also contains indices from the supercell expansion
        self.indices = sum([molecule.indices for molecule in self.molecules], [])

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, index):
        return self.molecules[index]

    def __next__(self):
        for molecule in self.molecules:
            yield molecule

    def __add__(self, other):
        molecules = self.molecules + other.molecules
        return NonSbuMoleculeCollection(molecules)

    def _get_composition(self):
        if self._composition is None:
            composition = []
            for mol in self.molecules:
                composition.append(str(mol.composition))
            composition_counter = Counter(composition)
            self._composition = dict(composition_counter)

        return self._composition

    @property
    def composition(self) -> str:
        return self._get_composition()
