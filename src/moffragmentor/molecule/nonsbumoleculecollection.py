# -*- coding: utf-8 -*-
"""Collections of molecules, e.g. bound solvents and non-bound solvents."""
from collections import Counter
from typing import List

from .nonsbumolecule import NonSbuMolecule


class NonSbuMoleculeCollection:
    """Class to handle collections of molecules.

    For example, bound solvents and non-bound solvents.
    """

    def __init__(self, non_sbu_molecules: List[NonSbuMolecule]):
        """Construct a NonSbuMoleculeCollection.

        Args:
            non_sbu_molecules (List[NonSbuMolecule]): List of NonSbuMolecule objects.
        """
        self.molecules = non_sbu_molecules
        self._composition = None
        # currently also contains indices from the supercell expansion
        self.indices = sum([molecule.indices for molecule in self.molecules], [])

    def __len__(self):
        """Return number of molecules in the collection."""
        return len(self.molecules)

    def __repr__(self):
        """Return a string representation of the collection."""
        return f"NonSbuMoleculeCollection({self._get_composition()})"

    def __getitem__(self, index):
        """Get a molecule from the collection."""
        return self.molecules[index]

    def __next__(self):
        """Iterate over the molecules in the collection."""
        for molecule in self.molecules:
            yield molecule

    def __add__(self, other: "NonSbuMoleculeCollection"):  # noqa: F821
        """Add two collections of molecules."""
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
        """Get a string describing the composition."""
        return self._get_composition()
