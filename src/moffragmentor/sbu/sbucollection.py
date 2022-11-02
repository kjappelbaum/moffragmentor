# -*- coding: utf-8 -*-
"""Collection for MOF building blocks"""
from collections import Counter
from typing import List

from backports.cached_property import cached_property

from .sbu import SBU


class SBUCollection:
    """Container for a collection of SBUs"""

    def __init__(self, sbus: List[SBU]):
        """Construct a SBUCollection.

        Args:
            sbus (List[SBU]): List of SBU objects.

        Raises:
            ValueError: If there is an unexpected number of SBUs
                (inconsistency in the extracted indices and the SBUs).
        """
        self.sbus = sbus
        self._sbu_types = None
        self._composition = None
        self._unique_sbus = None
        self._centers = [sbu.center for sbu in self.sbus]
        self._indices = [sbu.get_indices() for sbu in self.sbus]

        if len(self._indices) != len(self.sbus):
            raise ValueError("Number of SBUs is inconsistent")

    def __len__(self) -> int:
        """Return number of SBUs."""
        return len(self.sbus)

    def __getitem__(self, index) -> SBU:
        """Return SBU at index."""
        return self.sbus[index]

    def __iter__(self):
        """Iterate over SBUs (generator)."""
        for sbu in self.sbus:
            yield sbu

    @property
    def indices(self):
        return set(sum(self._indices, []))

    @property
    def centers(self):
        return self._centers

    @cached_property
    def smiles(self):
        """
        Return a list of the SMILES strings of the SBUs.

        Returns:
            List[str]: A list of smiles strings.
        """
        return [sbu.smiles for sbu in self.sbus]

    @property
    def sbu_types(self):
        if not self._sbu_types:
            self._get_unique()
        return self._sbu_types

    @property
    def coordination_numbers(self):
        return [sbu.coordination for sbu in self.sbus]

    @property
    def molar_masses(self):
        return [sbu.molar_mass for sbu in self.sbus]

    @property
    def unique_sbus(self):
        if not self._sbu_types:
            self._get_unique()
        return self._unique_sbus

    @property
    def sbu_properties(self):
        return dict(zip(self.sbu_types, self.coordination_numbers))

    def _get_unique(self):
        all_strings = [str(sbu) for sbu in self.sbus]
        unique_strings = set(all_strings)
        unique_mapping = []
        for string in all_strings:
            for i, unique_string in enumerate(unique_strings):
                if string == unique_string:
                    unique_mapping.append(i)
                    break
        self._unique_sbus = unique_strings
        self._sbu_types = unique_mapping

    def _get_composition(self):
        if self._composition is None:
            composition = []
            for mol in self.sbus:
                composition.append(str(mol.composition))
            composition_counter = Counter(composition)
            self._composition = dict(composition_counter)

        return self._composition

    @property
    def composition(self):
        return self._get_composition()

    def __repr__(self) -> str:
        """Return string representation."""
        return f"SBUCollection({self.composition})"
