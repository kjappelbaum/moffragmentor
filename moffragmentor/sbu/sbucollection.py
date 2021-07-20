# -*- coding: utf-8 -*-
from collections import Counter
from typing import List

from .sbu import SBU


class SBUCollection:
    def __init__(self, sbus: List[SBU]):
        self.sbus = sbus
        self._sbu_types = None
        self._composition = None
        self._unique_sbus = None
        self._centers = [sbu.center for sbu in self.sbus]
        self._indices = [sbu.indices for sbu in self.sbus]

        assert len(self._indices) == len(self.sbus)
        assert len(self._indices) == len(self.sbus)

    def __len__(self):
        return len(self.sbus)

    def __getitem__(self, index):
        return self.sbus[index]

    def __iter__(self):
        for sbu in self.sbus:
            yield sbu

    @property
    def indices(self):
        return self._indices

    @property
    def centers(self):
        return self._centers

    @property
    def sbu_types(self):
        if not self._sbu_types:
            self._get_unique()
        return self._sbu_types

    @property
    def coordination_numbers(self):
        return [sbu.coordination for sbu in self.sbus]

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
