# -*- coding: utf-8 -*-
from .sbucollection import SBUCollection


class LinkerCollection(SBUCollection):
    @property
    def building_block_composition(self):
        if self._composition is None:
            self._composition = ["".join(["L", i]) for i in self.sbu_types]
        return self._composition
