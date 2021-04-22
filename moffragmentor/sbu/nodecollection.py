# -*- coding: utf-8 -*-
from .sbucollection import SBUCollection


class NodeCollection(SBUCollection):
    @property
    def building_block_composition(self):
        if self._composition is None:
            self._composition = ["".join(["N", i]) for i in self.sbu_types]
        return self._composition
