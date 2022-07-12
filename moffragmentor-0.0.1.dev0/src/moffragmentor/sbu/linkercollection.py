# -*- coding: utf-8 -*-
"""Describing an collection of linkers"""
from typing import List

from .sbucollection import SBUCollection

__all__ = ["LinkerCollection"]


class LinkerCollection(SBUCollection):
    """Collection of linker building blocks"""

    @property
    def building_block_composition(self) -> List[str]:
        """Return a list of strings of building blocks.

        Strings are of the form of L{i} where i is an integer.

        Returns:
            List[str]: List of strings of building blocks.
        """
        if self._composition is None:
            self._composition = ["".join(["L", i]) for i in self.sbu_types]
        return self._composition
