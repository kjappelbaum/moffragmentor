# -*- coding: utf-8 -*-
"""Describing an collection of linkers"""
from typing import List

from pymatgen.core import Molecule

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

    def show_linker_structure(self):
        import nglview as nv

        all_linker_sites = []

        for n in self:
            n.molecule.coords = n.cart_coords
            s = n.molecule.sites
            all_linker_sites.extend(s)

        super_node = Molecule.from_sites(all_linker_sites)
        return nv.show_pymatgen(super_node)
