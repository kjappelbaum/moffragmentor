# -*- coding: utf-8 -*-
"""Describing a collection of nodes"""
from hashlib import sha1

from pymatgen.core import Molecule

from .sbucollection import SBUCollection

__all__ = ["NodeCollection"]


class NodeCollection(SBUCollection):
    """Collection of node building blocks"""

    @property
    def building_block_composition(self):
        if self._composition is None:
            self._composition = ["".join(["N", i]) for i in self.sbu_types]
        return self._composition

    def show_node_structure(self):
        import nglview as nv

        all_node_sites = []

        for n in self:
            n.molecule.coords = n.cart_coords
            s = n.molecule.sites
            all_node_sites.extend(s)

        super_node = Molecule.from_sites(all_node_sites)
        return nv.show_pymatgen(super_node)
