# -*- coding: utf-8 -*-
"""Describing a collection of nodes"""
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

    def _node_structure(self):
        all_node_sites = []

        for n in self:
            n.molecule.coords = n.cart_coords
            s = n.molecule.sites
            all_node_sites.extend(s)

        super_node = Molecule.from_sites(all_node_sites)
        return super_node

    def show_node_structure(self):
        import nglview as nv

        super_node = self._node_structure()
        return nv.show_pymatgen(super_node)
