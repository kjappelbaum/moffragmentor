# -*- coding: utf-8 -*-
"""Describing the organic building blocks, i.e., linkers."""
from copy import deepcopy

from pymatgen.core import Molecule, PeriodicSite, Structure
from rdkit.Chem import AllChem

from .sbu import SBU


class Linker(SBU):
    """Describe a linker in a MOF"""

    def __repr__(self) -> str:
        """Return a string representation of the linker."""
        return f"Linker ({self.composition})"

    def _get_branching_sites_structure(self) -> Structure:
        new_sites = []
        s = self._get_boxed_structure()
        for i, site in enumerate(s):
            if set([self.mapping_to_original_indices[i]]) & self.original_graph_branching_indices:
                if "original_species" in site.properties and site.properties["original_species"]:
                    species = site.properties["original_species"]
                else:
                    species = site.species

                new_site = PeriodicSite(species, site.coords, site.lattice)
                new_sites.append(new_site)

        return Structure.from_sites(new_sites)


def _get_edge_dict_from_rdkit_mol(mol):
    edges = {}
    for bond in mol.GetBonds():
        edges[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = None
    return edges


def _make_mol_from_rdkit_mol(mol):
    """Take the first conformer"""
    molecule = deepcopy(mol)
    AllChem.EmbedMolecule(molecule)  # pylint:disable=no-member
    # not clear to me why pylint complains
    conf = molecule.GetConformers()
    positions = conf[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Molecule(symbols, positions)
