# -*- coding: utf-8 -*-
"""Representation for a secondary building block"""
import warnings
from typing import Collection, List

import networkx as nx
import numpy as np
import pubchempy as pcp
from backports.cached_property import cached_property
from loguru import logger
from openbabel import pybel as pb
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from scipy.spatial.distance import pdist


from ..utils import add_suffix_to_dict_keys, pickle_dump
from ..utils.mol_compare import mcs_rank


def ob_mol_without_metals(obmol):
    import openbabel as ob  # pylint: disable=import-outside-toplevel

    mol = obmol.clone
    for atom in ob.OBMolAtomIter(mol.OBMol):  # pylint:disable=no-member
        if atom.IsMetal():
            mol.OBMol.DeleteAtom(atom)

    return mol


__all__ = ["SBU"]


def obmol_to_rdkit_mol(obmol):

    smiles = obmol.write("can").strip()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)  # pylint: disable=no-member
    if mol is None:
        warnings.warn("Attempting to remove metals to generate RDKit molecule")
        new_obmol = ob_mol_without_metals(obmol)
        smiles = new_obmol.write("can").strip()
        mol = Chem.MolFromSmiles(smiles, sanitize=True)  # pylint: disable=no-member

    return mol


class SBU:  # pylint:disable=too-many-instance-attributes, too-many-public-methods
    """Representation for a secondary building block.
    It also acts as container for site indices:

    - graph_branching_indices: are the branching indices according
      to the graph-based definition. They might not be part of the molecule.
    - closest_branching_index_in_molecule: those are always part of the molecule.
      In case the branching index is part of the molecule,
      they are equal to to the graph_branching_indices.
      Otherwise they are obtained as the closest vertex of the original
      branching vertex that is part of the molecule.
    - binding_indices: are the indices of the sites between
        the branching index and metal
    - original_indices: complete original set of indices that has been selected
      for this building blocks
    - persistent_non_metal_bridged: components that are connected
      via a bridge both in the MOF structure
      and building block molecule. No metal is part of the edge,
      i.e., bound solvents are not included in this set
    - terminal_in_mol_not_terminal_in_struct: indices that are terminal
       in the molecule but not terminal in the structure
    """

    def __init__(  # pylint:disable=too-many-arguments
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        center: np.ndarray,
        graph_branching_indices: Collection[int],
        closest_branching_index_in_molecule: Collection[int],
        binding_indices: Collection[int],
        original_indices: Collection[int],
        persistent_non_metal_bridged=None,
        terminal_in_mol_not_terminal_in_struct=None,
        graph_branching_coords=None,
        connecting_paths=None,
    ):
        self.molecule = molecule
        self.center = center
        self._original_indices = original_indices
        self.molecule_graph = molecule_graph
        self._original_graph_branching_indices = graph_branching_indices
        self._original_closest_branching_index_in_molecule = closest_branching_index_in_molecule

        self._persistent_non_metal_bridged = persistent_non_metal_bridged
        self._terminal_in_mol_not_terminal_in_struct = terminal_in_mol_not_terminal_in_struct
        self.graph_branching_coords = graph_branching_coords

        self._original_binding_indices = binding_indices

        self.mapping_from_original_indices = dict(zip(original_indices, range(len(molecule))))
        self.mapping_to_original_indices = dict(zip(range(len(molecule)), original_indices))
        self._indices = original_indices
        self._original_connecting_paths = connecting_paths
        self.connecting_paths = []

        for i in connecting_paths:
            try:
                self.connecting_paths.append(self.mapping_from_original_indices[i])
            except KeyError:
                pass

    def search_pubchem(self, listkey_counts=10, **kwargs) -> tuple:
        """Search for a molecule in pubchem
        Second element of return tuple is true if there was an identity match
        """
        try:
            return pcp.get_compounds(self.smiles, namespace="smiles", **kwargs), True
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                f"Could not find {self.smiles} in pubchem, \
                    now performing substructure search"
            )
            res = pcp.get_compounds(
                self.smiles,
                namespace="smiles",
                searchtype="substructure",
                listkey_counts=listkey_counts,
                **kwargs,
            )
            smiles = [r.canonical_smiles for r in res]

            return mcs_rank(self.smiles, smiles, additional_attributes=res), False

    def get_neighbor_indices(self, site: int) -> List[int]:
        """Get list of indices of neighboring sites"""
        return [site.index for site in self.molecule_graph.get_connected_sites(site)]

    # make this function so we can have different flavors
    def get_indices(self):
        return self._indices

    def __len__(self):
        return len(self.molecule)

    def __str__(self):
        return self.smiles

    def dump(self, path):
        pickle_dump(self, path)

    def _get_nx_graph(self):
        return nx.Graph(self.molecule_graph.graph.to_undirected())

    @cached_property
    def metal_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.molecule.species) if species.is_metal]

    @cached_property
    def nx_graph(self):
        return self._get_nx_graph()

    @property
    def composition(self):
        return self.molecule.composition.alphabetical_formula

    @property
    def cart_coords(self):
        return self.molecule.cart_coords

    @property
    def coordination(self):
        return len(self.original_graph_branching_indices)

    @property
    def original_graph_branching_indices(self):
        return self._original_graph_branching_indices

    @property
    def graph_branching_indices(self):
        return [
            self.mapping_from_original_indices[i] for i in self.original_graph_branching_indices
        ]

    @cached_property
    def branching_coords(self):
        if self.graph_branching_coords is not None:
            return self.graph_branching_coords
        # ToDo: add here also a try capture for the case that the graph
        # branching indices are not part of the molecule
        return self.cart_coords[self.graph_branching_indices]

    @cached_property
    def connecting_indices(self):
        return [
            self.mapping_from_original_indices[i]
            for i in self._original_closest_branching_index_in_molecule
        ]

    @property
    def original_binding_indices(self):
        return self._original_binding_indices

    @cached_property
    def binding_indices(self):
        return [self.mapping_from_original_indices[i] for i in self.original_binding_indices]

    @cached_property
    def rdkit_mol(self):
        return obmol_to_rdkit_mol(self.openbabel_mol)

    @cached_property
    def openbabel_mol(self):
        return self.get_openbabel_mol()

    def get_openbabel_mol(self):
        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        return pm

    def show_molecule(self):
        import nglview  # pylint:disable=import-outside-toplevel

        return nglview.show_pymatgen(self.molecule)

    def show_connecting_structure(self):
        import nglview  # pylint:disable=import-outside-toplevel

        return nglview.show_pymatgen(self._get_connected_sites_structure())

    def show_binding_structure(self):
        import nglview  # pylint:disable=import-outside-toplevel

        return nglview.show_pymatgen(self._get_binding_sites_structure())

    def to(self, fmt: str, filename: str):
        return self.molecule.to(fmt, filename)

    @cached_property
    def smiles(self) -> str:
        """Return canonical SMILES.
        Using openbabel to compute the SMILES, but then get the
        canonical version with RDKit as we observed sometimes the same
        molecule ends up as different canonical SMILES for openbabel.
        If RDKit cannot make a canonical SMILES (can happen with organometallics)
        we simply use the openbabel version.
        """
        mol = self.openbabel_mol
        smiles = mol.write("can").strip()
        try:
            canonical = Chem.CanonSmiles(smiles)
            return canonical
        except Exception:  # pylint: disable=broad-except
            return smiles

    def _get_boxed_structure(self):
        max_size = _get_max_sep(self.molecule.cart_coords)
        structure = self.molecule.get_boxed_structure(
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            reorder=False,
        )
        return structure

    def _get_connected_sites_structure(self):
        sites = []
        s = self._get_boxed_structure()
        for i in self.connecting_indices:
            sites.append(s[i])
        return Structure.from_sites(sites)

    def _get_binding_sites_structure(self):
        sites = []
        s = self._get_boxed_structure()
        for i in self.binding_indices:
            sites.append(s[i])
        return Structure.from_sites(sites)


    @cached_property
    def descriptors(self):
        return self._get_descriptors()


def _get_max_sep(coordinates):
    if len(coordinates) > 1:
        distances = pdist(coordinates)
        return np.max(distances)
    return 5
