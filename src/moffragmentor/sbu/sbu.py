# -*- coding: utf-8 -*-
"""Representation for a secondary building block."""
import warnings
from collections import defaultdict
from typing import Collection, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pubchempy as pcp
from backports.cached_property import cached_property
from loguru import logger
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from scipy.spatial.distance import pdist

from moffragmentor.utils import pickle_dump
from moffragmentor.utils.mol_compare import mcs_rank


def ob_mol_without_metals(obmol):
    """Remove metals from an OpenBabel molecule."""
    import openbabel as ob

    mol = obmol.clone
    for atom in ob.OBMolAtomIter(mol.OBMol):
        if atom.IsMetal():
            mol.OBMol.DeleteAtom(atom)

    return mol


__all__ = ["SBU"]


def obmol_to_rdkit_mol(obmol):
    """Convert an OpenBabel molecule to a RDKit molecule."""
    smiles = obmol.write("can").strip()
    mol = Chem.MolFromSmiles(smiles, sanitize=True)
    if mol is None:
        warnings.warn("Attempting to remove metals to generate RDKit molecule")
        new_obmol = ob_mol_without_metals(obmol)
        smiles = new_obmol.write("can").strip()
        mol = Chem.MolFromSmiles(smiles, sanitize=True)

    return mol


class SBU:
    """Representation for a secondary building block.

    It also acts as container for site indices:

        * graph_branching_indices: are the branching indices according
            to the graph-based definition. They might not be part of the molecule.
        * binding_indices: are the indices of the sites between
            the branching index and metal
        * original_indices: complete original set of indices that has been selected
            for this building blocks

    .. note::

        The coordinates in the molecule object are not the ones directly
        extracted from the MOF. They are the coordinates of sites unwrapped
        to ensure that there are no "broken molecules" .

        To obtain the "original" coordinates, use the `_coordinates` attribute.

    .. note:: Dummy molecules

        In dummy molecules the binding and branching sites are replaces by
        dummy atoms (noble gas). They also have special properties that indicate
        the original species.

    Examples:
        >>> # visualize the molecule
        >>> sbu_object.show_molecule()
        >>> # search pubchem for the molecule
        >>> sbu_object.search_pubchem()
    """

    def __init__(
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        graph_branching_indices: Collection[int],
        binding_indices: Collection[int],
        molecule_original_indices_mapping: Optional[Dict[int, List[int]]] = None,
        dummy_molecule: Optional[Molecule] = None,
        dummy_molecule_graph: Optional[MoleculeGraph] = None,
        dummy_molecule_indices_mapping: Optional[Dict[int, List[int]]] = None,
        dummy_branching_indices: Optional[Collection[int]] = None,
    ):
        """Initialize a secondary building block.

        In practice, you won't use this constructor directly.

        Args:
            molecule (Molecule): Pymatgen molecule object.
            molecule_graph (MoleculeGraph): Pymatgen molecule graph object.
            graph_branching_indices (Collection[int]): Branching indices
                in the original structure.
            binding_indices (Collection[int]): Binding indices in the original structure.
            molecule_original_indices_mapping (Optional[Dict[int, List[int]]], optional):
                Mapping from molecule indices to original indices. Defaults to None.
            dummy_molecule (Optional[Molecule], optional): Dummy molecule. Defaults to None.
            dummy_molecule_graph (Optional[MoleculeGraph], optional): Dummy molecule graph.
                Defaults to None.
            dummy_molecule_indices_mapping (Optional[Dict[int, List[int]]], optional):
                Dummy molecule indices mapping. Defaults to None.
            dummy_branching_indices (Optional[Collection[int]], optional):
                Dummy branching indices. Defaults to None.
        """
        self.molecule = molecule
        self._mapping = molecule_original_indices_mapping

        self._indices = sum(list(molecule_original_indices_mapping.values()), [])

        self._original_indices = self._indices
        self.molecule_graph = molecule_graph
        self._original_graph_branching_indices = graph_branching_indices
        self._original_binding_indices = binding_indices

        self._dummy_molecule = dummy_molecule
        self._dummy_molecule_graph = dummy_molecule_graph
        self._dummy_molecule_indices_mapping = dummy_molecule_indices_mapping
        self._dummy_branching_indices = dummy_branching_indices

        self.mapping_from_original_indices = defaultdict(list)
        if molecule_original_indices_mapping is None:
            for ori_index, index in zip(self._indices, range(len(molecule))):
                self.mapping_from_original_indices[ori_index].append(index)
        else:
            for k, v in molecule_original_indices_mapping.items():
                for i in v:
                    self.mapping_from_original_indices[i].append(k)

        if dummy_molecule:
            self.dummy_mapping_from_original_indices = defaultdict(list)
            if dummy_molecule_indices_mapping is None:
                for ori_index, index in zip(self._indices, range(len(dummy_molecule))):
                    self.dummy_mapping_from_original_indices[ori_index].append(index)
            else:
                for k, v in dummy_molecule_indices_mapping.items():
                    for i in v:
                        self.dummy_mapping_from_original_indices[i].append(k)

        self.mapping_to_original_indices = {}
        for key, value in self.mapping_from_original_indices.items():
            for v in value:
                self.mapping_to_original_indices[v] = key

    @property
    def center(self):
        return self.molecule.center_of_mass

    def search_pubchem(self, listkey_counts: int = 10, **kwargs) -> Tuple[List[str], bool]:
        """Search for a molecule in pubchem # noqa: DAR401

        Second element of return tuple is true if there was an identity match

        Args:
            listkey_counts (int): Number of list keys to return
                (relevant for substructure search).
                Defaults to 10.
            kwargs: Additional arguments to pass to PubChem.search

        Returns:
            Tuple[List[str], bool]: List of pubchem ids and whether there was an identity match
        """
        try:
            matches = pcp.get_compounds(
                self.smiles, namespace="smiles", searchtype="fastidentity", **kwargs
            )
            if matches:
                return (
                    matches,
                    True,
                )
            else:
                raise ValueError("No matches found")
        except Exception:
            logger.warning(
                f"Could not find {self.smiles} in pubchem, \
                    now performing substructure search"
            )
            # we use `fastsubstructure` as it fixes
            # https://github.com/kjappelbaum/moffragmentor/issues/63
            res = pcp.get_compounds(
                self.smiles,
                namespace="smiles",
                searchtype="fastsubstructure",
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

    @property
    def is_edge(self):
        return len(self.branching_coords) == 2

    def __len__(self):
        """Return the number of atoms in the molecule."""
        return len(self.molecule)

    def __str__(self):
        """Return the SMILES string for the molecule."""
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

    @cached_property
    def mol_with_coords(self):
        mol = self.molecule.copy()
        mol.coords = self.cart_coords
        sites = mol.sites
        return Molecule.from_sites(sites)

    @property
    def coordination(self):
        return len(self.original_graph_branching_indices)

    @property
    def original_graph_branching_indices(self):
        return self._original_graph_branching_indices

    @property
    def graph_branching_indices(self):
        indices = []
        if self._dummy_branching_indices is None:
            for i in self.original_graph_branching_indices:
                for index in self.mapping_from_original_indices[i]:
                    indices.append(index)
        else:
            for i in self._dummy_branching_indices:
                for index in self.dummy_mapping_from_original_indices[i]:
                    indices.append(index)
        return indices

    @cached_property
    def weisfeiler_lehman_graph_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.molecule_graph.graph, node_attr="specie")

    @property
    def molar_mass(self):
        return self.molecule.composition.weight

    @cached_property
    def hash(self) -> str:
        """Return hash.

        The hash is a combination of Weisfeiler-Lehman graph hash and center.

        Returns:
            str: Hash.
        """
        wl_hash = self.weisfeiler_lehman_graph_hash
        center = self.molecule.cart_coords.mean(axis=0)
        return f"{wl_hash}-{center[0]:.2f}-{center[1]:.2f}-{center[2]:.2f}"

    def __eq__(self, other: "SBU") -> bool:
        """Check if two molecules are equal.

        Based on the Weisfeiler-Lehman graph hash and center of mass.

        Args:
            other (SBU): SBU to compare to.

        Returns:
            bool: True if equal, False otherwise.
        """
        if hash(self) != hash(other):
            return False
        return True

    @cached_property
    def branching_coords(self):
        return (
            self.cart_coords[self.graph_branching_indices]
            if self._dummy_branching_indices is None
            else self._dummy_molecule.cart_coords[self.graph_branching_indices]
        )

    @property
    def original_binding_indices(self):
        return self._original_binding_indices

    @cached_property
    def binding_indices(self):
        indices = []
        for i in self.original_binding_indices:
            for index in self.mapping_from_original_indices[i]:
                indices.append(index)
        return indices

    @cached_property
    def rdkit_mol(self):
        return obmol_to_rdkit_mol(self.openbabel_mol)

    @cached_property
    def openbabel_mol(self):
        return self.get_openbabel_mol()

    def get_openbabel_mol(self):
        from openbabel import pybel as pb

        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        return pm

    def show_molecule(self):
        import nglview

        return nglview.show_pymatgen(self.molecule)

    def show_connecting_structure(self):
        import nglview

        return nglview.show_pymatgen(self._get_connected_sites_structure())

    def show_binding_structure(self):
        import nglview

        return nglview.show_pymatgen(self._get_binding_sites_structure())

    def to(self, fmt: str, filename: str):
        return self.molecule.to(fmt, filename)

    @cached_property
    def smiles(self) -> str:
        """Return canonical SMILES.

        Use openbabel to compute the SMILES, but then get the
        canonical version with RDKit as we observed sometimes the same
        molecule ends up as different canonical SMILES for openbabel.
        If RDKit cannot make a canonical SMILES (can happen with organometallics)
        we simply use the openbabel version.

        Returns:
            str: Canonical SMILES
        """
        mol = self.openbabel_mol
        smiles = mol.write("can").strip()
        try:
            canonical = Chem.CanonSmiles(smiles)
            return canonical
        except Exception:
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
