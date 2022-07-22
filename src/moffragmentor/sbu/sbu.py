# -*- coding: utf-8 -*-
"""Representation for a secondary building block."""
import warnings
from collections import defaultdict
from typing import Collection, List, Optional, Tuple

import networkx as nx
import numpy as np
import pubchempy as pcp
from backports.cached_property import cached_property
from loguru import logger
from openbabel import pybel as pb
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.core import Lattice, Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from scipy.spatial.distance import pdist

from ..utils import pickle_dump
from ..utils.mol_compare import mcs_rank


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
        * closest_branching_index_in_molecule: those are always part of the molecule.
            In case the branching index is part of the molecule,
            they are equal to to the graph_branching_indices.
            Otherwise they are obtained as the closest vertex of the original
            branching vertex that is part of the molecule.
        * binding_indices: are the indices of the sites between
            the branching index and metal
        * original_indices: complete original set of indices that has been selected
            for this building blocks
        * persistent_non_metal_bridged: components that are connected
            via a bridge both in the MOF structure
            and building block molecule. No metal is part of the edge,
            i.e., bound solvents are not included in this set
        * terminal_in_mol_not_terminal_in_struct: indices that are terminal
            in the molecule but not terminal in the structure.
            This is for instance, the case for carboxy groups that are only
            coordinated with one O. In this case, a chemically more faithful
            representation might be to not include the C of the carboxy
            in the node. This collection allows us to do so.

    .. note::

        The coordinates in the molecule object are not the ones directly
        extracted from the MOF. They are the coordinates of sites unwrapped
        to ensure that there are no "broken molecules" .

        To obtain the "original" coordinates, use the `_coordinates` attribute.

    Examples:
        >>> # visualize the molecule
        >>> sbu_object.show_molecule()
        >>> # search pubchem for the molecule
        >>> sbu_object.search_pubchem()
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
        persistent_non_metal_bridged: Optional[Collection[int]] = None,
        terminal_in_mol_not_terminal_in_struct: Optional[Collection[int]] = None,
        graph_branching_coords: Optional[Collection[np.ndarray]] = None,
        connecting_paths: Optional[Collection[int]] = None,
        coordinates: Optional[np.ndarray] = None,
        lattice: Optional[Lattice] = None,
    ):
        """Initialize a secondary building block.

        In practice, you won't use this constructor directly.

        Args:
            molecule (Molecule): Pymatgen molecule object.
            molecule_graph (MoleculeGraph): Pymatgen molecule graph object.
            center (np.ndarray): Center of the SBU.
            graph_branching_indices (Collection[int]): Branching indices
                in the original structure.
            closest_branching_index_in_molecule (Collection[int]):
                Closest branching index in the molecule.
            binding_indices (Collection[int]): Binding indices in the original structure.
            original_indices (Collection[int]): List of all indicies in the original
                structure this SBU corresponds to.
            persistent_non_metal_bridged (Optional[Collection[int]], optional):
                components that are connected via a bridge both in the MOF structure
                and building block molecule. No metal is part of the edge, i.e.,
                bound solvents are not included in this set. Defaults to None.
            terminal_in_mol_not_terminal_in_struct (Optional[Collection[int]], optional):
                Tndices that are terminal in the molecule but not terminal in the structure.
                Defaults to None.
            connecting_paths (Optional[Collection[int]], optional):
                Paths between node atoms and branching atoms. Defaults to None.
            coordinates (Optional[np.ndarray], optional): Coordinates of all atoms in the molecule.
                Defaults to None.
            lattice (Optional[Lattice], optional): Pymatgen Lattice object of the original structure.
                Defaults to None.
        """
        self.molecule = molecule
        self._center = center
        self._original_indices = original_indices
        self.molecule_graph = molecule_graph
        self._original_graph_branching_indices = graph_branching_indices
        self._original_closest_branching_index_in_molecule = closest_branching_index_in_molecule

        self._persistent_non_metal_bridged = persistent_non_metal_bridged
        self._terminal_in_mol_not_terminal_in_struct = terminal_in_mol_not_terminal_in_struct
        self._original_binding_indices = binding_indices

        self.mapping_from_original_indices = defaultdict(list)
        for ori_index, index in zip(self._original_indices, range(len(molecule))):
            self.mapping_from_original_indices[ori_index].append(index)
        self.mapping_to_original_indices = dict(zip(range(len(molecule)), original_indices))
        self._indices = original_indices
        self._original_connecting_paths = connecting_paths
        self.connecting_paths = []
        self._coordinates = coordinates
        self._lattice = lattice
        for i in connecting_paths:
            try:
                for index in self.mapping_from_original_indices[i]:
                    self.connecting_paths.append(index)
            except KeyError:
                pass

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
        # if self._coordinates is not None:
        #    return self._coordinates
        return self.molecule.cart_coords
        # return np.array(self._coordinates)

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
        # todo: we need a mechanism if the branching site is not
        # included in the representation of the node
        # we should at least add a warning here
        # we also should have a fallback, when we create the node,
        # that we have those coordinates wrapped in the same way
        # perhaps we can wrap one where we include this
        indices = []
        for i in self.original_graph_branching_indices:
            for index in self.mapping_from_original_indices[i]:
                indices.append(index)
        return indices

    @cached_property
    def weisfeiler_lehman_graph_hash(self):
        return nx.weisfeiler_lehman_graph_hash(self.molecule_graph.graph, node_attr="specie")

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
        return self.cart_coords[self.graph_branching_indices]

    @cached_property
    def connecting_indices(self):
        indices = []

        for p in self._original_closest_branching_index_in_molecule:
            for index in self.mapping_from_original_indices[p]:
                indices.append(index)

        return indices

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
