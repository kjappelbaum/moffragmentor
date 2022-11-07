# -*- coding: utf-8 -*-
"""Defining the main representation of a MOF."""
import os
from collections import defaultdict
from typing import Dict, List, Optional, Union

import networkx as nx
import numpy as np
from backports.cached_property import cached_property
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from structuregraph_helpers.create import VestaCutoffDictNN

from .descriptors.sbu_dimensionality import get_structure_graph_dimensionality
from .fragmentor import FragmentationResult, run_fragmentation
from .fragmentor.branching_points import get_branch_points
from .utils import IStructure, pickle_dump, write_cif

__all__ = ["MOF"]


class MOF:
    """Main representation for a MOF structure.

    This container holds a structure and its associated graph.
    It also provides some convenience methods for getting neighbors
    or results of the fragmentation.

    Internally, this code typically uses IStructure objects to
    avoid bugs due to the mutability of Structure objects
    (e.g. the fragmentation code performs operations on the structure
    and we want to be sure that there is no impact on the input).

    Examples:
        >>> from moffragmentor import MOF
        >>> mof = MOF(structure, structure_graph)
        >>> # equivalent is to read from a cif file
        >>> mof = MOF.from_cif(cif_file)
        >>> # visualize the structure
        >>> mof.show_structure()
        >>> # get the neighbors of a site
        >>> mof.get_neighbor_indices(0)
        >>> # perform fragmentation
        >>> fragments mof.fragment()
    """

    def __init__(self, structure: Structure, structure_graph: StructureGraph):
        """Initialize a MOF object.

        Args:
            structure (Structure): Pymatgen Structure object
            structure_graph (StructureGraph): Pymatgen StructureGraph object
        """
        self._structure = structure
        # checker = MOFChecker(structure)
        # if checker.has_atomic_overlaps:
        #     raise ValueError("Structure has atomic overlaps.")

        self._structure_graph = structure_graph
        self._bridges = None
        self._nx_graph = None
        nx.set_node_attributes(
            self._structure_graph.graph,
            name="idx",
            values=dict(zip(range(len(structure_graph)), range(len(structure_graph)))),
        )

    def _reset(self):
        """Reset all parameters that are computed at some point."""
        self._bridges = None
        self._nx_graph = None

    def __copy__(self):
        """Make a a new MOF object with copies of the same structure and structure graph."""
        return MOF(IStructure.from_sites(self._structure.sites), self.structure_graph.__copy__())

    def dump(self, path) -> None:
        """Dump this object as pickle file"""
        pickle_dump(self, path)

    def __len__(self) -> str:
        """Length of the MOF. Equivalent to the number of sites."""
        return len(self.structure)

    @property
    def structure(self):
        return self._structure

    @property
    def structure_graph(self):
        return self._structure_graph

    @cached_property
    def dimensionality(self):
        return get_structure_graph_dimensionality(self.structure_graph)

    @property
    def lattice(self) -> Lattice:
        return self.structure.lattice

    @property
    def composition(self) -> str:
        return self.structure.composition.alphabetical_formula

    @property
    def cart_coords(self) -> np.ndarray:
        return self.structure.cart_coords

    @cached_property
    def frac_coords(self) -> np.ndarray:
        """Return fractional coordinates of the structure.

        We cache this call as pymatgen seems to re-compute this.

        Returns:
            np.ndarray: fractional coordinates of the structure
                in array of shape (n_sites, 3)
        """
        return self.structure.frac_coords

    @classmethod
    def from_cif(
        cls,
        cif: Union[str, os.PathLike],
        symprec: Optional[float] = None,
        angle_tolerance: Optional[float] = None,
        get_primitive: bool = True,
    ):
        """Initialize a MOF object from a cif file.

        Note that this method, by default, symmetrizes the structure.

        Args:
            cif (str): path to the cif file
            symprec (float, optional): Symmetry precision
            angle_tolerance (float, optional): Angle tolerance
            get_primitive (bool): Whether to get the primitive cell

        Returns:
            MOF: MOF object
        """
        # using the IStructure avoids bugs where somehow the structure changes
        structure = IStructure.from_file(cif)
        if get_primitive:
            structure = structure.get_primitive_structure(0.5)
        if symprec is not None and angle_tolerance is not None:
            spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            structure = spga.get_conventional_standard_structure()
        structure = IStructure.from_sites(structure)
        structure_graph = StructureGraph.with_local_env_strategy(structure, VestaCutoffDictNN)
        return cls(structure, structure_graph)

    @classmethod
    def from_structure(
        cls,
        structure: Structure,
        symprec: Optional[float] = 0.5,
        angle_tolerance: Optional[float] = 5,
    ):
        if (symprec is not None) and (angle_tolerance is not None):
            spga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
            structure = spga.get_conventional_standard_structure()
        structure = IStructure.from_sites(structure)
        structure_graph = StructureGraph.with_local_env_strategy(structure, VestaCutoffDictNN)
        return cls(structure, structure_graph)

    def _is_terminal(self, index):
        return len(self.get_neighbor_indices(index)) == 1

    @cached_property
    def terminal_indices(self) -> List[int]:
        """Return the indices of the terminal sites.

        A terminal site is a site that has only one neighbor.
        And is connected via a bridge to the rest of the structure.
        That means, splitting the bond between the terminal site
        and the rest of the structure will increase the number
        of connected components.

        Typical examples of terminal sites are hydrogren atoms,
        or halogen functional groups.

        Returns:
            List[int]: indices of the terminal sites
        """
        return [i for i in range(len(self.structure)) if self._is_terminal(i)]

    def _get_nx_graph(self):
        if self._nx_graph is None:
            self._nx_graph = nx.Graph(self.structure_graph.graph.to_undirected())
        return self._nx_graph

    def _leads_to_terminal(self, edge):
        sorted_edge = sorted(edge)
        try:
            bridge_edge = self.bridges[sorted_edge[0]]
            return sorted_edge[1] in bridge_edge
        except KeyError:
            return False

    @cached_property
    def nx_graph(self):
        """Structure graph as networkx graph object"""
        return self._get_nx_graph()

    def _generate_bridges(self) -> Dict[int, int]:
        if self._bridges is None:
            bridges = list(nx.bridges(self.nx_graph))

            bridges_dict = defaultdict(list)
            for key, value in bridges:
                bridges_dict[key].append(value)

            self._bridges = dict(bridges_dict)
        return self._bridges

    @cached_property
    def bridges(self) -> Dict[int, int]:
        """Get a dictionary of bridges.

        Bridges are edges in a graph that, if deleted, increase the number of connected components.

        Returns:
            Dict[int, int]: dictionary of bridges
        """
        return self._generate_bridges()

    @cached_property
    def adjaceny_matrix(self):
        return nx.adjacency_matrix(self.structure_graph.graph)

    def show_adjacency_matrix(self, highlight_metals=False):
        """Plot structure graph as adjaceny matrix"""
        import matplotlib.pylab as plt  # pylint:disable=import-outside-toplevel

        matrix = self.adjaceny_matrix.todense()
        if highlight_metals:
            cols = np.nonzero(matrix[self.metal_indices, :])
            rows = np.nonzero(matrix[:, self.metal_indices])
            matrix[self.metal_indices, cols] = 2
            matrix[rows, self.metal_indices] = 2
        plt.imshow(self.adjaceny_matrix.todense(), cmap="Greys_r")

    @cached_property
    def _branching_indices_list(self):
        return get_branch_points(self)

    def _is_branch_point(self, index):
        return index in self._branching_indices_list

    @cached_property
    def metal_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if species.is_metal]

    @cached_property
    def h_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if str(species) == "H"]

    @cached_property
    def c_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if str(species) == "C"]

    @cached_property
    def n_indices(self) -> List[int]:
        return [i for i, species in enumerate(self.structure.species) if str(species) == "N"]

    def get_neighbor_indices(self, site: int) -> List[int]:
        """Get list of indices of neighboring sites."""
        return [site.index for site in self.structure_graph.get_connected_sites(site)]

    def get_symbol_of_site(self, site: int) -> str:
        """Get elemental symbol of site indexed site."""
        return str(self.structure[site].specie)

    def show_structure(self):
        """Visualize structure using nglview."""
        import nglview  # pylint:disable=import-outside-toplevel

        return nglview.show_pymatgen(self.structure)

    def _fragment(
        self, check_dimensionality, create_single_metal_bus, break_organic_nodes_at_metal
    ):
        fragmentation_result = run_fragmentation(
            self,
            check_dimensionality,
            create_single_metal_bus,
            break_organic_nodes_at_metal=break_organic_nodes_at_metal,
        )
        return fragmentation_result

    def fragment(
        self,
        check_dimensionality: bool = True,
        create_single_metal_bus: bool = False,
        break_organic_nodes_at_metal: bool = True,
    ) -> FragmentationResult:
        """Split the MOF into building blocks.

        The building blocks are linkers, nodes, bound,
        unbound solvent, net embedding of those building blocks.

        Args:
            check_dimensionality (bool): Check if the node is 0D.
                If not, split into isolated metals.
                Defaults to True.
            create_single_metal_bus (bool): Create a single metal BUs.
                Defaults to False.
            break_organic_nodes_at_metal (bool): Break nodes into single metal BU
                if they appear "too organic".

        Returns:
            FragmentationResult: FragmentationResult object.
        """
        return self._fragment(
            check_dimensionality,
            create_single_metal_bus,
            break_organic_nodes_at_metal=break_organic_nodes_at_metal,
        )

    def _get_cif_text(self) -> str:
        return write_cif(self.structure, self.structure_graph, [])

    def write_cif(self, filename) -> None:
        """Write the structure to a CIF file."""
        with open(filename, "w", encoding="utf8") as file_handle:
            file_handle.write(self._get_cif_text())
