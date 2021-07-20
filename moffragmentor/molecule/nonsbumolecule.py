# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import List

from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.core import Molecule

from ..utils import get_edge_dict


class NonSbuMolecule:
    """Class to handle solvent or other non-SBU molecules"""

    def __init__(
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        indices: List[int],
        connecting_index: int = None,
    ):
        self.molecule = molecule
        self.molecule_graph = molecule_graph
        self.indices = indices
        # We store the connecting index to see which atom we would need to give the elctron from the bond to the metal,
        # it is not used atm but (hopefully) will be
        self.connecting_index = connecting_index

    @property
    def composition(self) -> str:
        return self.molecule.composition.alphabetical_formula

    def __str__(self):
        return str(self.composition)

    def __len__(self):
        return len(self.molecule)

    @classmethod
    def from_structure_graph_and_indices(
        cls, structure_graph: StructureGraph, indices: List[int]
    ) -> object:
        """Create a a new NonSbuMolecule from a part of a structure graph.

        Args:
            structure_graph (StructureGraph): Structure graph with structure attribute
            indices (List[int]): Indices that label nodes in the structure graph,
                indexing the molecule of interest

        Returns:
            NonSbuMolecule: Instance of NonSbuMolecule
        """
        my_graph = deepcopy(structure_graph)
        to_delete = [i for i in range(len(structure_graph)) if i not in indices]
        my_graph.remove_nodes(to_delete)
        structure = my_graph.structure
        sites = []
        for site in structure:
            sites.append(site)
        mol = Molecule.from_sites(sites)
        molecule_graph = MoleculeGraph.with_edges(mol, get_edge_dict(my_graph))
        return cls(mol, molecule_graph, indices)

    def show_molecule(self):
        import nglview  # pre-commit:disable=import-outside-toplevel

        return nglview.show_pymatgen(self.molecule)