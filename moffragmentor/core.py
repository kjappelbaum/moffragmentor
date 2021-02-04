# -*- coding: utf-8 -*-
__all__ = ["get_subgraphs_as_molecules", "MOF"]


import os
from copy import deepcopy
from typing import List, Tuple, Union
import warnings
import matplotlib.pylab as plt
import networkx as nx
import numpy as np
from pymatgen import Molecule, Structure
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import CrystalNN, CutOffDictNN, JmolNN, VoronoiNN
import tempfile
from .sbu import Linker, Node
import timeout_decorator
VestaCutoffDictNN = CutOffDictNN.from_preset("vesta_2019")


JULIA_AVAILABLE = False
try:
    from julia.api import Julia

    jl = Julia(compiled_modules=False)
    from julia import CrystalNets

    JULIA_AVAILABLE = True
except ImportError:
    warnings.warn(
        "Could not load the julia package for topology determination.", UserWarning
    )


@timeout_decorator.timeout(60)
def call_crystalnet(f):
    c, n = CrystalNets.do_clustering(
        CrystalNets.parse_chemfile(f), CrystalNets.MOFClustering
    )
    code = CrystalNets.reckognize_topology(CrystalNets.topological_genome(n))
    return code

def get_topology_code(structure: Structure) -> str:
    code = ""
    if JULIA_AVAILABLE:
        with tempfile.NamedTemporaryFile(suffix='.cif') as temp:
            try:
               structure.to("cif", temp.name)
            except Exception as e:
                warnings.warn("Could not determine topology", UserWarning)
    return code


def get_subgraphs_as_molecules(structure_graph: StructureGraph, use_weights=False):
    """Copied from
    http://pymatgen.org/_modules/pymatgen/analysis/graphs.html#StructureGraph.get_subgraphs_as_molecules
    and removed the duplicate check
    Args:
        structure_graph ( pymatgen.analysis.graphs.StructureGraph): Structuregraph
    Returns:
        List: list of molecules
    """
    # ideally, we flag those atoms that are connected to the metals in some special way and make sure
    # this gets propagated

    # creating a supercell is an easy way to extract
    # molecules (and not, e.g., layers of a 2D crystal)
    # without adding extra logic
    supercell_sg = structure_graph * (3, 3, 3)

    # make undirected to find connected subgraphs
    supercell_sg.graph = nx.Graph(supercell_sg.graph)

    # find subgraphs
    all_subgraphs = [
        supercell_sg.graph.subgraph(c)
        for c in nx.connected_components(supercell_sg.graph)
    ]

    # discount subgraphs that lie across *supercell* boundaries
    # these will subgraphs representing crystals
    molecule_subgraphs = []
    for subgraph in all_subgraphs:
        intersects_boundary = any(
            [d["to_jimage"] != (0, 0, 0) for u, v, d in subgraph.edges(data=True)]
        )
        if not intersects_boundary:
            molecule_subgraphs.append(nx.MultiDiGraph(subgraph))

    # add specie names to graph to be able to test for isomorphism
    for subgraph in molecule_subgraphs:
        for node in subgraph:
            subgraph.add_node(node, specie=str(supercell_sg.structure[node].specie))

    unique_subgraphs = []

    def node_match(n1, n2):
        return n1["specie"] == n2["specie"]

    def edge_match(e1, e2):
        if use_weights:
            return e1["weight"] == e2["weight"]
        else:
            return True

    for subgraph in molecule_subgraphs:

        already_present = [
            nx.is_isomorphic(subgraph, g, node_match=node_match, edge_match=edge_match)
            for g in unique_subgraphs
        ]

        if not any(already_present):
            unique_subgraphs.append(subgraph)

    def make_mols(molecule_subgraphs=molecule_subgraphs, center=False):
        molecules = []
        indices = []
        for subgraph in molecule_subgraphs:
            coords = [supercell_sg.structure[n].coords for n in subgraph.nodes()]
            species = [supercell_sg.structure[n].specie for n in subgraph.nodes()]
            binding = [
                supercell_sg.structure[n].properties["binding"]
                for n in subgraph.nodes()
            ]
            idx = [n for n in subgraph.nodes()]
            molecule = Molecule(species, coords, site_properties={"binding": binding})

            # shift so origin is at center of mass
            if center:
                molecule = molecule.get_centered_molecule()
            indices.append(idx)
            molecules.append(molecule)
        return molecules, indices

    #     molecules, indices = make_mols(molecule_subgraphs)
    molecules_unique, _ = make_mols(unique_subgraphs, center=True)

    def relabel_graph(multigraph):
        mapping = dict(zip(multigraph, range(0, len(multigraph.nodes()))))
        return nx.readwrite.json_graph.adjacency_data(
            nx.relabel_nodes(multigraph, mapping)
        )

    return molecules_unique, [
        MoleculeGraph(mol, relabel_graph(graph))
        for mol, graph in zip(molecules_unique, unique_subgraphs)
    ]


class MOF:
    def __init__(self, structure: Structure, structure_graph: StructureGraph):
        self.structure = structure
        self.structure_graph = structure_graph
        self._node_indices = None
        self._linker_indices = None
        self.nodes = []
        self.linker = []
        self._topology = None

    @classmethod
    def from_cif(cls, cif: Union[str, os.PathLike]):
        s = Structure.from_file(cif)
        sg = StructureGraph.with_local_env_strategy(s, VestaCutoffDictNN)
        return cls(s, sg)

    @property
    def topology(self):
        if not self._topology:
            self._topology = get_topology_code(self.structure)
            return self.topology
        return self._topology
        
    @property
    def adjaceny_matrix(self):
        return nx.adjacency_matrix(self.structure_graph.graph)

    def show_adjacency_matrix(self, highlight_metals=False):
        matrix = self.adjaceny_matrix.todense()
        if highlight_metals:
            cols = np.nonzero(matrix[self.metal_indices, :])
            rows = np.nonzero(matrix[:, self.metal_indices])
            matrix[self.metal_indices, cols] = 2
            matrix[rows, self.metal_indices] = 2
        plt.imshow(self.adjaceny_matrix.todense(), cmap="Greys_r")

    @property
    def metal_indices(self) -> List[int]:
        return [
            i for i, species in enumerate(self.structure.species) if species.is_metal
        ]

    @property
    def h_indices(self) -> List[int]:
        return [
            i for i, species in enumerate(self.structure.species) if str(species) == "H"
        ]

    def get_neighbor_indices(self, site: int) -> List[int]:
        return [site.index for site in self.structure_graph.get_connected_sites(site)]

    def get_symbol_of_site(self, site: int) -> str:
        return str(self.structure[site].specie)

    @property
    def node_indices(self) -> set:
        if self._node_indices is None:
            node_indices = self._get_node_indices()
        else:
            node_indices = self._node_indices

        return node_indices

    @property
    def linker_indices(self) -> set:
        node_indices = self.node_indices

        return set(range(len(self.structure))) - node_indices

    def _label_site(self, site: int):
        self.structure[site].properties = {"binding": True}

    def _label_structure(self):
        """Label node and linker atoms that are connected"""
        for metal_idx in self.metal_indices:
            neighbor_indices = self.get_neighbor_indices(metal_idx)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in self.linker_indices:
                    self._label_site(metal_idx)
                    self._label_site(neighbor_idx)

    def _fragment(self) -> Tuple[List[Linker], List[Node]]:
        self._label_structure()
        sg0 = deepcopy(self.structure_graph)
        sg1 = deepcopy(self.structure_graph)
        sg0.remove_nodes(list(self.linker_indices))
        sg1.remove_nodes(list(self.node_indices))
        node_molecules, node_graphs = get_subgraphs_as_molecules(sg0)
        linker_molecules, linker_graphs = get_subgraphs_as_molecules(sg1)
        linkers = [
            Linker.from_labled_molecule(molecule, graph)
            for molecule, graph in zip(linker_molecules, linker_graphs)
        ]

        nodes = [
            Node.from_labled_molecule(molecule, graph)
            for molecule, graph in zip(node_molecules, node_graphs)
        ]

        self.nodes = nodes
        self.linkers = linkers
        return linkers, nodes

    def fragment(self):
        return self._fragment()

    def _get_node_indices(self):
        # make a set of all metals and atoms connected to them:
        h_indices = set(self.h_indices)
        metals_and_neighbor_indices = set()
        node_atom_set = set(self.metal_indices)

        for metal_index in self.metal_indices:
            metals_and_neighbor_indices.add(metal_index)
            bonded_to_metal = self.get_neighbor_indices(metal_index)
            metals_and_neighbor_indices.update(bonded_to_metal)

        # in a node there might be some briding O/OH
        # this is an approximation there might be other bridging modes
        for index in metals_and_neighbor_indices:
            neighboring_indices = self.get_neighbor_indices(index)
            if len(set(neighboring_indices) - h_indices - node_atom_set) == 0:
                node_atom_set.update(set([index]))
                for neighbor_of_neighbor in neighboring_indices:
                    if self.get_symbol_of_site(neighbor_of_neighbor) == "H":
                        node_atom_set.update(set([neighbor_of_neighbor]))

        self._node_indices = node_atom_set
        return node_atom_set
