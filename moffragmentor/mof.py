# -*- coding: utf-8 -*-
__all__ = ["get_subgraphs_as_molecules", "MOF"]


import os
import tempfile
import warnings
from copy import deepcopy
from typing import List, Tuple, Union

import matplotlib.pylab as plt
import networkx as nx
import numpy as np
import timeout_decorator
from pymatgen import Structure
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import (
    CrystalNN,
    CutOffDictNN,
    JmolNN,
    MinimumDistanceNN,
    VoronoiNN,
)

from .sbu import Linker, Node
from .utils import  pickle_dump, write_cif
from .fragmentor import fragment_all_node, fragment_oxo_node, get_subgraphs_as_molecules, is_valid_node, is_valid_linker

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
        with tempfile.NamedTemporaryFile(suffix=".cif") as temp:
            try:
                structure.to(fmt="cif", filename=temp.name)
                code = call_crystalnet(temp.name)
            except Exception as e:
                warnings.warn(f"Could not determine topology {e}", UserWarning)
    return code



class MOF:
    
    def __init__(self, structure: Structure, structure_graph: StructureGraph):
        self.structure = structure
        self.structure_graph = structure_graph
        self._node_indices = None
        self._linker_indices = None
        self.nodes = []
        self.linker = []
        self._topology = None
        self._connecting_node_indices  = None
        self.meta = {}
        self._filter_solvent = True
        self.fragmentation_method =  'all_node'
        self._solvent_indices = None

    def _reset(self):
        self._node_indices = None
        self._linker_indices = None
        self.nodes = []
        self.linker = []
        self._topology = None
        self._connecting_node_indices  = None
        self._solvent_indices = None
        for site in range(len(self.structure)):
            self.structure[site].properties = {}

    def dump(self, path):
        """Dump this object as pickle file"""
        pickle_dump(self, path)

    # Todo: use properties and property setters 
    def set_meta(self, key, value):
        """Set metadata"""
        self.meta[key] = value

    @property
    def fragmentation_method(self):
        return self._fragmentation_method

    @fragmentation_method.setter
    def fragmentation_method(self, value):
        assert value in ["all_node", "oxo"]
        self._fragmentation_method  = value
        self._reset()

    def set_use_solvent_filter(self, use_solvent_filter):
        assert isinstance(use_solvent_filter, bool)
        self._filter_solvent = use_solvent_filter


    @classmethod
    def from_cif(cls, cif: Union[str, os.PathLike]):
        s = Structure.from_file(cif)
        sg = StructureGraph.with_local_env_strategy(s, VestaCutoffDictNN)
        return cls(s, sg)

    def _is_branch_point(self, index):
        valid_connections = 0
        connected_sites = self.get_neighbor_indices(index)
        if len(connected_sites) < 3:
            return False
        
        # ToDo: generalize to consider if this is yields to a terminal thing branch or not
        for connected_site in connected_sites:
            if len(self.get_neighbor_indices(connected_site)) > 1:
                valid_connections += 1

        return valid_connections >= 3

    def _is_terminal(self, index):
        return len(self.get_neighbor_indices(index)) == 1

    @property
    def topology(self):
        if not isinstance(self._topology, str):
            self._topology = get_topology_code(self.structure)
            return self.topology
        return self._topology

    @property
    def _undirected_graph(self):
        return self.structure_graph.graph.to_undirected()

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
        """Returns node indices from cache if they already have been determined, otherwise calculates them"""
        if self._node_indices is None:
            self._get_node_indices()

        return self._node_indices

    @property
    def linker_indices(self) -> set:
        if self._linker_indices is None:
            node_indices = self.node_indices
            self._linker_indices =  set(range(len(self.structure))) 
            self._linker_indices -= node_indices 
            self._linker_indices -= set(sum(self._solvent_indices, []))
        return self._linker_indices

    def _label_site(self, site: int):
        """adds the binding property to a structure"""
        self.structure[site].properties = {"binding": True}

    def _label_structure(self):
        """Label node and linker atoms that are connected"""
        for branching_atom in self._connecting_node_indices:
            neighbor_indices = self.get_neighbor_indices(branching_atom)
            for neighbor_idx in neighbor_indices:
                if neighbor_idx in self.linker_indices:
                    self._label_site(branching_atom)
                    self._label_site(neighbor_idx)

    def _fragment(self) -> Tuple[List[Linker], List[Node]]:
        node_indices  = self.node_indices
        self._label_structure()
        sg1 = deepcopy(self.structure_graph)
        linkers = []
        nodes = []
        node_structure = Structure.from_sites(
            [self.structure[i] for i in node_indices]
        )
        sg0 = StructureGraph.with_local_env_strategy(
            node_structure, MinimumDistanceNN(0.5)
        )
        
        node_molecules, node_graphs, node_indices = get_subgraphs_as_molecules(sg0)
        for node_molecule, node_graph, node_idx in zip(node_molecules, node_graphs, node_indices):
            valid_node, to_add = is_valid_node(self, node_idx)
            if valid_node:
                nodes.append(Node.from_labled_molecule(node_molecule, node_graph, self.meta))
            if to_add: 
                self._linker_indices.update(to_add)
                for idx in to_add:
                    self._node_indices.remove(idx)
    
        sg1.remove_nodes(list(self.node_indices))
        linker_molecules, linker_graphs, linker_indices = get_subgraphs_as_molecules(sg1)
        for linker_molecule, linker_graph, linker_idx in zip(linker_molecules, linker_graphs, linker_indices):
            if is_valid_linker(linker_molecule):
                linkers.append(Linker.from_labled_molecule(linker_molecule, linker_graph, self.meta))
            else: 
                # Something went wrong, we should raise a warning 
                ...

        self.nodes = nodes
        self.linkers = linkers
        return linkers, nodes


    def fragment(self, fragmentation_method="all_node", filter_out_solvent: bool=True):
        self.fragmentation_method = fragmentation_method
        self.set_use_solvent_filter(filter_out_solvent)

        return self._fragment()

    def _get_cif_text(self):
        return write_cif(self.structure, self.structure_graph, [])

    def write_cif(self, filename):
        with open(filename, "w") as f:
            f.write(self._get_cif_text())

    def _get_node_indices(self):
        if self.fragmentation_method == 'all_node':
            result = fragment_all_node(self, self._filter_solvent)
        elif self.fragmentation_method == 'oxo':
            result = fragment_oxo_node(self, self._filter_solvent)

        self._node_indices = result['node_indices']
        self._solvent_connections = result['solvent_connections']
        self._solvent_indices = result['solvent_indices']
        self._connecting_node_indices = result['connecting_node_indices']
