# -*- coding: utf-8 -*-
"""Helper functions"""
import datetime
import os
import pickle
from collections import defaultdict
from copy import deepcopy
from shutil import which
from typing import Collection, Dict, List, Tuple

import networkx as nx
import numpy as np
from openbabel import pybel as pb
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor


def revert_dict(d):
    new_d = {}

    for k, v in d.items():
        new_d[v] = k

    return new_d


def get_smiles_from_pmg_mol(pmg_mol):
    a = BabelMolAdaptor(pmg_mol)
    pm = pb.Molecule(a.openbabel_mol)
    smiles = pm.write("can").strip()
    return smiles


def build_mols_from_structure_graph(structure, structure_graph):
    site_pool = list(np.arange(len(structure_graph)))
    all_sites, all_edges = [], []

    new_pool, _, _ = explore_neighbors(0, site_pool, structure_graph)

    while len(new_pool) > 0:
        new_pool, sites, edges = explore_neighbors(
            new_pool[0], new_pool, structure_graph
        )
        all_sites.append(sites)
        all_edges.append(edges)

    molecules = []
    graphs = []
    for all_site, all_edge in zip(all_sites, all_edges):
        molecule, graph = build_molecule_and_graph(structure, all_site, all_edge)
        molecules.append(molecule)
        graphs.append(graph)

    return molecules, graphs


def explore_neighbors(index, site_pool, structure_graph):
    sites = set()
    edges = []

    def find_neigh(index):
        sites.add(index)
        connected_sites = structure_graph.get_connected_sites(index)
        indices = [s.index for s in connected_sites]
        new_sites = set(indices) - sites
        for new_site in new_sites:
            edges.append((index, new_site))
            sites.add(new_site)
        return new_sites

    new_sites = find_neigh(index)

    while len(new_sites) > 0:
        for site in new_sites:
            new_sites = find_neigh(site)

    for site in sites:
        site_pool.remove(site)

    return site_pool, sites, edges


def mol_from_sites(sites):
    coords = [n.coords for n in sites]
    species = [n.specie for n in sites]
    binding = [n.properties["binding"] for n in sites]

    molecule = Molecule(species, coords, site_properties={"binding": binding})
    return molecule.get_centered_molecule()


def build_molecule_and_graph(s, sites, edges):
    selected_sites = [s[i] for i in sites]

    remapper = dict(zip(sites, np.arange(len(sites))))
    new_edges = {}

    for a, b in edges:
        new_edges[(remapper[a], remapper[b])] = None

    mol = mol_from_sites(selected_sites)
    molecule_graph = MoleculeGraph.with_edges(mol, new_edges)
    return mol, molecule_graph


def write_cif(s, graph, connection_indices, molecule=None, write_bonding_mode=False):
    header_lines = [
        "data_cif",
        "_audit_creation_date              "
        + datetime.datetime.today().strftime("%Y-%m-%d"),
        "_audit_creation_method            'moffragmentor'",
        "_symmetry_space_group_name_H-M    'P 1'",
        "_symmetry_Int_Tables_number       1",
        "_symmetry_cell_setting            triclinic",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  x,y,z",
    ]

    cell_lines = [
        "_cell_length_a                    " + str(s.lattice.a),
        "_cell_length_b                    " + str(s.lattice.b),
        "_cell_length_c                    " + str(s.lattice.c),
        "_cell_angle_alpha                 " + str(s.lattice.alpha),
        "_cell_angle_beta                  " + str(s.lattice.beta),
        "_cell_angle_gamma                 " + str(s.lattice.gamma),
    ]

    loop_header = [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_charge",
    ]

    loop_content = []
    site_index = {}
    for i, site in enumerate(s):
        vec = site.frac_coords
        es = str(site.specie)

        if i in connection_indices:
            ind = "X" + str(i)
            loop_content.append(
                " {} {}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                    ind, es, vec[0], vec[1], vec[2], 0
                )
            )
        else:
            ind = es + str(i)
            loop_content.append(
                " {} {}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                    ind, es, vec[0], vec[1], vec[2], 0
                )
            )
        site_index[i] = ind

    connection_loop_header = [
        "loop_",
        "_geom_bond_atom_site_label_1",
        "_geom_bond_atom_site_label_2",
    ]

    if write_bonding_mode:

        connection_loop_header += ["_geom_bond_distance", "_ccdc_geom_bond_type"]

    connection_loop_content = []
    set_bond = set()
    if molecule:
        s = molecule
    for i, site in enumerate(s):
        neighbors = graph.get_connected_sites(i)
        for j, neighbor_site in enumerate(neighbors):
            ind0 = site_index[i]
            ind1 = site_index[neighbor_site.index]

            tuple_a = (ind0, ind1)
            tuple_b = (ind1, ind0)

            if not (tuple_a in set_bond) and not (tuple_b in set_bond):
                set_bond.add(tuple_a)

                dist = np.round(s.get_distance(i, j), 3)
                if write_bonding_mode:
                    connection_loop_content.append(
                        "{:7} {:>7} {:>7} {:>3} {:>3}".format(
                            ind0, ind1, dist, ".", "S"
                        )
                    )
                else:
                    connection_loop_content.append(
                        "{:7} {:>7} {:>7} {:>3}".format(ind0, ind1, "", "")
                    )

    return "\n".join(
        header_lines
        + cell_lines
        + loop_header
        + loop_content
        + connection_loop_header
        + connection_loop_content
    )


def pickle_dump(obj, path):
    with open(path, "wb") as handle:
        pickle.dump(obj, handle)


def make_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable.
    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script"""

    return which(name) is not None


def _not_relevant_structure_indices(
    structure: Structure, indices: Collection[int]
) -> List[int]:
    """Returns the indices of the structure that are *not* in the indices
    collection

    Args:
        structure (Structure): pymatgen Structure object
        indices (Collection[int]): Collection of integers

    Returns:
        List[int]: indices of structure that are not in the indices collection
    """
    not_relevant = []
    for i in range(len(structure)):
        if i not in indices:
            not_relevant.append(i)
    return not_relevant


def visualize_part(mof, indices: Collection):
    import nglview

    sites = []
    for index in indices:
        sites.append(mof.structure[index])

    s = Structure.from_sites(sites, to_unit_cell=True)

    return nglview.show_pymatgen(s)


def _flatten_list_of_sets(parts):
    flattened_parts = []

    for sublist in parts:
        for elm in sublist:
            flattened_parts.append(elm)
    return flattened_parts


def _is_in_cell(frac_coords):
    return all(frac_coords <= 1)


def _is_any_atom_in_cell(frac_coords):
    for row in frac_coords:
        if _is_in_cell(row):
            return True
    return False


def get_image(mof, reference_index: int, index: int):
    _, image = mof.structure.lattice.get_distance_and_image(
        mof.frac_coords[reference_index], mof.frac_coords[index]
    )
    return image


def get_cartesian_coords(mof, reference_index: int, index: int):
    image = get_image(mof, reference_index, index)
    position = mof.frac_coords[index] + image
    return mof.lattice.get_cartesian_coords(position)


def get_molecule_edge_label(u: int, v: int):
    return tuple(sorted((u, v)))


def reindex_list_of_tuple(list_of_tuples, remapping: Dict[int, int]):
    new_list = []
    for tupl in list_of_tuples:
        new_tuple = (remapping[tupl[0]], remapping[tupl[1]])
        new_list.append(new_tuple)
    return new_list


def metal_in_edge(site_collection, edge):
    for edge_partner in edge:
        if site_collection[edge_partner].specie.is_metal:
            return True
    return False


def get_vertices_of_smaller_component_upon_edge_break(graph, edge):
    graph_copy = deepcopy(graph)
    graph_copy.remove_edge(edge[0], edge[1])
    connected_components = nx.connected_components(graph_copy)
    smallest_connected_component = min(connected_components, key=len)
    return smallest_connected_component


def connected_mol_and_graph_from_indices(
    mof, indices: set
) -> Tuple[Molecule, List[int], List[int]]:
    """
    Given the indices that we identified to belong to the group
    1) Build a molecule from the structure, handling the periodic images
    2) Build the molecule graph from the structure graph, handling the periodic images
    3) Flag vertices in the graph that we call "hidden". Those have only one neighbor in the molecule but multiple in the structure.
    This happens, for example, in MOF-74 where the "connecting site" is not on the carboxy carbon given that one of the carboxy Os is not connected to the metal cluster
    To get reasonable nodes/linkers we will remove these from the node.
    We also flag indices that are connected via a bridge that is also a bridge in the original graph. This is, for example, the case for one carboxy O in MOF-74
    """
    new_positions = []
    persistent_non_metal_bridged_components = []
    hidden_vertices = []
    # we store edges as list of tuples as we can
    # do a cleaner loop for reindexing
    edges = []
    index_mapping = {}

    # cast set to list, so we can loop
    indices = list(indices)
    species = []
    for new_idx, idx in enumerate(indices):
        new_positions.append(get_cartesian_coords(mof, indices[0], idx))
        index_mapping[idx] = new_idx
        # This will fail for alloys/fractional occupancy
        # ToDo: check early if this might be an issue
        species.append(str(mof.structure[idx].specie))
        neighbors = mof.structure_graph.get_connected_sites(idx)
        # counter for the number of neighbors that are also part
        # of the molecule, i.e., in `indices`
        num_neighbors_in_molecule = 0
        for neighbor in neighbors:
            # we also identified the neighbor to be part of the molecule
            if neighbor.index in indices:
                edges.append(get_molecule_edge_label(idx, neighbor.index))
                num_neighbors_in_molecule += 1
        if (num_neighbors_in_molecule == 1) and len(neighbors) > 1:
            hidden_vertices.append(idx)

    reindexed_edges = reindex_list_of_tuple(edges, index_mapping)
    edge_dict = dict(zip(reindexed_edges, [None] * len(reindexed_edges)))

    mol = Molecule(species, new_positions)
    graph = MoleculeGraph.with_edges(mol, edge_dict)

    new_bridges = set(nx.bridges(nx.Graph(graph.graph.to_undirected())))
    new_index_to_old_index = revert_dict(index_mapping)
    new_bridges_w_old_indices = reindex_list_of_tuple(
        new_bridges, new_index_to_old_index
    )

    relevant_bridges = []

    for new_index_bridge, old_index_bridge in zip(
        new_bridges, new_bridges_w_old_indices
    ):
        if not metal_in_edge(mol, new_index_bridge):
            if mof._leads_to_terminal(old_index_bridge):
                relevant_bridges.append(new_index_bridge)

    for bridge in relevant_bridges:
        persistent_non_metal_bridged_components.append(
            get_vertices_of_smaller_component_upon_edge_break(
                graph.graph.to_undirected(), bridge
            )
        )

    return mol, graph, hidden_vertices, persistent_non_metal_bridged_components
