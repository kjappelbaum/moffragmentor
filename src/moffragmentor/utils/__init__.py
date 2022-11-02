# -*- coding: utf-8 -*-
"""Helper functions."""
import datetime
import json
import os
import pickle
from collections import defaultdict
from shutil import which
from typing import Collection, Dict, Iterable, List, Union

import networkx as nx
import numpy as np
import pymatgen
from backports.cached_property import cached_property
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.core import Molecule, Structure
from pymatgen.io.babel import BabelMolAdaptor
from skspatial.objects import Points


def get_molecule_mass(molecule):
    mass = 0
    for site in molecule:
        mass += site.specie.atomic_mass
    return mass


def unwrap(positions, lattice):
    celldiag = np.diagonal(lattice.matrix)
    dxyz = positions - positions[0]
    dxyz -= celldiag * np.around(dxyz / celldiag)
    dxyz += positions[0]

    return dxyz


def _get_metal_sublist(
    indices: List[int],
    metal_indices: List[int],
    periodic_index_map: Union[dict, None] = None,
) -> List[int]:
    metal_subset_in_node = []

    for node_index in indices:
        if node_index in metal_indices:
            if periodic_index_map is not None:
                # adding the periodic images
                metal_subset_in_node.extend(periodic_index_map[node_index])
            else:
                metal_subset_in_node.append(node_index)

    return metal_subset_in_node


def _get_metal_sublists(
    indices: List[List[int]],
    metal_indices: list,
    periodic_index_map: Union[dict, None] = None,
) -> List[List[int]]:
    """Recover the metal fragments from the nodes.

     We need it, for example, in the node filtering step where we analyze
     if the removal of a node creates new connected components.

    Args:
        indices (List[List[int]]): input indices, e.g., node indices
        metal_indices (list): indices of the metals in the structure
        periodic_index_map (dict): If not None, then we will also
            add the periodic images according to this map.
            Defaults to None.

    Returns:
        List[List[int]]: filtered input list,
            that now only contains indices of atoms that are metals
    """
    output_list = []
    for sublist in indices:
        new_sublist = _get_metal_sublist(sublist, metal_indices, periodic_index_map)
        output_list.append(new_sublist)
    return output_list


def revert_dict(dictionary):
    new_d = {}

    for key, value in dictionary.items():
        new_d[value] = key

    return new_d


def get_smiles_from_pmg_mol(pmg_mol):
    from openbabel import pybel as pb  # pylint:disable=import-outside-toplevel

    adaptor = BabelMolAdaptor(pmg_mol)
    pm = pb.Molecule(adaptor.openbabel_mol)
    smiles = pm.write("can").strip()
    return smiles


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


def write_cif(  # pylint:disable=too-many-locals
    s, graph, connection_indices, molecule=None, write_bonding_mode=False
):
    header_lines = [
        "data_cif",
        "_audit_creation_date              " + datetime.datetime.today().strftime("%Y-%m-%d"),
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
            loop_content.append(f" {ind} {es}  {vec[0]:.6f}  {vec[1]:.6f}  {vec[2]:.6f}  {0:.6f}")
        else:
            ind = es + str(i)
            loop_content.append(f" {ind} {es}  {vec[0]:.6f}  {vec[1]:.6f}  {vec[2]:.6f}  {0:.6f}")
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
    for i, _ in enumerate(s):
        neighbors = graph.get_connected_sites(i)
        for j, neighbor_site in enumerate(neighbors):
            ind0 = site_index[i]
            ind1 = site_index[neighbor_site.index]

            tuple_a = (ind0, ind1)
            tuple_b = (ind1, ind0)

            if not (tuple_a in set_bond) and not (
                tuple_b in set_bond
            ):  # pylint:disable=superfluous-parens
                set_bond.add(tuple_a)

                dist = np.round(s.get_distance(i, j), 3)
                if write_bonding_mode:
                    connection_loop_content.append(f"{ind0:7} {ind1:>7} {dist:>7} {'.':>3} {s:>3}")
                else:
                    connection_loop_content.append(f"{ind0:7} {ind1:>7} {'':>7} {'':>3}")

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


def is_tool(name: str) -> bool:
    """Check whether `name` is on PATH and marked as executable.

    https://stackoverflow.com/questions/11210104/check-if-a-program-exists-from-a-python-script

    Args:
        name (str): The name of the tool to check for.

    Returns:
        bool: True if the tool is on PATH and marked as executable.
    """
    return which(name) is not None


def _not_relevant_structure_indices(structure: Structure, indices: Collection[int]) -> List[int]:
    """Return the indices of the structure that are *not* in the indices collection.

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


def get_sub_structure(mof: "MOF", indices: Collection[int]) -> Structure:  # noqa: F821
    """Return a sub-structure of the structure with only the sites with the given indices.

    Args:
        mof (MOF): MOF object
        indices (Collection[int]): Collection of integers

    Returns:
        Structure: sub-structure of the structure with only the sites with the given indices
    """
    sites = []
    for index in indices:
        sites.append(mof.structure[index])

    s = Structure.from_sites(sites, to_unit_cell=True)
    return s


def visualize_part(mof, indices: Collection):
    import nglview  # pylint:disable=import-outside-toplevel

    sites = []
    for index in indices:
        sites.append(mof.structure[index])

    s = Structure.from_sites(sites, to_unit_cell=True)

    return nglview.show_pymatgen(s)


def joint_visualize_mols(mols: Iterable[Molecule]):
    import nglview as nv

    sites = []
    for mol in mols:
        sites.extend(mol.sites)
    super_mol = Molecule.from_sites(sites)
    return nv.show_pymatgen(super_mol)


def _flatten_list_of_sets(parts):
    flattened_parts = []

    for sublist in parts:
        for elm in sublist:
            flattened_parts.append(elm)
    return flattened_parts


def _get_image(mof, reference_index: int, index: int):
    _, image = mof.structure.lattice.get_distance_and_image(
        mof.frac_coords[reference_index], mof.frac_coords[index]
    )
    return image


def _get_cartesian_coords(mof, reference_index: int, index: int):
    image = _get_image(mof, reference_index, index)
    position = mof.frac_coords[index] + image
    return mof.lattice.get_cartesian_coords(position)


def _get_molecule_edge_label(u: int, v: int):
    return tuple(sorted((u, v)))


def _reindex_list_of_tuple(list_of_tuples, remapping: Dict[int, int]):
    new_list = []
    for tupl in list_of_tuples:
        new_tuple = (remapping[tupl[0]], remapping[tupl[1]])
        new_list.append(new_tuple)
    return new_list


def _metal_in_edge(site_collection, edge):
    return any(site_collection[edge_partner].specie.is_metal for edge_partner in edge)


def _get_vertices_of_smaller_component_upon_edge_break(graph, edge):
    graph_copy = graph.copy()
    graph_copy.remove_edge(edge[0], edge[1])
    connected_components = nx.connected_components(graph_copy)
    smallest_connected_component = min(connected_components, key=len)
    return smallest_connected_component


def get_edge_dict(structure_graph: StructureGraph) -> dict:
    def get_label(u, v):
        return sorted((u, v))

    types = defaultdict(list)
    for u, v, _ in structure_graph.graph.edges(data=True):
        label = get_label(u, v)
        types[tuple(label)] = None
    return dict(types)


def get_linker_connectivity(edge_dict):
    linker_connectivity = defaultdict(set)
    for node, linkers in edge_dict.items():
        for linker_tuple in linkers:
            linker_connectivity[linker_tuple[0]].add(node)
    num_neighbors = {}
    for linker, neighbors in linker_connectivity.items():
        num_neighbors[linker] = len(neighbors)

    return num_neighbors


def get_neighbors_from_nx_graph(graph, node_idx):
    return list(nx.neighbors(graph, node_idx))


def remove_all_nodes_not_in_indices(graph: nx.Graph, indices) -> nx.Graph:
    to_delete = [i for i in range(len(graph)) if i not in indices]
    graph.structure = Structure.from_sites(graph.structure.sites)
    graph.remove_nodes(to_delete)


class IStructure(pymatgen.core.structure.IStructure):
    """pymatgen IStructure with faster equality comparison.

    This dramatically speeds up lookups in the LRU cache when an object
    with the same __hash__ is already in the cache.
    """

    __hash__ = pymatgen.core.structure.IStructure.__hash__

    def __eq__(self, other):
        """Use specific, yet performant hash for equality comparison."""
        return self._dict_hash == other._dict_hash

    @cached_property
    def _dict_hash(self):
        """Specific, yet performant hash."""
        return hash(json.dumps(self.as_dict(), sort_keys=True))


def remove_site(structure: Union[Structure, IStructure]) -> None:
    if isinstance(structure, IStructure):
        structure = Structure.from_sites(structure.sites)


def add_suffix_to_dict_keys(dictionary: dict, suffix: str) -> dict:
    return {f"{k}_{suffix}": v for k, v in dictionary.items()}


def are_coplanar(mof, indices, tol=0.1):
    coords = mof.frac_coords[indices]
    points = Points(coords)
    return points.are_coplanar(tol=tol)
