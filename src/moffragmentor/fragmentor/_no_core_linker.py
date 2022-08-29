from pymatgen.core import Structure
from structuregraph_helpers.subgraph import get_subgraphs_as_molecules

from moffragmentor.fragmentor.branching_points import has_metal_in_path
from moffragmentor.fragmentor.linkerlocator import _pick_central_linker_indices
from moffragmentor.fragmentor.molfromgraph import wrap_molecule
from moffragmentor.sbu.node import Node
from moffragmentor.sbu.nodecollection import NodeCollection


def generate_new_node_collection(mof, node_result):

    graph_ = mof.structure_graph.__copy__()
    graph_.structure = Structure.from_sites(graph_.structure.sites)
    graph_.remove_nodes(node_result.branching_indices)

    mols, graphs, idxs, centers, coordinates = get_subgraphs_as_molecules(
        graph_,
        return_unique=False,
        filter_in_cell=False,
        disable_boundary_crossing_check=True,
        prune_long_edges=True,
    )
    node_indices, coords = _pick_central_linker_indices(mof, coordinates)
    node_indices = [i for i in node_indices if has_metal_in_path(mof, idxs[i])]
    mols = [mol for i, mol in enumerate(mols) if i in node_indices]

    branching_sites_for_node = {}

    for i in node_indices:
        neighbor_pool = set()
        for vertex in idxs[i]:
            neighbor_pool.update(mof.get_neighbor_indices(vertex))

        branching_sites_for_node[i] = neighbor_pool & node_result.branching_indices

    nodes = []
    found_hashes = set()
    for i, node_index in enumerate(node_indices):
        idx = idxs[node_index]
        center = centers[node_index]
        coords_ = coords[i]
        branching_indices = list(branching_sites_for_node[node_index])
        mol, mapping = wrap_molecule(idx + branching_indices, mof)

        node = Node(
            molecule=mol,
            molecule_graph=graphs[node_index],
            center=center,
            graph_branching_indices=branching_indices,
            closest_branching_index_in_molecule=branching_indices,
            binding_indices=[],
            coordinates=coords_,
            original_indices=idx + list(branching_indices),
            connecting_paths=[],
            molecule_original_indices_mapping=mapping,
        )
        if node.hash not in found_hashes:
            found_hashes.add(node.hash)
            nodes.append(node)

    return NodeCollection(nodes)
