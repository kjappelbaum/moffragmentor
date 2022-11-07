# -*- coding: utf-8 -*-
"""Generate molecules as the subgraphs from graphs"""
from collections import defaultdict
from typing import Iterable, Optional

import numpy as np
from loguru import logger
from pymatgen.core import Molecule, Site
from structuregraph_helpers.create import VestaCutoffDictNN

from .. import mof


def wrap_molecule(
    mol_idxs: Iterable[int],
    mof: "mof.MOF",
    starting_index: Optional[int] = None,
    add_additional_site: bool = True,  # noqa: F821
) -> Molecule:
    """Wrap a molecule in the cell of the MOF by walking along the structure graph.

    For this we perform BFS from the starting index. That is, we use a queue to
    keep track of the indices of the atoms that we still need to visit
    (the neighbors of the current index).
    We then compute new coordinates by computing the Cartesian coordinates
    of the neighbor image closest to the new coordinates of the current atom.

    To then create a Molecule with the correct ordering of sites, we walk
    through the hash table in the order of the original indices.

    Args:
        mol_idxs (Iterable[int]): The indices of the atoms in the molecule in the MOF.
        mof (MOF): MOF object that contains the mol_idxs.
        starting_index (int, optional): Starting index for the walk.
            Defaults to 0.
        add_additional_site (bool): Whether to add an additional site

    Returns:
        Molecule: wrapped molecule
    """
    if starting_index is None:
        if len(mol_idxs) == 1:
            starting_index = 0
        # take the index of the atom which coordinates are closest to the origin
        else:
            # Here was a bug before because i missed the zip
            starting_index = min(
                (zip(np.arange(len(mol_idxs)), mof.structure.cart_coords[mol_idxs])),
                key=lambda x: np.linalg.norm(x[1]),
            )[0]

    new_positions_cart = {}
    new_positions_frac = {}
    still_to_wrap_queue = [mol_idxs[starting_index]]
    new_positions_cart[mol_idxs[starting_index]] = mof.cart_coords[mol_idxs[starting_index]]
    new_positions_frac[mol_idxs[starting_index]] = mof.frac_coords[mol_idxs[starting_index]]
    additional_sites = []
    while still_to_wrap_queue:
        current_index = still_to_wrap_queue.pop(0)
        if current_index in mol_idxs:
            neighbor_indices = mof.get_neighbor_indices(current_index)
            for neighbor_index in neighbor_indices:
                if (neighbor_index not in new_positions_cart) & (neighbor_index in mol_idxs):
                    _, image = mof.structure[neighbor_index].distance_and_image_from_frac_coords(
                        new_positions_frac[current_index]
                    )
                    new_positions_frac[neighbor_index] = mof.frac_coords[neighbor_index] - image
                    new_positions_cart[neighbor_index] = mof.lattice.get_cartesian_coords(
                        new_positions_frac[neighbor_index]
                    )
                    still_to_wrap_queue.append(neighbor_index)
                else:
                    if neighbor_index in new_positions_cart:
                        species_a = str(mof.structure[current_index].specie)
                        species_b = str(mof.structure[neighbor_index].specie)

                        if (
                            np.linalg.norm(
                                new_positions_cart[neighbor_index]
                                - new_positions_cart[current_index]
                            )
                            > VestaCutoffDictNN._lookup_dict[species_a][species_b]
                        ) & add_additional_site:
                            logger.warning(
                                "Warning: neighbor_index {} is already in new_positions_cart, "
                                "but the distance is too large. "
                                "Will add an additional site. This is unusual and not well tested".format(
                                    neighbor_index
                                )
                            )

                            _, image = mof.structure[
                                neighbor_index
                            ].distance_and_image_from_frac_coords(new_positions_frac[current_index])
                            new_frac = mof.frac_coords[neighbor_index] - image
                            new_cart = mof.lattice.get_cartesian_coords(new_frac)
                            additional_sites.append((neighbor_index, new_cart, new_frac))

    new_sites = []
    for _, idx in enumerate(mol_idxs):
        new_sites.append(Site(mof.structure[idx].species, new_positions_cart[idx]))

    idx_mapping = defaultdict(list)
    for i, idx in enumerate(mol_idxs):
        idx_mapping[i].append(idx)
    for i, (idx, cart, _) in enumerate(additional_sites):
        new_sites.append(Site(mof.structure[idx].species, cart))
        idx_mapping[i + len(idx_mapping)].append(idx)

    return Molecule.from_sites(new_sites), idx_mapping
