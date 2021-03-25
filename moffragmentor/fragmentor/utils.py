# -*- coding: utf-8 -*-
from typing import List, Union


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
    """This helper function is useful to recover the metal fragments from the nodes. We need it, for example, in the node filtering step where we analyze if the removal of a node creates new connected components.

    Args:
        indices (List[List[int]]): input indices, e.g., node indices
        metal_indices (list): indices of the metals in the structure
        periodic_index_map (dict): If not None, then we will also add the periodic images according to this map.
            Defaults to None.
    Returns:
        List[List[int]]: filtered input list, that now only contains indices of atoms that are metals
    """
    output_list = []
    for sublist in indices:
        new_sublist = _get_metal_sublist(sublist, metal_indices, periodic_index_map)
        output_list.append(new_sublist)
    return output_list
