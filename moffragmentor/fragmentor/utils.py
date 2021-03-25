# -*- coding: utf-8 -*-
from typing import List, Union


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
        new_sublist = []
        for item in sublist:
            if item in metal_indices:
                if periodic_index_map is not None:
                    # adding the periodic images
                    new_sublist.extend(periodic_index_map[item])
                else:
                    new_sublist.append(item)
        output_list.append(new_sublist)
    return output_list
