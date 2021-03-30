# -*- coding: utf-8 -*-
from collections import namedtuple

from .splitter import get_floating_solvent_molecules

__all__ = ["FragmentationResult"]

FragmentationResult = namedtuple(
    "FragmentationResult",
    ["nodes", "linkers", "bound_solvent", "unbound_solvent", "net_embedding"],
)


def run_fragmentation(mof) -> FragmentationResult:
    unbound_solvent = get_floating_solvent_molecules(mof)
    # Find nodes

    # Find bound solvent

    # Filter the linkers (valid linkers have at least two branch points)

    # Now, get the net

    fragmentation_results = FragmentationResult()
