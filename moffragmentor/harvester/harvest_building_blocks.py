# -*- coding: utf-8 -*-
"""Functionality to fragment all MOFs in a folder of CIF files"""
import concurrent.futures
import logging
import os
from functools import partial
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import timeout_decorator

from ..descriptors.sbu_dimensionality import get_sbu_dimensionality
from ..mof import MOF
from ..utils import get_linker_connectivity, make_if_not_exists, remove_edge_duplicates

logging.basicConfig(
    format="[%(levelname)s]:%(lineno)s - %(message)s", level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def sbu_descriptors(
    sbu, mof, bb_type="linker", topology="", dimensionality=np.nan, connectivity=np.nan
):
    descriptors = sbu.descriptors
    descriptors["smiles"] = sbu.smiles
    descriptors["type"] = bb_type
    descriptors["topology"] = topology
    descriptors["mof_dimensionality"] = dimensionality
    descriptors["coordination"] = sbu.coordination
    descriptors["connectivity"] = connectivity
    descriptors["dimensionality"] = get_sbu_dimensionality(
        mof, sbu._original_indices
    )  # pylint:disable=protected-access
    return {**descriptors}


MAX_ATOMS = 3000


class Harvester:
    def __init__(self, mof, outdir=None):
        self.mof = mof
        self.outdir = outdir

    @classmethod
    def from_cif(cls, cif, outdir=None):
        mof = MOF.from_cif(cif)
        if len(mof.structure) > MAX_ATOMS:
            raise ValueError("Structure too large")
        return cls(mof, outdir)

    def run_harvest(self):
        if self.outdir is not None:
            self.mof.dump(os.path.join(self.outdir, "mof.pkl"))
        descriptors = []

        parts = self.mof.fragment()
        topology = parts.net_embedding.rcsr_code
        dimensionality = self.mof.dimensionality
        edge_dict = parts.net_embedding.edge_dict
        linker_connectivity = get_linker_connectivity(edge_dict)
        # descriptors.append({"dimensionality", self.mof.dimensionality})
        for i, linker in enumerate(parts.linkers):
            if self.outdir is not None:
                linker.dump(os.path.join(self.outdir, f"linker_{i}.pkl"))
            descriptors.append(
                sbu_descriptors(
                    linker,
                    self.mof,
                    bb_type="linker",
                    topology=topology,
                    dimensionality=dimensionality,
                    connectivity=linker_connectivity[i],
                )
            )
        for i, node in enumerate(parts.nodes):
            if self.outdir is not None:
                node.dump(os.path.join(self.outdir, f"node_{i}.pkl"))
            descriptors.append(
                sbu_descriptors(
                    node,
                    self.mof,
                    bb_type="node",
                    topology=topology,
                    dimensionality=dimensionality,
                    connectivity=len(remove_edge_duplicates(edge_dict[i])),
                )
            )

        df = pd.DataFrame(descriptors)
        if self.outdir is not None:
            df.to_csv(os.path.join(self.outdir, "descriptors.csv"), index=False)
        return df


def harvest_w_timeout(cif, dumpdir=None):
    try:
        return harvest_cif(cif, dumpdir=dumpdir)
    except Exception as e:
        LOGGER.exception(f"Exception occured for {cif}. Exception: {e}.")
        return None


@timeout_decorator.timeout(200)
def harvest_cif(cif, dumpdir=None):
    try:
        stem = Path(cif).stem
        if dumpdir is not None:
            path = os.path.join(dumpdir, stem)
            make_if_not_exists(path)
            dumpdir = path
        harvester = Harvester.from_cif(cif, dumpdir)
        df = harvester.run_harvest()
        return df
    except Exception as e:
        LOGGER.exception(f"Exception occured for {cif}. Exception: {e}.")
        return None


def harvest_directory(
    directory, njobs=1, outdir=None, skip_existing=True, reverse=False, offset: int = 0
):
    all_cifs = glob(os.path.join(directory, "*.cif"))
    if outdir is not None:
        make_if_not_exists(outdir)

    harvest_partial = partial(harvest_w_timeout, dumpdir=outdir)

    if skip_existing:
        existing_stems = [
            Path(p).parents[0].split("_")[0]
            for p in glob(os.path.join(outdir, "*", "descriptors.csv"))
        ]
        filtered_all_cifs = []
        for cif in all_cifs:
            if Path(cif).stem.split("_")[0] not in existing_stems:
                filtered_all_cifs.append(cif)
    else:
        filtered_all_cifs = all_cifs

    filtered_all_cifs = sorted(filtered_all_cifs, reverse=reverse)
    all_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as exec:
        for i, res in enumerate(exec.map(harvest_partial, filtered_all_cifs[offset:])):
            try:
                if res is not None:
                    all_res.append(res)
                else:
                    LOGGER.error(f"Failed for {filtered_all_cifs[i]}")
            except concurrent.futures.process.BrokenProcessPool as ex:
                LOGGER.error("Broken process pool")

    df = pd.concat(all_res)
    if outdir is None:
        df.to_csv("harvest_results.csv", index=False)

    else:
        df.to_csv(os.path.join(outdir, "harvest_results.csv"), index=False)
