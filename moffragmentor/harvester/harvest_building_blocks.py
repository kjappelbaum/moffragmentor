# -*- coding: utf-8 -*-
"""Functionality to fragment all MOFs in a folder of CIF files"""
import concurrent.futures
import logging
import os
from functools import partial
from glob import glob
from pathlib import Path

import pandas as pd

from ..mof import MOF
from ..utils import make_if_not_exists

logging.basicConfig(
    format="[%(levelname)s]:%(lineno)s - %(message)s", level=logging.INFO
)
LOGGER = logging.getLogger(__name__)


def sbu_descriptors(sbu, type="linker", topology=""):
    descriptors = sbu.get_descriptors()
    descriptors["smiles"] = sbu.smiles
    descriptors["type"] = type
    descriptors["topology"] = topology

    return {**descriptors, **sbu.meta}


class Harvester:
    def __init__(self, mof, outdir=None):
        self.mof = mof
        self.outdir = outdir

    @classmethod
    def from_cif(cls, cif, outdir=None):
        mof = MOF.from_cif(cif)
        return cls(mof, outdir)

    def run_harvest(self):
        if self.outdir is not None:
            self.mof.dump(os.path.join(self.outdir, "mof.pkl"))
        descriptors = []
        topology = self.mof.topology
        linkers, nodes = self.mof.fragment()
        for i, linker in enumerate(linkers):
            if self.outdir is not None:
                linker.dump(os.path.join(self.outdir, f"linker_{i}.pkl"))
            descriptors.append(
                sbu_descriptors(linker, type="linker", topology=topology)
            )
        for i, node in enumerate(nodes):
            if self.outdir is not None:
                node.dump(os.path.join(self.outdir, f"node_{i}.pkl"))
            descriptors.append(sbu_descriptors(node, type="node", topology=topology))

        df = pd.DataFrame(descriptors)
        if self.outdir is not None:
            df.to_csv(os.path.join(self.outdir, "descriptors.csv"), index=False)
        return df


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
        LOGGER.warning(f"Exception occured for {cif}. Exception: {e}.")
        return None


def harvest_directory(directory, njobs=1, outdir=None):
    all_cifs = glob(os.path.join(directory, "*.cif"))
    if outdir is not None:
        make_if_not_exists(outdir)

    harvest_partial = partial(harvest_cif, dumpdir=outdir)

    all_res = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=njobs) as exec:
        for i, res in enumerate(exec.map(harvest_partial, all_cifs)):
            if res is not None:
                all_res.append(res)
            else:
                LOGGER.error(f"Failed for {all_cifs[i]}")

    df = pd.concat(all_res)
    if outdir is None:
        df.to_csv("harvest_results.csv", index=False)

    else:
        df.to_csv(os.path.join(outdir, "harvest_results.csv"), index=False)
