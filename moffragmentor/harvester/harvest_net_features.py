# -*- coding: utf-8 -*-
"""Wrapper for minin a diretory with cgd files"""
import concurrent.futures
import logging
import os
import pickle
from glob import glob

from moffragmentor.descriptors.net import get_net_descriptors

logging.basicConfig(
    filename="harvester_log.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
LOGGER = logging.getLogger(__name__)


def net_description_w_log(file):
    print(f"Featurizing {file}")
    try:
        return get_net_descriptors(file)
    except Exception:  # pylint: disable=broad-except
        return None


def harvest_net_information(indir, outfile, max_workers=10):
    net_files = glob(os.path.join(indir, "*.cgd"))
    descriptors = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for desc in executor.map(net_description_w_log, net_files):
            if desc is not None:
                descriptors.append(desc)

    with open(outfile, "w") as handle:  # pylint:disable=unspecified-encoding
        pickle.dump(descriptors, handle)
