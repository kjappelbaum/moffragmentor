# -*- coding: utf-8 -*-
"""Setup tests"""
import os

import networkx as nx
import numpy as np
import pytest
from pymatgen.analysis.graphs import MoleculeGraph, StructureGraph
from pymatgen.analysis.local_env import JmolNN
from pymatgen.core import Molecule, Structure

from moffragmentor import MOF
from moffragmentor.sbu import Linker, Node

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

HKUST_cdg = """CRYSTAL
   NAME
   GROUP P1
   CELL 26.343 26.343 26.343 90.0 90.0 90.0
   NODE 0 3 0.3425 0.1576 0.8425
   NODE 1 3 0.8425 0.3425 0.8425
   NODE 2 3 0.6576 0.8425 0.8425
   NODE 3 3 0.1576 0.6576 0.8425
   NODE 4 3 0.3425 0.8425 0.1576
   NODE 5 3 0.1576 0.3425 0.1576
   NODE 6 3 0.6576 0.1576 0.1576
   NODE 7 3 0.8425 0.6576 0.1576
   NODE 8 3 0.8425 0.3425 0.1576
   NODE 9 3 0.6576 0.8425 0.1576
   NODE 10 3 0.1576 0.6576 0.1576
   NODE 11 3 0.3425 0.1576 0.1576
   NODE 12 3 0.8425 0.6576 0.8425
   NODE 13 3 0.3425 0.8425 0.8425
   NODE 14 3 0.1576 0.3425 0.8425
   NODE 15 3 0.6576 0.1576 0.8425
   NODE 16 3 0.1576 0.8425 0.3425
   NODE 17 3 0.1576 0.1576 0.6576
   NODE 18 3 0.8425 0.1576 0.6576
   NODE 19 3 0.8425 0.8425 0.6576
   NODE 20 3 0.1576 0.8425 0.6576
   NODE 21 3 0.8425 0.1576 0.3425
   NODE 22 3 0.8425 0.8425 0.3425
   NODE 23 3 0.1576 0.1576 0.3425
   NODE 24 3 0.3425 0.6576 0.3425
   NODE 25 3 0.6576 0.3425 0.3425
   NODE 26 3 0.3425 0.3425 0.6576
   NODE 27 3 0.6576 0.6576 0.6576
   NODE 28 3 0.6576 0.3425 0.6576
   NODE 29 3 0.3425 0.6576 0.6576
   NODE 30 3 0.3425 0.3425 0.3425
   NODE 31 3 0.6576 0.6576 0.3425
   NODE 32 4 0.2500 0.2500 -0.0000
   NODE 33 4 0.7500 0.2500 0.0000
   NODE 34 4 0.7500 0.7500 0.0000
   NODE 35 4 0.2500 0.7500 0.0000
   NODE 36 4 -0.0000 0.2500 0.2500
   NODE 37 4 0.7500 -0.0000 0.2500
   NODE 38 4 -0.0000 0.7500 0.2500
   NODE 39 4 0.2500 -0.0000 0.2500
   NODE 40 4 -0.0000 0.7500 0.7500
   NODE 41 4 0.2500 -0.0000 0.7500
   NODE 42 4 -0.0000 0.2500 0.7500
   NODE 43 4 0.7500 -0.0000 0.7500
   NODE 44 4 0.2500 0.7500 0.5000
   NODE 45 4 0.7500 0.7500 0.5000
   NODE 46 4 0.7500 0.2500 0.5000
   NODE 47 4 0.2500 0.2500 0.5000
   NODE 48 4 0.7500 0.5000 0.7500
   NODE 49 4 0.2500 0.5000 0.7500
   NODE 50 4 0.2500 0.5000 0.2500
   NODE 51 4 0.7500 0.5000 0.2500
   NODE 52 4 0.5000 0.2500 0.7500
   NODE 53 4 0.5000 0.7500 0.7500
   NODE 54 4 0.5000 0.7500 0.2500
   NODE 55 4 0.5000 0.2500 0.2500
   EDGE   0.1576 0.8425 0.3425 -0.0000 0.7500 0.2500
   EDGE   0.3425 0.3425 0.3425 0.5000 0.2500 0.2500
   EDGE   0.8425 0.1576 0.3425 0.7500 -0.0000 0.2500
   EDGE   0.8425 0.6576 0.8425 0.7500 0.7500 1.0000
   EDGE   0.8425 0.1576 0.3425 0.7500 0.2500 0.5000
   EDGE   0.1576 0.6576 0.8425 -0.0000 0.7500 0.7500
   EDGE   0.3425 0.8425 0.1576 0.2500 1.0000 0.2500
   EDGE   0.1576 0.6576 0.8425 0.2500 0.5000 0.7500
   EDGE   0.1576 0.3425 0.8425 0.2500 0.5000 0.7500
   EDGE   0.8425 0.3425 0.8425 0.7500 0.2500 1.0000
   EDGE   0.3425 0.8425 0.8425 0.2500 0.7500 1.0000
   EDGE   0.8425 0.3425 0.8425 1.0000 0.2500 0.7500
   EDGE   0.3425 0.6576 0.3425 0.2500 0.7500 0.5000
   EDGE   0.1576 0.8425 0.6576 -0.0000 0.7500 0.7500
   EDGE   0.8425 0.8425 0.3425 0.7500 1.0000 0.2500
   EDGE   0.1576 0.1576 0.3425 -0.0000 0.2500 0.2500
   EDGE   0.1576 0.3425 0.8425 -0.0000 0.2500 0.7500
   EDGE   0.6576 0.6576 0.6576 0.7500 0.7500 0.5000
   EDGE   0.1576 0.6576 0.1576 -0.0000 0.7500 0.2500
   EDGE   0.6576 0.8425 0.8425 0.7500 0.7500 1.0000
   EDGE   0.3425 0.3425 0.3425 0.2500 0.5000 0.2500
   EDGE   0.6576 0.8425 0.8425 0.7500 1.0000 0.7500
   EDGE   0.3425 0.1576 0.1576 0.5000 0.2500 0.2500
   EDGE   0.1576 0.6576 0.8425 0.2500 0.7500 1.0000
   EDGE   0.1576 0.3425 0.1576 0.2500 0.2500 -0.0000
   EDGE   0.8425 0.6576 0.1576 0.7500 0.5000 0.2500
   EDGE   0.6576 0.6576 0.3425 0.7500 0.5000 0.2500
   EDGE   0.1576 0.1576 0.3425 0.2500 0.2500 0.5000
   EDGE   0.6576 0.3425 0.6576 0.7500 0.2500 0.5000
   EDGE   0.8425 0.8425 0.6576 0.7500 1.0000 0.7500
   EDGE   0.3425 0.1576 0.1576 0.2500 -0.0000 0.2500
   EDGE   0.1576 0.8425 0.6576 0.2500 0.7500 0.5000
   EDGE   0.8425 0.6576 0.8425 1.0000 0.7500 0.7500
   EDGE   0.3425 0.8425 0.1576 0.5000 0.7500 0.2500
   EDGE   0.8425 0.3425 0.1576 1.0000 0.2500 0.2500
   EDGE   0.3425 0.1576 0.1576 0.2500 0.2500 -0.0000
   EDGE   0.3425 0.1576 0.8425 0.2500 0.2500 1.0000
   EDGE   0.8425 0.8425 0.6576 0.7500 0.7500 0.5000
   EDGE   0.6576 0.3425 0.6576 0.7500 0.5000 0.7500
   EDGE   0.3425 0.1576 0.8425 0.2500 -0.0000 0.7500
   EDGE   0.8425 0.1576 0.3425 1.0000 0.2500 0.2500
   EDGE   0.3425 0.8425 0.8425 0.5000 0.7500 0.7500
   EDGE   0.3425 0.6576 0.6576 0.2500 0.5000 0.7500
   EDGE   0.8425 0.1576 0.6576 0.7500 0.2500 0.5000
   EDGE   0.1576 0.1576 0.6576 0.2500 -0.0000 0.7500
   EDGE   0.1576 0.6576 0.1576 0.2500 0.7500 0.0000
   EDGE   0.6576 0.8425 0.1576 0.7500 1.0000 0.2500
   EDGE   0.3425 0.3425 0.3425 0.2500 0.2500 0.5000
   EDGE   0.3425 0.1576 0.8425 0.5000 0.2500 0.7500
   EDGE   0.6576 0.1576 0.8425 0.7500 -0.0000 0.7500
   EDGE   0.6576 0.1576 0.8425 0.5000 0.2500 0.7500
   EDGE   0.1576 0.3425 0.8425 0.2500 0.2500 1.0000
   EDGE   0.3425 0.3425 0.6576 0.5000 0.2500 0.7500
   EDGE   0.8425 0.8425 0.6576 1.0000 0.7500 0.7500
   EDGE   0.6576 0.6576 0.6576 0.5000 0.7500 0.7500
   EDGE   0.1576 0.3425 0.1576 0.2500 0.5000 0.2500
   EDGE   0.6576 0.3425 0.6576 0.5000 0.2500 0.7500
   EDGE   0.6576 0.1576 0.1576 0.7500 0.2500 0.0000
   EDGE   0.3425 0.6576 0.6576 0.2500 0.7500 0.5000
   EDGE   0.1576 0.8425 0.3425 0.2500 0.7500 0.5000
   EDGE   0.8425 0.3425 0.1576 0.7500 0.2500 0.0000
   EDGE   0.3425 0.8425 0.8425 0.2500 1.0000 0.7500
   EDGE   0.8425 0.6576 0.1576 0.7500 0.7500 0.0000
   EDGE   0.8425 0.3425 0.8425 0.7500 0.5000 0.7500
   EDGE   0.3425 0.3425 0.6576 0.2500 0.2500 0.5000
   EDGE   0.3425 0.6576 0.3425 0.2500 0.5000 0.2500
   EDGE   0.8425 0.1576 0.6576 0.7500 -0.0000 0.7500
   EDGE   0.3425 0.8425 0.1576 0.2500 0.7500 0.0000
   EDGE   0.1576 0.3425 0.1576 -0.0000 0.2500 0.2500
   EDGE   0.6576 0.8425 0.1576 0.7500 0.7500 0.0000
   EDGE   0.1576 0.1576 0.6576 0.2500 0.2500 0.5000
   EDGE   0.6576 0.1576 0.1576 0.7500 -0.0000 0.2500
   EDGE   0.1576 0.8425 0.3425 0.2500 1.0000 0.2500
   EDGE   0.3425 0.3425 0.6576 0.2500 0.5000 0.7500
   EDGE   0.6576 0.3425 0.3425 0.7500 0.5000 0.2500
   EDGE   0.8425 0.8425 0.3425 0.7500 0.7500 0.5000
   EDGE   0.6576 0.8425 0.1576 0.5000 0.7500 0.2500
   EDGE   0.6576 0.1576 0.8425 0.7500 0.2500 1.0000
   EDGE   0.8425 0.6576 0.1576 1.0000 0.7500 0.2500
   EDGE   0.3425 0.6576 0.3425 0.5000 0.7500 0.2500
   EDGE   0.1576 0.8425 0.6576 0.2500 1.0000 0.7500
   EDGE   0.8425 0.8425 0.3425 1.0000 0.7500 0.2500
   EDGE   0.3425 0.6576 0.6576 0.5000 0.7500 0.7500
   EDGE   0.1576 0.1576 0.6576 -0.0000 0.2500 0.7500
   EDGE   0.8425 0.3425 0.1576 0.7500 0.5000 0.2500
   EDGE   0.6576 0.3425 0.3425 0.7500 0.2500 0.5000
   EDGE   0.6576 0.8425 0.8425 0.5000 0.7500 0.7500
   EDGE   0.6576 0.3425 0.3425 0.5000 0.2500 0.2500
   EDGE   0.1576 0.1576 0.3425 0.2500 -0.0000 0.2500
   EDGE   0.8425 0.6576 0.8425 0.7500 0.5000 0.7500
   EDGE   0.6576 0.6576 0.6576 0.7500 0.5000 0.7500
   EDGE   0.1576 0.6576 0.1576 0.2500 0.5000 0.2500
   EDGE   0.8425 0.1576 0.6576 1.0000 0.2500 0.7500
   EDGE   0.6576 0.6576 0.3425 0.7500 0.7500 0.5000
   EDGE   0.6576 0.1576 0.1576 0.5000 0.2500 0.2500
   EDGE   0.6576 0.6576 0.3425 0.5000 0.7500 0.2500
END"""


SYSTRE_OUT = """Structure #1 - .

   Input structure described as 3-periodic.
   Given space group is P1.
   56 nodes and 96 edges in repeat unit as given.

   Ideal repeat unit smaller than given (24 vs 96 edges).
   Point group has 48 elements.
   2 kinds of node.

   Equivalences for non-unique nodes:
      1 --> 0
      2 --> 0
      3 --> 0
      4 --> 0
      5 --> 0
      6 --> 0
      7 --> 0
      8 --> 0
      9 --> 0
      10 --> 0
      11 --> 0
      12 --> 0
      13 --> 0
      14 --> 0
      15 --> 0
      16 --> 0
      17 --> 0
      18 --> 0
      19 --> 0
      20 --> 0
      21 --> 0
      22 --> 0
      23 --> 0
      24 --> 0
      25 --> 0
      26 --> 0
      27 --> 0
      28 --> 0
      29 --> 0
      30 --> 0
      31 --> 0
      33 --> 32
      34 --> 32
      35 --> 32
      36 --> 32
      37 --> 32
      38 --> 32
      39 --> 32
      40 --> 32
      41 --> 32
      42 --> 32
      43 --> 32
      44 --> 32
      45 --> 32
      46 --> 32
      47 --> 32
      48 --> 32
      49 --> 32
      50 --> 32
      51 --> 32
      52 --> 32
      53 --> 32
      54 --> 32
      55 --> 32

   Coordination sequences:
      Node 0:    3 9 15 33 45 82 90 153 150 241
      Node 32:    4 8 20 30 60 68 120 126 200 180

   TD10 = 820

   Ideal space group is Fm-3m.
   Ideal group or setting differs from given (Fm-3m vs P1).

   Structure was identified with RCSR symbol:
       Name:            tbo


   Relaxed cell parameters:
       a = 4.89898, b = 4.89898, c = 4.89898
       alpha = 90.0000, beta = 90.0000, gamma = 90.0000
   Cell volume: 117.57551
   Relaxed positions:
      Node 0:    0.16667 0.16667 0.16667
      Node 32:    0.00000 0.25000 0.25000
   Edges:
      0.00000 0.25000 0.25000  <->  0.16667 0.16667 0.16667
   Edge centers:
      0.08333 0.20833 0.20833

   Edge statistics: minimum = 1.00000, maximum = 1.00000, average = 1.00000
   Angle statistics: minimum = 70.52878, maximum = 180.00000, average = 120.00000
   Shortest non-bonded distance = 1.15470

   Degrees of freedom: 2

Finished structure #1 - ."""

SYSTRE_OUT_2 = """Data file "/Users/kevinmaikjablonka/Downloads/test_systre.cgd".

Structure #1.

   Input structure described as 3-periodic.
   Given space group is P1.
   14 nodes and 24 edges in repeat unit as given.

   Structure is not connected.
   Processing components separately.

   ==========
   Processing component 1:
      dimension = 2
   Input structure described as 2-periodic.
   Given space group is P1.
   14 nodes and 24 edges in repeat unit as given.

   Ideal repeat unit smaller than given (12 vs 24 edges).
   Point group has 8 elements.
   3 kinds of node.

   Equivalences for non-unique nodes:
      V3 --> V2
      V5 --> V1
      V6 --> V1
      V7 --> V1
      V8 --> V1
      V9 --> V1
      V10 --> V1
      V11 --> V4
      V12 --> V2
      V13 --> V2
      V14 --> V1

   Coordination sequences:
      Node V1:    3 7 8 15 15 24 21 32 27 40
      Node V2:    4 6 12 12 20 18 28 24 36 30
      Node V4:    4 4 8 12 20 16 28 28 36 24

   TD10 = 191

   Ideal space group is p4mm.
   Ideal group or setting differs from given (p4mm vs P1).

   Structure was identified with RCSR symbol:
       Name:        mtf


   Coordinates are for a full conventional cell.
   Relaxed cell parameters:
       a = 3.05792, b = 3.05792, gamma = 90.0000
   Barycentric positions:
      Node 1:    0.33333 0.33333
      Node 2:    0.33333 0.66667
      Node 3:    0.66667 0.33333
      Node 4:    0.66667 0.66667
      Node 5:    0.00000 0.50000
      Node 6:    0.50000 0.00000
      Node 7:    0.50000 0.50000
   Edges:
      0.33333 0.33333  <->  0.50000 0.50000
      0.50000 0.50000  <->  0.33333 0.33333
      0.50000 0.50000  <->  0.33333 0.66667
      0.50000 0.50000  <->  0.66667 0.33333
      0.50000 0.50000  <->  0.66667 0.66667
      0.33333 0.66667  <->  0.50000 0.50000
      0.66667 0.33333  <->  0.50000 0.50000
      0.66667 0.66667  <->  0.50000 0.50000
      0.00000 0.50000  <->  0.33333 0.33333
      0.00000 0.50000  <->  0.33333 0.66667
      0.00000 0.50000  <->  -0.33333 0.33333
      0.00000 0.50000  <->  -0.33333 0.66667
      0.50000 0.00000  <->  0.33333 0.33333
      0.50000 0.00000  <->  0.33333 -0.33333
      0.50000 0.00000  <->  0.66667 0.33333
      0.50000 0.00000  <->  0.66667 -0.33333
      0.33333 0.33333  <->  0.00000 0.50000
      0.33333 0.33333  <->  0.50000 0.00000
      0.33333 0.66667  <->  0.00000 0.50000
      0.33333 0.66667  <->  0.50000 1.00000
      0.66667 0.33333  <->  0.50000 0.00000
      0.66667 0.33333  <->  1.00000 0.50000
      0.66667 0.66667  <->  0.50000 1.00000
      0.66667 0.66667  <->  1.00000 0.50000
   Edge centers:
      0.41667 0.41667
      0.41667 0.41667
      0.41667 0.58333
      0.58333 0.41667
      0.58333 0.58333
      0.41667 0.58333
      0.58333 0.41667
      0.58333 0.58333
      0.16667 0.41667
      0.16667 0.58333
      -0.16667 0.41667
      -0.16667 0.58333
      0.41667 0.16667
      0.41667 -0.16667
      0.58333 0.16667
      0.58333 -0.16667
      0.16667 0.41667
      0.41667 0.16667
      0.16667 0.58333
      0.41667 0.83333
      0.58333 0.16667
      0.83333 0.41667
      0.58333 0.83333
      0.83333 0.58333

   Edge statistics: minimum = 0.72076, maximum = 1.13962, average = 1.00000
   Angle statistics: minimum = 53.13010, maximum = 180.00000, average = 120.00000
   Shortest non-bonded distance = 1.01931

   Degrees of freedom: 2

   Finished component 1.

   ==========

Finished structure #1.

Finished data file "/Users/kevinmaikjablonka/Downloads/test_systre.cgd".
"""


@pytest.fixture(scope="session")
def get_agn_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "RSM3317.cif"))
    return mof


@pytest.fixture(scope="session")
def get_cuiiibtc_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    return mof


@pytest.fixture(scope="session")
def get_hkust_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    return mof


@pytest.fixture(scope="session")
def get_mof5():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "mof-5_cellopt.cif"))
    return mof


@pytest.fixture(scope="session")
def get_single_metal_mof():
    # PIQMUZ_clean
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "RSM3434.cif"))
    return mof


@pytest.fixture(scope="session")
def get_porphryin_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "porphyrin_mof.cif"))
    return mof


@pytest.fixture(scope="session")
def get_hypothetical_mof():
    mof = MOF.from_cif(
        os.path.join(THIS_DIR, "test_files", "bcs_v1-litcic_1B_4H_Ch_opt_charge.cif")
    )
    return mof


@pytest.fixture(scope="session")
def get_p_linker_with_floating():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "MAGBON.cif"))
    return mof


@pytest.fixture(scope="session")
def get_li_mof_with_floating():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "LISZOE.cif"))
    return mof


@pytest.fixture(scope="session")
def abaxin():
    # https://github.com/kjappelbaum/moffragmentor/issues/70
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "ABAXIN.cif"))
    return mof


@pytest.fixture(scope="session")
def get_1d_node_with_floating():
    """https://pubs.rsc.org/en/content/articlelanding/2014/cc/c3cc49684h#!divAbstract

    Found with Google search 'complicated node mof.

    Returns:
        MOF: 1D node with floating atoms
    """
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "c3cc49684h3.cif"))
    return mof


@pytest.fixture(scope="session")
def get_single_metal_mof_previously_failing():
    """Issue #119"""
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "BAMHIJ.cif"))
    return mof


@pytest.fixture(scope="session")
def get_1d_node_graph():
    not_node = [
        1,
        3,
        5,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        49,
        51,
        53,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        69,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        93,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        105,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        117,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        129,
        131,
        133,
        135,
        137,
        139,
        141,
        143,
        145,
        147,
        149,
        151,
        153,
        155,
        157,
        159,
        160,
        161,
        162,
        163,
        165,
        167,
        168,
        169,
        170,
        171,
    ]
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "c3cc49684h3.cif"))
    graph = mof.structure_graph
    graph.structure = Structure.from_sites(graph.structure.sites)
    graph.remove_nodes(not_node)
    return graph


@pytest.fixture(scope="session")
def get_0d_node_graph():
    not_node = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        108,
        109,
        110,
        111,
        112,
        113,
        114,
        115,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        146,
        147,
        148,
        149,
        150,
        151,
        152,
        153,
        154,
        155,
        156,
        157,
        158,
        159,
        160,
        161,
        162,
        163,
        164,
        165,
        166,
        167,
        168,
        169,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        178,
        179,
        180,
        181,
        182,
        183,
        184,
        185,
        186,
        187,
        188,
        189,
        190,
        191,
        192,
        193,
        194,
        195,
        196,
        197,
        198,
        199,
        200,
        201,
        202,
        203,
        204,
        205,
        206,
        207,
        208,
        209,
        210,
        211,
        212,
        213,
        214,
        215,
        216,
        217,
        218,
        219,
        220,
        221,
        222,
        223,
        224,
        225,
        226,
        227,
        228,
        229,
        230,
        231,
        232,
        233,
        234,
        235,
        236,
        237,
        238,
        239,
        240,
        241,
        242,
        243,
        244,
        245,
        247,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
        256,
        257,
        258,
        259,
        260,
        261,
        262,
        263,
        264,
        265,
        266,
        268,
        269,
        270,
        271,
        273,
        274,
        275,
        276,
        277,
        278,
        279,
        280,
        281,
        282,
        283,
        284,
        285,
        286,
        287,
        288,
        289,
        290,
        291,
        292,
        293,
        294,
        295,
        296,
        297,
        298,
        299,
        300,
        301,
        302,
        303,
        304,
        305,
        306,
        307,
        308,
        309,
        310,
        311,
        312,
        313,
        314,
        315,
        316,
        317,
        318,
        319,
        320,
        321,
        322,
        323,
        324,
        325,
        326,
        327,
        328,
        329,
        330,
        331,
        332,
        333,
        334,
        335,
        336,
        337,
        338,
        339,
        340,
        341,
        342,
        343,
        344,
        345,
        346,
        347,
        348,
        349,
        350,
        351,
        352,
        353,
        354,
        355,
        356,
        357,
        358,
        359,
        360,
        361,
        362,
        363,
        364,
        365,
        366,
        367,
        368,
        369,
        370,
        371,
        372,
        373,
        374,
        375,
        376,
        377,
        378,
        379,
        380,
        381,
        382,
        383,
        384,
        385,
        386,
        387,
        388,
        389,
        390,
        391,
        392,
        393,
        394,
        395,
        396,
        397,
        398,
        399,
        400,
        401,
        402,
        403,
        404,
        405,
        406,
        407,
        408,
        409,
        410,
        411,
        412,
        413,
        414,
        415,
        416,
        417,
        418,
        419,
        420,
        421,
        422,
        423,
        424,
        425,
        426,
        427,
        428,
        429,
        430,
        431,
        433,
        434,
        435,
        436,
        438,
        439,
        440,
        441,
        442,
        443,
        444,
        445,
        446,
        447,
        448,
        449,
        450,
        451,
        452,
        453,
        454,
        455,
        456,
        457,
        459,
        460,
        461,
        462,
        464,
        465,
        466,
        467,
        468,
        469,
        470,
        471,
        472,
        473,
        474,
        475,
        476,
        477,
        478,
        479,
        480,
        481,
        482,
        483,
        484,
        485,
        486,
        487,
        488,
        489,
        490,
        491,
        492,
        493,
        494,
        495,
        496,
        497,
        498,
        499,
        500,
        501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        531,
        532,
        533,
        534,
        535,
        536,
        537,
        538,
        539,
        540,
        541,
        542,
        543,
        544,
        545,
        546,
        547,
        548,
        549,
        550,
        551,
        552,
        553,
        554,
        555,
        556,
        557,
        558,
        559,
        560,
        561,
        562,
        563,
        564,
        565,
        566,
        567,
        568,
        569,
        570,
        571,
        572,
        573,
        574,
        575,
        576,
        577,
        579,
        580,
        581,
        582,
        584,
        585,
        586,
        587,
        588,
        589,
        590,
        591,
        592,
        593,
        594,
        595,
        596,
        597,
        598,
        599,
        601,
        602,
        603,
        604,
        606,
        607,
        608,
        609,
        610,
        611,
        612,
        613,
        614,
        615,
        616,
        617,
        618,
        619,
        620,
        621,
        622,
        623,
    ]
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    graph = mof.structure_graph
    graph.structure = Structure.from_sites(graph.structure.sites)
    graph.remove_nodes(not_node)
    return graph


@pytest.fixture(scope="session")
def get_cgd_file():
    return HKUST_cdg


@pytest.fixture(scope="session")
def get_systre_output():
    return SYSTRE_OUT


@pytest.fixture(scope="session")
def get_systre_output2():
    return SYSTRE_OUT_2


@pytest.fixture(scope="session")
def get_formate_structure_and_graph():
    s = Structure.from_file(os.path.join(THIS_DIR, "test_files", "WIHSUD_clean.cif"))
    sg = StructureGraph.with_local_env_strategy(s, JmolNN())
    return s, sg


@pytest.fixture(scope="session")
def get_dicarboxy_biphenyl_graph():
    graph = nx.read_graphml(
        os.path.join(THIS_DIR, "test_files", "test_graph_biphenyl_dicarboxy.graphml")
    ).to_undirected()
    metal_indices = []
    for node_l in graph.nodes:
        node = graph.nodes[node_l]
        if "m" in node["label"]:
            metal_indices.append(node_l)

    branching_indices = []
    for node_l in graph.nodes:
        node = graph.nodes[node_l]
        if "b" in node["label"]:
            branching_indices.append(node_l)

    return graph, metal_indices, branching_indices


@pytest.fixture()
def porphyrin_mof_structure_and_graph():
    s = Structure.from_file(os.path.join(THIS_DIR, "test_files", "porphyrin_mof.cif"))
    sg = StructureGraph.with_local_env_strategy(s, JmolNN())
    return s, sg


@pytest.fixture()
def hkust_structure_and_graph():
    s = Structure.from_file(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    sg = StructureGraph.with_local_env_strategy(s, JmolNN())
    return s, sg


@pytest.fixture()
def get_methane_molecule_and_graph():
    coords = [
        [0.000000, 0.000000, 0.000000],
        [0.000000, 0.000000, 1.089000],
        [1.026719, 0.000000, -0.363000],
        [-0.513360, -0.889165, -0.363000],
        [-0.513360, 0.889165, -0.363000],
    ]
    methane = Molecule(["C", "H", "H", "H", "H"], coords)
    mg = MoleculeGraph.with_local_env_strategy(methane, JmolNN())
    return methane, mg


@pytest.fixture()
def get_acetate_zr_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "1585873-1585874.cif"))
    return mof


@pytest.fixture()
def get_across_periodic_boundary_node():
    # Zn4(μ3-OH)2(TPO)2(H2O)2]
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "NIYZIG.cif"))
    return mof


@pytest.fixture()
def get_linker_object():
    mg_dict = {
        "@module": "pymatgen.analysis.graphs",
        "@class": "MoleculeGraph",
        "molecule": {
            "@module": "pymatgen.core.structure",
            "@class": "Molecule",
            "charge": 0.0,
            "spin_multiplicity": 2,
            "sites": [
                {
                    "name": "H",
                    "species": [{"element": "H", "occu": 1}],
                    "xyz": [-42.67565999999999, -36.353339999999996, -45.520703999999995],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-45.915848999999994, -38.14466399999999, -44.335269],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-44.203554000000004, -36.537741000000004, -44.203554],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-40.884336, -33.11315100000001, -44.335269],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-43.044461999999996, -35.958195, -44.75675699999999],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-45.915849, -34.69373100000001, -40.884336],
                    "properties": {},
                },
                {
                    "name": "H",
                    "species": [{"element": "H", "occu": 1}],
                    "xyz": [-45.52070399999999, -36.35333999999999, -42.67565999999999],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-44.203554, -34.825446, -42.491259],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-44.862129, -34.166871, -41.332167],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-43.044461999999996, -34.272243, -43.044461999999996],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-44.335269, -33.113150999999995, -40.88433599999999],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-44.335269, -38.144664, -45.915848999999994],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-44.75675699999999, -35.958194999999996, -43.044461999999996],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-41.332167000000005, -34.166871, -44.862128999999996],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-42.49125899999999, -34.825446, -44.203554],
                    "properties": {},
                },
                {
                    "name": "H",
                    "species": [{"element": "H", "occu": 1}],
                    "xyz": [-42.67566, -33.508296, -42.67565999999999],
                    "properties": {},
                },
                {
                    "name": "O",
                    "species": [{"element": "O", "occu": 1}],
                    "xyz": [-40.884336, -34.693731, -45.915849],
                    "properties": {},
                },
                {
                    "name": "C",
                    "species": [{"element": "C", "occu": 1}],
                    "xyz": [-44.862129, -37.696833, -44.862128999999996],
                    "properties": {},
                },
            ],
        },
        "graphs": {
            "directed": True,
            "multigraph": True,
            "graph": [("edge_weight_name", None), ("edge_weight_units", None), ("name", "bonds")],
            "nodes": [
                {
                    "idx": 96,
                    "specie": "H",
                    "coords": np.array([-42.67566, -36.35334, -45.520704]),
                    "properties": {},
                    "coord": np.array([-42.67566, -36.35334, -45.520704]),
                    "id": 0,
                },
                {
                    "idx": 129,
                    "specie": "O",
                    "coords": np.array([-45.915849, -38.144664, -44.335269]),
                    "properties": {},
                    "coord": np.array([-45.915849, -38.144664, -44.335269]),
                    "id": 1,
                },
                {
                    "idx": 23,
                    "specie": "C",
                    "coords": np.array([-44.203554, -36.537741, -44.203554]),
                    "properties": {},
                    "coord": np.array([-44.203554, -36.537741, -44.203554]),
                    "id": 2,
                },
                {
                    "idx": 152,
                    "specie": "O",
                    "coords": np.array([-40.884336, -33.113151, -44.335269]),
                    "properties": {},
                    "coord": np.array([-40.884336, -33.113151, -44.335269]),
                    "id": 3,
                },
                {
                    "idx": 60,
                    "specie": "C",
                    "coords": np.array([-43.044462, -35.958195, -44.756757]),
                    "properties": {},
                    "coord": np.array([-43.044462, -35.958195, -44.756757]),
                    "id": 4,
                },
                {
                    "idx": 137,
                    "specie": "O",
                    "coords": np.array([-45.915849, -34.693731, -40.884336]),
                    "properties": {},
                    "coord": np.array([-45.915849, -34.693731, -40.884336]),
                    "id": 5,
                },
                {
                    "idx": 105,
                    "specie": "H",
                    "coords": np.array([-45.520704, -36.35334, -42.67566]),
                    "properties": {},
                    "coord": np.array([-45.520704, -36.35334, -42.67566]),
                    "id": 6,
                },
                {
                    "idx": 0,
                    "specie": "C",
                    "coords": np.array([-44.203554, -34.825446, -42.491259]),
                    "properties": {},
                    "coord": np.array([-44.203554, -34.825446, -42.491259]),
                    "id": 7,
                },
                {
                    "idx": 1,
                    "specie": "C",
                    "coords": np.array([-44.862129, -34.166871, -41.332167]),
                    "properties": {},
                    "coord": np.array([-44.862129, -34.166871, -41.332167]),
                    "id": 8,
                },
                {
                    "idx": 2,
                    "specie": "C",
                    "coords": np.array([-43.044462, -34.272243, -43.044462]),
                    "properties": {},
                    "coord": np.array([-43.044462, -34.272243, -43.044462]),
                    "id": 9,
                },
                {
                    "idx": 108,
                    "specie": "O",
                    "coords": np.array([-44.335269, -33.113151, -40.884336]),
                    "properties": {},
                    "coord": np.array([-44.335269, -33.113151, -40.884336]),
                    "id": 10,
                },
                {
                    "idx": 141,
                    "specie": "O",
                    "coords": np.array([-44.335269, -38.144664, -45.915849]),
                    "properties": {},
                    "coord": np.array([-44.335269, -38.144664, -45.915849]),
                    "id": 11,
                },
                {
                    "idx": 69,
                    "specie": "C",
                    "coords": np.array([-44.756757, -35.958195, -43.044462]),
                    "properties": {},
                    "coord": np.array([-44.756757, -35.958195, -43.044462]),
                    "id": 12,
                },
                {
                    "idx": 37,
                    "specie": "C",
                    "coords": np.array([-41.332167, -34.166871, -44.862129]),
                    "properties": {},
                    "coord": np.array([-41.332167, -34.166871, -44.862129]),
                    "id": 13,
                },
                {
                    "idx": 14,
                    "specie": "C",
                    "coords": np.array([-42.491259, -34.825446, -44.203554]),
                    "properties": {},
                    "coord": np.array([-42.491259, -34.825446, -44.203554]),
                    "id": 14,
                },
                {
                    "idx": 84,
                    "specie": "H",
                    "coords": np.array([-42.67566, -33.508296, -42.67566]),
                    "properties": {},
                    "coord": np.array([-42.67566, -33.508296, -42.67566]),
                    "id": 15,
                },
                {
                    "idx": 120,
                    "specie": "O",
                    "coords": np.array([-40.884336, -34.693731, -45.915849]),
                    "properties": {},
                    "coord": np.array([-40.884336, -34.693731, -45.915849]),
                    "id": 16,
                },
                {
                    "idx": 46,
                    "specie": "C",
                    "coords": np.array([-44.862129, -37.696833, -44.862129]),
                    "properties": {},
                    "coord": np.array([-44.862129, -37.696833, -44.862129]),
                    "id": 17,
                },
            ],
            "adjacency": [
                [{"to_jimage": (0, 0, 0), "id": 4, "key": 0}],
                [{"to_jimage": (0, 0, 0), "id": 17, "key": 0}],
                [
                    {"to_jimage": (0, 0, 0), "id": 17, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 12, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 4, "key": 0},
                ],
                [{"to_jimage": (0, 0, 0), "id": 13, "key": 0}],
                [
                    {"to_jimage": (0, 0, 0), "id": 0, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 2, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 14, "key": 0},
                ],
                [{"to_jimage": (0, 0, 0), "id": 8, "key": 0}],
                [{"to_jimage": (0, 0, 0), "id": 12, "key": 0}],
                [
                    {"to_jimage": (0, 0, 0), "id": 12, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 8, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 9, "key": 0},
                ],
                [
                    {"to_jimage": (0, 0, 0), "id": 5, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 7, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 10, "key": 0},
                ],
                [
                    {"to_jimage": (0, 0, 0), "id": 7, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 14, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 15, "key": 0},
                ],
                [{"to_jimage": (0, 0, 0), "id": 8, "key": 0}],
                [{"to_jimage": (0, 0, 0), "id": 17, "key": 0}],
                [
                    {"to_jimage": (0, 0, 0), "id": 2, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 6, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 7, "key": 0},
                ],
                [
                    {"to_jimage": (0, 0, 0), "id": 3, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 14, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 16, "key": 0},
                ],
                [
                    {"to_jimage": (0, 0, 0), "id": 4, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 9, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 13, "key": 0},
                ],
                [{"to_jimage": (0, 0, 0), "id": 9, "key": 0}],
                [{"to_jimage": (0, 0, 0), "id": 13, "key": 0}],
                [
                    {"to_jimage": (0, 0, 0), "id": 1, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 2, "key": 0},
                    {"to_jimage": (0, 0, 0), "id": 11, "key": 0},
                ],
            ],
        },
    }
    mg = MoleculeGraph.from_dict(mg_dict)
    mol = mg.molecule
    linker = Linker(
        mol,
        mg,
        graph_branching_indices={8, 13, 17},
        binding_indices=[129, 137, 108, 141, 120, 152],
        molecule_original_indices_mapping={
            0: [96],
            1: [129],
            2: [23],
            3: [152],
            4: [60],
            5: [137],
            6: [105],
            7: [0],
            8: [1],
            9: [2],
            10: [108],
            11: [141],
            12: [69],
            13: [37],
            14: [14],
            15: [84],
            16: [120],
            17: [46],
        },
    )
    return linker


@pytest.fixture()
def get_node_object():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    node = Node.from_mof_and_indices(
        mof,
        node_indices={32, 1, 132, 134, 72, 137, 74, 139, 108, 110, 113, 115, 27, 30},
        branching_indices={1, 27, 30, 32},
        binding_indices={108, 110, 113, 115, 132, 134, 137, 139},
    )

    return node
