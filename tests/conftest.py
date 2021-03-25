# -*- coding: utf-8 -*-
import os

import networkx as nx
import pytest
from pymatgen import Structure
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import JmolNN

from moffragmentor import MOF

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


@pytest.fixture(scope="module")
def get_cuiiibtc_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "KAJZIH_freeONLY.cif"))
    return mof


@pytest.fixture(scope="module")
def get_hkust_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "HKUST-1.cif"))
    return mof


@pytest.fixture(scope="module")
def get_porphryin_mof():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "porphyrin_mof.cif"))
    return mof


@pytest.fixture(scope="module")
def get_hypothetical_mof():
    mof = MOF.from_cif(
        os.path.join(THIS_DIR, "test_files", "bcs_v1-litcic_1B_4H_Ch_opt_charge.cif")
    )
    return mof


@pytest.fixture(scope="module")
def get_p_linker_with_floating():
    mof = MOF.from_cif(os.path.join(THIS_DIR, "test_files", "MAGBON.cif"))
    return mof


@pytest.fixture(scope="module")
def get_cgd_file():
    return HKUST_cdg


@pytest.fixture(scope="module")
def get_systre_output():
    return SYSTRE_OUT


@pytest.fixture(scope="module")
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
