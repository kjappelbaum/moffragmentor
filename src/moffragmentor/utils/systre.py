# -*- coding: utf-8 -*-
# pylint: disable=line-too-long
"""Methods for running systre"""
import os
import re
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import List, Tuple

from loguru import logger
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core import Lattice

from . import is_tool
from .errors import JavaNotFoundError

__all__ = ["run_systre"]

THIS_DIR = os.path.dirname(os.path.realpath(__file__))

SYSTRE_JAR = os.path.abspath(os.path.join(THIS_DIR, "..", "Systre-19.6.0.jar"))


@contextmanager
def closed_named_tempfile(contents, mode="w", suffix=".cdg"):
    file = NamedTemporaryFile(delete=False, mode=mode, suffix=suffix)
    try:
        with file:
            file.write(contents)
        yield file.name
    finally:
        os.unlink(file.name)


def run_systre(systre_string: str) -> dict:
    if not is_tool("java"):
        raise JavaNotFoundError("To determine the topology of the net, `java` must be in the PATH.")
    try:
        with closed_named_tempfile(systre_string, suffix=".cgd", mode="w") as filename:
            cmd_list = [
                "java",
                "-cp",
                str(SYSTRE_JAR),
                "org.gavrog.apps.systre.SystreCmdline",
                filename,
            ]
            out = subprocess.run(
                cmd_list,
                universal_newlines=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            systre_result = _parse_systre_lines(out.stdout.split("\n"))

            return systre_result
    except Exception:  # pylint:disable=broad-except
        logger.warning("Error running Systre.")
        return ""


def _parse_systre_lines(
    lines: List[str],
) -> dict:
    """Parse the output of Systre.

    Input for this function might be created with

    ```python
    with open('systre.out', 'r') as handle:
        lines = handle.readlines()
    ```

    Args:
        lines (List[str]): Output of systre run,
            parsed into a list of strings (one line per string)

    Returns:
        dict: parsed output
    """
    rcsr_line = float("inf")
    cell_line = float("inf")
    angle_line = float("inf")
    in_node_block = False
    in_edge_block = False
    node_start_line = float("inf")
    edge_start_line = float("inf")
    nodes = defaultdict(list)
    edges = []
    for i, line in enumerate(lines):
        if "Ideal space group" in line:
            space_group = line.split("is")[-1].strip().replace(".", "")
        if "Structure was identified with RCSR symbol:" in line:
            rcsr_line = i + 1
        if i == rcsr_line:
            rcsr_code = line.split()[-1].strip()
        if "Relaxed cell parameters" in line:
            cell_line = i + 1
            angle_line = i + 2
        if i == cell_line:
            cell = _line_to_coords(line)
        if i == angle_line:
            angles = _line_to_coords(line)
        if "Relaxed positions:" in line:
            node_start_line = i + 1
        if i == node_start_line:
            in_node_block = True
        if i == edge_start_line:
            in_edge_block = True
        if "Edges:" in line:
            in_node_block = False
            edge_start_line = i + 1
        if "Edge centers:" in line:
            in_edge_block = False
        if in_node_block:
            try:
                node, coords = _parse_node_line(line)
                nodes[node].append(coords)
            except IndexError:
                pass

        if in_edge_block:
            try:
                edge_coords = _line_to_coords(line)
                edges.append((edge_coords[:3], edge_coords[3:]))
            except IndexError:
                pass

    results = {
        "space_group": space_group,
        "rcsr_code": rcsr_code,
        "relaxed_cell": cell,
        "relaxed_angles": angles,
        "relaxed_node_positions": dict(nodes),
        "relaxed_edges": edges,
    }

    return results


def _line_to_coords(line: str) -> List[float]:
    return [float(i) for i in re.findall(r"\d+.\d+", line)]


def _parse_node_line(line: str) -> Tuple[int, List[float]]:
    node_number = int(re.findall(r"Node\s(\d+):", line)[0])
    coords = _line_to_coords(line)

    return node_number, coords


def _get_systre_input_from_pmg_structure_graph(
    structure_graph: StructureGraph, lattice: Lattice = None
) -> str:
    """
    Loop over all atoms in a StructureGraph and use them as nodes.

    Place edges such that all nodes are represented with their
    full connectivity.

    Args:
        structure_graph (StructureGraph): pymatgen StructureGraph
            object representing the net. This will correspond to having
            one atom per SBU in the structure graph.
        lattice (Lattice): pymatgen lattice object. Needed for the cell dimension

    Returns:
        str: systre input string. Does not contain the optional edge centers
    """
    lattice = structure_graph.structure.lattice if lattice is None else lattice
    vertices = []
    edges = []

    symmetry_group = "   GROUP P1"

    cell_line = f"   CELL {lattice.a} {lattice.b} {lattice.c} {lattice.alpha} {lattice.beta} {lattice.gamma}"

    frac_coords = structure_graph.structure.frac_coords

    for i in range(len(structure_graph)):
        vertices.append((len(structure_graph.get_connected_sites(i)), frac_coords[i]))
        # vertices.append((structure_graph.get_coordination_of_site(i), frac_coords[i]))

    for edge in structure_graph.graph.edges(data=True):
        start = frac_coords[edge[0]]
        end = frac_coords[edge[1]] + edge[2]["to_jimage"]
        edges.append((tuple(start), tuple(end)))

    def _create_vertex_string(counter, coordination, coordinate):
        return f"   NODE {counter} {coordination} {coordinate[0]:.4f} {coordinate[1]:.4f} {coordinate[2]:.4f}"

    def _create_edge_string(coordinate_0, coordinate_1):
        return f"   EDGE {coordinate_0[0]:.4f} {coordinate_0[1]:.4f} {coordinate_0[2]:.4f} {coordinate_1[0]:.4f} {coordinate_1[1]:.4f} {coordinate_1[2]:.4f}"  # noqa:E501

    edge_lines = set()
    vertex_lines = set()

    for i, vertex in enumerate(vertices):
        vertex_lines.add(_create_vertex_string(i, vertex[0], vertex[1]))

    for _, edge in enumerate(edges):
        edge_lines.add(_create_edge_string(edge[0], edge[1]))

    file_lines = (
        ["CRYSTAL", "   NAME", symmetry_group, cell_line]
        + list(vertex_lines)
        + list(edge_lines)
        + ["END"]
    )

    return "\n".join(file_lines)
