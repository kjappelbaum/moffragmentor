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


def _parse_systre_lines(  # pylint:disable=too-many-locals, too-many-branches
    lines: List[str],
) -> dict:
    """Given the lines from a Systre output file, which might be created with
    ```
    with open('systre.out', 'r') as handle:
        lines = handle.readlines()
    ```

    Args:
        lines (List[str]): [description]

    Returns:
        dict: [description]
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
