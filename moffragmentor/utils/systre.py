# -*- coding: utf-8 -*-
import os
import re
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import List, Tuple


@contextmanager
def ClosedNamedTempfile(contents, mode="w", suffix=".cdg"):
    f = NamedTemporaryFile(delete=False, mode=mode, suffix=suffix)
    try:
        with f:
            f.write(contents)
        yield f.name
    finally:
        os.unlink(f.name)


THIS_DIR = os.path.dirname(os.path.realpath(__file__))

SYSTRE_JAR = os.path.abspath(os.path.join(THIS_DIR, "..", "Systre-19.6.0.jar"))


def run_systre(systre_string: str) -> dict:
    with ClosedNamedTempfile(systre_string, suffix=".cgd", mode="w") as filename:
        cmd_list = [
            "java",
            "-cp",
            "/Users/kevinmaikjablonka/Dropbox (LSMO)/Documents/open_source/pymoffragmentor/moffragmentor/Systre-19.6.0.jar",
            "org.gavrog.apps.systre.SystreCmdline",
            filename,
        ]
        out = subprocess.run(
            cmd_list,
            universal_newlines=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        systre_result = parse_systre_lines(out.stdout.split("\n"))

        return systre_result


def parse_systre_lines(lines: List[str]) -> dict:
    rscr_line = float("inf")
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
            rscr_line = i + 1
        if i == rscr_line:
            rscr_code = line.split()[-1].strip()
        if "Relaxed cell parameters" in line:
            cell_line = i + 1
            angle_line = i + 2
        if i == cell_line:
            cell = line_to_coords(line)
        if i == angle_line:
            angles = line_to_coords(line)
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
            node, coords = parse_node_line(line)
            nodes[node].append(coords)

        if in_edge_block:
            edge_coords = line_to_coords(line)
            edges.append((edge_coords[:3], edge_coords[3:]))

    results = {
        "space_group": space_group,
        "rscr_code": rscr_code,
        "relaxed_cell": cell,
        "relaxed_angles": angles,
        "relaxed_node_positions": nodes,
        "relaxed_edges": edges,
    }

    return results


def line_to_coords(line: str) -> List[float]:
    return [float(i) for i in re.findall("\d+.\d+", line)]


def parse_node_line(line) -> Tuple[int, List[float]]:
    node_number = int(re.findall("Node\s(\d+):", line)[0])
    coords = line_to_coords(line)

    return node_number, coords
