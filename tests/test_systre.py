# -*- coding: utf-8 -*-
from moffragmentor.utils.systre import parse_systre_lines, run_systre


def test_systre_call(get_cgd_file):
    """Test the call to systre and the parsing of the output"""
    input_file = get_cgd_file
    systre_output = run_systre(input_file)
    assert systre_output["space_group"] == "Fm-3m"
    assert systre_output["rscr_code"] == "tbo"
    assert systre_output["relaxed_cell"] == [4.89898, 4.89898, 4.89898]
    assert len(systre_output["relaxed_node_positions"]) == 2
    assert isinstance(systre_output["relaxed_node_positions"], dict)


def test_systre_parser(get_systre_output, get_systre_output_2):
    systre_output = parse_systre_lines(get_systre_output.split("\n"))
    assert systre_output["space_group"] == "Fm-3m"
    assert systre_output["rscr_code"] == "tbo"
    assert systre_output["relaxed_cell"] == [4.89898, 4.89898, 4.89898]
    assert len(systre_output["relaxed_node_positions"]) == 2
    assert isinstance(systre_output["relaxed_node_positions"], dict)

    systre_output2 = parse_systre_lines(get_systre_output_2.split("\n"))
    assert systre_output["space_group"] == "p4mm"
    assert systre_output["rscr_code"] == "mtf"
    assert systre_output["relaxed_cell"] == [3.05792, 3.05792, 90.0000]
