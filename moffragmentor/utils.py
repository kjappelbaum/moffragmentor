# -*- coding: utf-8 -*-
import datetime
import os
import pickle

import numpy as np


def write_cif(s, graph, connection_indices, molecule=None, write_bonding_mode=False):
    header_lines = [
        "data_cif",
        "_audit_creation_date              "
        + datetime.datetime.today().strftime("%Y-%m-%d"),
        "_audit_creation_method            'moffragmentor'",
        "_symmetry_space_group_name_H-M    'P 1'",
        "_symmetry_Int_Tables_number       1",
        "_symmetry_cell_setting            triclinic",
        "loop_",
        "_symmetry_equiv_pos_as_xyz",
        "  x,y,z",
    ]

    cell_lines = [
        "_cell_length_a                    " + str(s.lattice.a),
        "_cell_length_b                    " + str(s.lattice.b),
        "_cell_length_c                    " + str(s.lattice.c),
        "_cell_angle_alpha                 " + str(s.lattice.alpha),
        "_cell_angle_beta                  " + str(s.lattice.beta),
        "_cell_angle_gamma                 " + str(s.lattice.gamma),
    ]

    loop_header = [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
        "_atom_site_charge",
    ]

    loop_content = []
    site_index = {}
    for i, site in enumerate(s):
        vec = site.frac_coords
        es = str(site.specie)

        if i in connection_indices:
            ind = "X" + str(i)
            loop_content.append(
                " {} {}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                    ind, es, vec[0], vec[1], vec[2], 0
                )
            )
        else:
            ind = es + str(i)
            loop_content.append(
                " {} {}  {:.6f}  {:.6f}  {:.6f}  {:.6f}".format(
                    ind, es, vec[0], vec[1], vec[2], 0
                )
            )
        site_index[i] = ind

    connection_loop_header = [
        "loop_",
        "_geom_bond_atom_site_label_1",
        "_geom_bond_atom_site_label_2",
    ]

    if write_bonding_mode:

        connection_loop_header += ["_geom_bond_distance", "_ccdc_geom_bond_type"]

    connection_loop_content = []
    set_bond = set()
    if molecule:
        s = molecule
    for i, site in enumerate(s):
        neighbors = graph.get_connected_sites(i)
        for j, neighbor_site in enumerate(neighbors):
            ind0 = site_index[i]
            ind1 = site_index[neighbor_site.index]

            tuple_a = (ind0, ind1)
            tuple_b = (ind1, ind0)

            if not (tuple_a in set_bond) and not (tuple_b in set_bond):
                set_bond.add(tuple_a)

                dist = np.round(s.get_distance(i, j), 3)
                if write_bonding_mode:
                    connection_loop_content.append(
                        "{:7} {:>7} {:>7} {:>3}".format(ind0, ind1, dist, "S")
                    )
                else:
                    connection_loop_content.append(
                        "{:7} {:>7} {:>7} {:>3}".format(ind0, ind1, "", "")
                    )

    return "\n".join(
        header_lines
        + cell_lines
        + loop_header
        + loop_content
        + connection_loop_header
        + connection_loop_content
    )


def pickle_dump(object, path):
    with open(path, "wb") as handle:
        pickle.dump(object, handle)


def make_if_not_exists(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
