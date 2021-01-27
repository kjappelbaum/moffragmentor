# -*- coding: utf-8 -*-
__all__ = ["SBU", "Node", "Linker"]

import datetime
from typing import List

import nglview
import numpy as np
from openbabel import pybel as pb
from pymatgen import Molecule
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem


def write_tobacco_cif(m: Molecule, connection_indices):

    header_lines = [
        "data_",
        "_audit_creation_date              "
        + datetime.datetime.today().strftime("%Y-%m-%d"),
        "_audit_creation_method            'moffragmentor'",
        "_symmetry_space_group_name_H-M    'P1'",
        "_symmetry_Int_Tables_number       1",
        "_symmetry_cell_setting            triclinic",
        "loop_" "_symmetry_equiv_pos_as_xyz",
        "  x,y,z",
    ]

    cell_lines = [
        "_cell_length_a                    " + str(a),
        "_cell_length_b                    " + str(b),
        "_cell_length_c                    " + str(c),
        "_cell_angle_alpha                 " + str(alpha),
        "_cell_angle_beta                  " + str(beta),
        "_cell_angle_gamma                 " + str(gamma),
    ]

    loop_header = [
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z" "_atom_site_charge",
    ]

    index_dict = {}
    for i, site in enumerate(m):
        vec = data["fractional_position"]
        es = data["element_symbol"]
        ind = es + str(data["index"])
        index_dict[n] = ind
        chg = data["charge"]
        out.write(
            "{:7} {:>4} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}".format(
                ind, es, vec[0], vec[1], vec[2], chg
            )
        )
        out.write("\n")
        if i in connection_indices:
            # Add dummy for the connection
            ...

    out.write("loop_" + "\n")
    out.write("_geom_bond_atom_site_label_1" + "\n")
    out.write("_geom_bond_atom_site_label_2" + "\n")
    out.write("_geom_bond_distance" + "\n")
    # out.write('_geom_bond_site_symmetry_1' + '\n')
    out.write("_ccdc_geom_bond_type" + "\n")

    for n0, n1, data in G.edges(data=True):

        ind0 = index_dict[n0]
        ind1 = index_dict[n1]
        dist = np.round(data["length"], 3)
        bond_type = data["bond_type"]

        out.write("{:7} {:>7} {:>7} {:>3}".format(ind0, ind1, dist, bond_type))
        out.write("\n")


class SBU:
    def __init__(self, molecule: Molecule, connection_indices: List[int]):
        self.molecule = molecule
        self._ob_mol = None
        self._smiles = None
        self.connection_indices = connection_indices

    @property
    def rdkit_mol(self):
        if self._rdkti_mol is not None:
            return self._rdkti_mol
        else:
            self._rdkti_mol = Chem.MolFromSmiles(self.smiles)
            return self.rdkit_mol

    @property
    def openbabel_mol(self):
        if self._ob_mol is not None:
            return self._ob_mol
        else:
            return self.openbabel_mol()

    @classmethod
    def from_labled_molecule(cls, mol):
        connection_indices = get_binding_indices(mol)
        return cls(mol, connection_indices)

    def get_openbabel_mol(self):
        a = BabelMolAdaptor(self.molecule)
        pm = pb.Molecule(a.openbabel_mol)
        self._ob_mol = pm
        return pm

    def show_molecule(self):
        return nglview.show_pymatgen(self.molecule)

    @property
    def smiles(self):
        mol = self.openbabel_mol
        self._smiles = mol.write("can").strip()
        return self._smiles

    def write_tobacco_file(self, filename):
        """To create a database of building blocks it is practical to be able to
        write Tobacco input file.
        We need to only place the X for sites with property binding=True
        """
        ...


class Node(SBU):
    pass


class Linker(SBU):
    pass


def get_binding_indices(mol):
    indices = []
    for i, site in enumerate(mol):
        try:
            if site.properties["binding"] == True:
                indices.append(i)
        except KeyError:
            pass

    return indices
