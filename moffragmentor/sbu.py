# -*- coding: utf-8 -*-
__all__ = ["SBU", "Node", "Linker"]

import datetime
from copy import deepcopy
from typing import List

import nglview
import numpy as np
from openbabel import pybel as pb
from pymatgen import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.io.babel import BabelMolAdaptor
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import pdist


def get_max_sep(coordinates):
    distances = pdist(coordinates)
    return np.max(distances)


class SBU:
    def __init__(
        self,
        molecule: Molecule,
        molecule_graph: MoleculeGraph,
        connection_indices: List[int],
    ):
        self.molecule = molecule
        self._ob_mol = None
        self._smiles = None
        self.molecule_graph = molecule_graph
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
            return self.get_openbabel_mol()

    @classmethod
    def from_labled_molecule(cls, mol, mg):
        connection_indices = get_binding_indices(mol)
        return cls(mol, mg, connection_indices)

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

    def _get_tobacco_string(self):
        max_size = get_max_sep(self.molecule.cart_coords)
        s = self.molecule.get_boxed_structure(
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            max_size + 0.1 * max_size,
            reorder=False,
        )

        header_lines = [
            "data_",
            "_audit_creation_date              "
            + datetime.datetime.today().strftime("%Y-%m-%d"),
            "_audit_creation_method            'moffragmentor'",
            "_symmetry_space_group_name_H-M    'P1'",
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

            if i in self.connection_indices:
                ind = "X" + str(i)
                loop_content.append(
                    "{:7} {:>4} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}".format(
                        ind, es, vec[0], vec[1], vec[2], 0
                    )
                )
            else:
                ind = es + str(i)
                loop_content.append(
                    "{:7} {:>4} {:>15.6f} {:>15.6f} {:>15.6f} {:>15.6f}".format(
                        ind, es, vec[0], vec[1], vec[2], 0
                    )
                )
            site_index[i] = ind

        connection_loop_header = [
            "loop_",
            "_geom_bond_atom_site_label_1",
            "_geom_bond_atom_site_label_2",
            "_geom_bond_distance",
            "_geom_bond_site_symmetry_1",
            "_ccdc_geom_bond_type",
        ]

        connection_loop_content = []
        set_bond = set()
        for i, site in enumerate(self.molecule):
            neighbors = self.molecule_graph.get_connected_sites(0)
            for j, _ in enumerate(neighbors):
                ind0 = site_index[i]
                ind1 = site_index[j]

                tuple_a = (ind0, ind1)
                tuple_b = (ind1, ind0)

                if not (tuple_a in set_bond) and not (tuple_b in set_bond):
                    set_bond.add(tuple_a)

                    dist = np.round(self.molecule.get_distance(i, j), 3)
                    # bond_type = data["bond_type"]

                    connection_loop_content.append(
                        "{:7} {:>7} {:>7} {:>3}".format(ind0, ind1, dist, "S")
                    )

        return "\n".join(
            header_lines
            + cell_lines
            + loop_header
            + loop_content
            + connection_loop_header
            + connection_loop_content
        )

    def write_tobacco_file(self, filename=None):
        """To create a database of building blocks it is practical to be able to
        write Tobacco input file.
        We need to only place the X for sites with property binding=True
        """

        cif_string = self._get_tobacco_string()
        if filename is None:
            return cif_string


class Node(SBU):
    pass


def _get_edge_dict_from_rdkit_mol(mol):
    edges = {}
    for bond in mol.GetBonds():
        edges[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = None
    return edges


def _make_mol_from_rdkit_mol(mol):
    """Takes the first conformer"""
    m = deepcopy(mol)
    AllChem.EmbedMolecule(m)
    conf = m.GetConformers()
    positions = conf[0].GetPositions()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Molecule(symbols, positions)


class Linker(SBU):
    @classmethod
    def from_carboxy_mol(cls, mol):
        carboxy = Chem.MolFromSmarts("[O]C(=O)")
        matches = mol.GetSubstructMatches(carboxy)
        connection_sites = []
        for tpl in matches:
            connection_sites.append(tpl[0])
            connection_sites.append(tpl[2])

        edges = _get_edge_dict_from_rdkit_mol(mol)
        pmg_mol = _make_mol_from_rdkit_mol(mol)
        mg = MoleculeGraph.with_edges(pmg_mol, edges)

        return cls(pmg_mol, mg, connection_sites)


def get_binding_indices(mol):
    indices = []
    for i, site in enumerate(mol):
        try:
            if site.properties["binding"] == True:
                indices.append(i)
        except KeyError:
            pass

    return indices
