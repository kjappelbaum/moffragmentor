__all__ = ["SBU", "Node", "Linker", "get_binding_indices"]

from pymatgen.io.babel import BabelMolAdaptor
from openbabel import pybel as pb
from pymatgen import Molecule
import nglview


class SBU:
    def __init__(self, molecule, connection_indices):
        self.molecule = molecule
        self._ob_mol = None

    @property
    def openbabel_mol(self):
        if self._ob_mol is not None:
            return self._ob_mol
        else:
            return self.get_openbabel_mol()

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
        return mol.write("can").strip()

    def write_tobacco_file(self, filename):
        """Can be based on https://github.com/peteboyd/lammps_interface/blob/71953f5f6706b75f059197aa7f152695e54ded85/lammps_interface/structure_data.py#L1636"
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
