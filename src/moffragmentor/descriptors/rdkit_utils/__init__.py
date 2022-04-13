"""
Miscellaneous.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

from rdkit import Chem


class PicklableMol(Chem.Mol):
    """
    RDKit Mol that preserves molecule properties when pickling.

    This class is similar to the PropertyMol class in RDKit. However, this
    class can optionally preserve calculated properties.

    Parameters
    ----------
    mol : RDKit Mol, optional
        Molecule to convert to PicklableMol.
    preserve_computed : bool, optional (default True)
        Whether to preserve computed properties when pickling.
    """
    __getstate_manages_dict__ = True

    def __init__(self, mol=None, preserve_computed=True):
        if mol is None:
            super(PicklableMol, self).__init__()
        else:
            super(PicklableMol, self).__init__(mol)
        self.preserve_computed = preserve_computed

    def __getstate__(self):
        """
        Reduce the molecule and save property information.
        """
        properties = {}
        computed_properties = {}
        for prop in self.GetPropNames(includePrivate=True):
            properties[prop] = self.GetProp(prop)
        for prop in self.GetPropNames(includePrivate=True,
                                      includeComputed=True):
            if prop not in properties:
                try:
                    computed_properties[prop] = self.GetProp(prop)
                except RuntimeError:
                    pass
        mol = self.ToBinary()
        return mol, properties, computed_properties, self.__dict__

    def __setstate__(self, state):
        """
        Restore molecule properties without overwriting anything.

        Parameters
        ----------
        state : tuple
            Molecule state returned by __getstate__.
        """
        mol, properties, computed, object_dict = state
        self.__init__(mol)
        for key, value in object_dict.items():
            self.__dict__[key] = value
        for prop, value in properties.items():
            if not self.HasProp(prop):
                self.SetProp(prop, value)
        if self.preserve_computed:
            for prop, value in computed.items():
                if not self.HasProp(prop):
                    self.SetProp(prop, value, computed=True)