"""
I/O functions: reading and writing molecules.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import cPickle
import gzip
import numpy as np
import os
import warnings

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover


class MolIO(object):
    """
    Base class for molecule I/O.

    Parameters
    ----------
    f : file-like, optional
        File-like object.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf', 'smi', and 'pkl'.
    """
    def __init__(self, f=None, mol_format=None):
        self.f = f
        self.mol_format = mol_format

        # placeholder
        self.filename = None

    def __del__(self):
        self.close()

    def __enter__(self):
        """
        Context manager entrance. Returns this object so it can be closed
        after finishing the 'with' block.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit. Closes any open file handles.

        Parameters
        ----------
        exc_type : class
            Exception class that caused the context to exit.
        exc_val : object
            Exception that caused the context to exit.
        exc_tb : traceback
            Exception traceback.

        Note that all arguments will be None on successful completion of
        the context. See https://docs.python.org/2/reference/datamodel.html
        #context-managers for more information. Any exception will be
        raised as normal after this method is finished (since it does not
        return True).
        """
        self.close()

    def open(self, filename, mol_format=None, mode='rb'):
        """
        Open a file for reading or writing.

        Parameters
        ----------
        filename : str
            Filename.
        mol_format : str, optional
            Molecule file format. Currently supports 'sdf', 'smi', and
            'pkl'. If not provided, the format is inferred from the
            filename.
        mode : str, optional (default 'rb')
            Mode used to open file.
        """
        self.filename = filename
        if filename.endswith('.gz'):
            self.f = gzip.open(filename, mode)
        else:
            self.f = open(filename, mode)
        if mol_format is not None:
            self.mol_format = mol_format
        else:
            self.mol_format = self.guess_mol_format(filename)
        return self

    def close(self):
        """
        Close output file (only if it was opened by this object).
        """
        if self.f is not None and self.filename is not None:
            self.f.close()

        # cleanup
        self.filename = None

    def guess_mol_format(self, filename):
        """
        Guess molecule file format from filename.

        Parameters
        ----------
        filename : str
            Filename.
        """

        # strip gzip suffix
        if filename.endswith('.gz'):
            filename = os.path.splitext(filename)[0]

        # guess format from extension
        if filename.endswith('.sdf'):
            mol_format = 'sdf'
        elif filename.endswith(('.smi', '.can', '.ism')):
            mol_format = 'smi'
        elif filename.endswith('.pkl'):
            mol_format = 'pkl'
        else:
            raise NotImplementedError('Unrecognized file format.')
        return mol_format


class MolReader(MolIO):
    """
    Read molecules from files and file-like objects. Supports SDF, SMILES,
    and RDKit binary format (via pickle).

    Parameters
    ----------
    f : file, optional
        File-like object.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf', 'smi', and 'pkl'.
    remove_hydrogens : bool, optional (default False)
        Remove hydrogens from molecules.
    remove_salts : bool, optional (default True)
        Remove salts from molecules. Note that this will remove any hydrogens
        present on the molecule.
    compute_2d_coords : bool, optional (default True)
        Compute 2D coordinates when reading SMILES. If molecules are written to
        SDF without 2D coordinates, stereochemistry information will be lost.
    """
    def __init__(self, f=None, mol_format=None, remove_hydrogens=False,
                 remove_salts=True, compute_2d_coords=True):
        if not remove_hydrogens and remove_salts:
            warnings.warn('Compounds with salts will have hydrogens removed')
        super(MolReader, self).__init__(f, mol_format)
        self.remove_hydrogens = remove_hydrogens
        self.remove_salts = remove_salts
        if remove_salts:
            self.salt_remover = SaltRemover()
        self.compute_2d_coords = compute_2d_coords

    def __iter__(self):
        """
        Iterate over molecules.
        """
        return self.get_mols()

    def get_mols(self):
        """
        Read molecules from a file-like object.

        Molecule conformers are grouped into a single molecule. Two
        molecules are considered conformers of the same molecule if they:
        * Are contiguous in the file
        * Have identical (canonical isomeric) SMILES strings
        * Have identical compound names (if set)

        Returns
        -------
        A generator yielding (possibly multi-conformer) RDKit Mol objects.
        """
        parent = None
        for mol in self._get_mols():
            if parent is None:
                parent = mol
                continue
            if self.are_same_molecule(parent, mol):
                if mol.GetNumConformers():
                    for conf in mol.GetConformers():
                        parent.AddConformer(conf, assignId=True)
                else:
                    continue  # skip duplicate molecules without conformers
            else:
                parent = self.clean_mol(parent)
                if parent is not None:
                    yield parent
                parent = mol
        if parent is not None:
            parent = self.clean_mol(parent)
            if parent is not None:
                yield parent

    def _get_mols(self):
        """
        Read molecules from a file-like object.

        This method returns individual conformers from a file and does not
        attempt to combine them into multiconformer Mol objects.

        Returns
        -------
        A generator yielding RDKit Mol objects.
        """
        if self.mol_format == 'sdf':
            mols = self._get_mols_from_sdf()
        elif self.mol_format == 'smi':
            mols = self._get_mols_from_smiles()
        elif self.mol_format == 'pkl':
            mols = self._get_mols_from_pickle()
        else:
            raise NotImplementedError('Unrecognized molecule format ' +
                                      '"{}"'.format(self.mol_format))

        # skip read errors
        while True:
            try:
                mol = mols.next()
            except StopIteration:
                break
            except Exception:
                warnings.warn('Skipping molecule.')
                continue
            else:
                if mol is not None:
                    yield mol

    def _get_mols_from_sdf(self):
        """
        Read SDF molecules from a file-like object.
        """
        supplier = Chem.ForwardSDMolSupplier(self.f,
                                             removeHs=self.remove_hydrogens)
        for mol in supplier:
            yield mol

    def _get_mols_from_smiles(self):
        """
        Read SMILES molecules from a file-like object.
        """
        for line in self.f.readlines():
            line = line.strip()
            if not line:
                continue
            split_line = line.split()
            if len(split_line) > 1:
                smiles, name = split_line
            else:
                smiles, = split_line
                name = None

            # hydrogens are removed by default, which triggers sanitization
            try:
                if self.remove_hydrogens:
                    mol = Chem.MolFromSmiles(smiles)
                else:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    Chem.SanitizeMol(mol)

                if self.compute_2d_coords:
                    AllChem.Compute2DCoords(mol)
            except Exception:
                warnings.warn('Skipping ' + line)
                continue
            else:
                if name is not None:
                    mol.SetProp('_Name', name)
                yield mol

    def _get_mols_from_pickle(self):
        """
        Read pickled molecules from a file-like object.

        Files that contain multiple pickles are supported by repeated calls
        to load.
        """
        while True:
            try:
                mols = cPickle.load(self.f)
                for mol in np.atleast_1d(mols):
                    yield mol
            except EOFError:
                break

    def are_same_molecule(self, a, b):
        """
        Test whether two molecules are conformers of the same molecule.

        Test for:
        * Identical (canonical isomeric) SMILES strings
        * Identical compound names (if set)

        Parameters
        ----------
        a, b : RDKit Mol
            Molecules to compare.
        """

        # get names, if available
        a_name = self._get_name(a)
        b_name = self._get_name(b)

        # get canonical isomeric SMILES
        a_smiles = self._get_isomeric_smiles(a)
        b_smiles = self._get_isomeric_smiles(b)
        assert a_smiles and b_smiles

        # test for same molecule
        return a_smiles == b_smiles and a_name == b_name

    def _get_name(self, mol):
        """
        Get molecule name, if available.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        if mol.HasProp('_Name'):
            return mol.GetProp('_Name')
        else:
            return None

    def _get_isomeric_smiles(self, mol):
        """
        Get canonical isomeric SMILES for a molecule. Also sets the
        isomericSmiles property to avoid recomputing.

        Note that stereochemistry is not assigned from 3D coordinates; it
        must be explicitly present in the file or it will not show up in
        the SMILES conversion.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        if mol.HasProp('isomericSmiles'):
            return mol.GetProp('isomericSmiles')
        else:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            mol.SetProp('isomericSmiles', smiles, computed=True)
            return smiles

    def clean_mol(self, mol):
        """
        Clean a molecule.

        Parameters
        ----------
        mol : RDKit Mol
            Molecule.
        """
        if self.remove_salts:
            # hydrogens must be removed for pattern matching to work properly
            try:
                mol_no_h = Chem.RemoveHs(mol)
            except ValueError:
                if mol.HasProp('_Name'):
                    name = mol.GetProp('_Name')
                else:
                    name = Chem.MolToSmiles(mol, isomericSmiles=True,
                                            canonical=True)
                warnings.warn('Skipping ' + name)
                return None
            new = self.salt_remover.StripMol(mol_no_h)
            # only keep if it is valid (# the molecule may _be_ a salt) and has
            # actually been changed
            if new.GetNumAtoms() and mol_no_h.ToBinary() != new.ToBinary():
                mol = new
        return mol


class MolWriter(MolIO):
    """
    Write molecules to files or file-like objects. Supports SDF, SMILES,
    and RDKit binary format (via pickle).

    Parameters
    ----------
    f : file, optional
        File-like object.
    mol_format : str, optional
        Molecule file format. Currently supports 'sdf', 'smi', and 'pkl'.
    stereo : bool, optional (default True)
        Whether to preserve stereochemistry in output.
    """
    def __init__(self, f=None, mol_format=None, stereo=True):
        super(MolWriter, self).__init__(f, mol_format)
        self.stereo = stereo

    def open(self, filename, mol_format=None, mode='wb'):
        """
        Open output file.

        Parameters
        ----------
        filename : str
            Filename.
        mol_format : str, optional
            Molecule file format. Currently supports 'sdf', 'smi', and
            'pkl'. If not provided, the format is inferred from the
            filename.
        mode : str, optional (default 'wb')
            Mode used to open file.
        """
        return super(MolWriter, self).open(filename, mol_format, mode)

    def write(self, mols):
        """
        Write molecules to a file-like object.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        if self.mol_format == 'sdf':
            self._write_sdf(mols)
        elif self.mol_format == 'smi':
            self._write_smiles(mols)
        elif self.mol_format == 'pkl':
            self._write_pickle(mols)
        self.f.flush()  # flush changes

    def _write_sdf(self, mols):
        """
        Write molecules in SDF format.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        w = Chem.SDWriter(self.f)
        for mol in mols:
            if not self.stereo:
                mol = Chem.Mol(mol)  # create a copy
                Chem.RemoveStereochemistry(mol)
            if mol.GetNumConformers():
                for conf in mol.GetConformers():
                    w.write(mol, confId=conf.GetId())
            else:
                w.write(mol)
        w.close()

    def _write_smiles(self, mols):
        """
        Write molecules in SMILES format.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        for mol in mols:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=self.stereo,
                                      canonical=True)
            self.f.write(smiles)
            if mol.HasProp('_Name'):
                name = mol.GetProp('_Name')
                self.f.write('\t' + name)
            self.f.write('\n')

    def _write_pickle(self, mols):
        """
        Write molecules to a pickle.

        Parameters
        ----------
        mols : iterable
            Molecules to write.
        """
        cPickle.dump(mols, self.f, cPickle.HIGHEST_PROTOCOL)