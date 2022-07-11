# -*- coding: utf-8 -*-
# pylint:disable=no-member
"""Methods to rank molecules according to some measure of similarity."""
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFMCS


def tanimoto_rank(smiles_reference, smiles, additional_attributes=None):
    """Rank SMILES based on the Tanimoto similarity to the reference smiles."""
    distances = []
    mol = Chem.MolFromSmiles(smiles_reference)
    fp = Chem.RDKFingerprint(mol)
    for i, smile in enumerate(smiles):
        mol2 = Chem.MolFromSmiles(smile)
        fp2 = Chem.RDKFingerprint(mol2)
        add_attr = additional_attributes[i] if additional_attributes else None
        distances.append((smile, DataStructs.FingerprintSimilarity(fp, fp2), add_attr))

    sorted_by_second = sorted(distances, key=lambda tup: tup[1])
    return sorted_by_second


def mcs_rank(
    smiles_reference, smiles, additional_attributes=None
):  # pylint:disable=too-many-locals
    """Rank SMILES based on the maximum common substructure to the reference smiles."""
    distances = []
    mol = Chem.MolFromSmiles(smiles_reference)
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    for i, smile in enumerate(smiles):
        mol2 = Chem.MolFromSmiles(smile)
        res = rdFMCS.FindMCS([mol, mol2], completeRingsOnly=True, ringMatchesRingOnly=True)
        smarts = res.smartsString
        bond_diff = num_bonds - res.numBonds
        atom_diff = num_atoms - res.numAtoms
        add_attr = additional_attributes[i] if additional_attributes else None
        distances.append((smile, smarts, bond_diff, atom_diff, add_attr))

    sorted_by_atom_diff = sorted(distances, key=lambda tup: tup[-2])
    return sorted_by_atom_diff
