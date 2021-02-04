# -*- coding: utf-8 -*-
import os
from moffragmentor import MOF 
from pymatgen import Structure


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
def check_mof_creation_from_cif():
    mof = MOF.from_cif(os.path.join(THIS_DIR, 'test_files', 'HKUST-1.cif'))
    assert isinstance(mof, object)  
    assert isinstance(mof.structure, Structure)

def test_topology_computation(): 
    mof = MOF.from_cif(os.path.join(THIS_DIR, 'test_files', 'HKUST-1.cif'))
    assert mof.topology == 'tbo'