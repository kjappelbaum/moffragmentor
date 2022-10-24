from moffragmentor.sbu.sbucollection import SBUCollection

def test_sbu_collection(get_sbu_collection):
    sbu_collection = get_sbu_collection
    assert len(sbu_collection.sbus) == 2
    assert sbu_collection.sbu_types == [0, 1]
    assert sbu_collection.unique_sbus == {'[H]C([H])([H])[H]', '[H]C([H])([H])C([H])([H])[H]'}
    assert sbu_collection.composition == {'H': 8, 'C': 2}
    assert sbu_collection.molar_masses == [16.04303, 30.06904]
    assert sbu_collection.coordination_numbers == [4, 6]
    assert sbu_collection.sbu_properties == {0: 4, 1: 6}
    assert sbu_collection.sbu_smiles == ['[H]C([H])([H])[H]', '[H]C([H])([H])C([H])([H])[H]']