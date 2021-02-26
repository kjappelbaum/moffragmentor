# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pymatgen.analysis.local_env import LocalStructOrderParams
from pymatgen.core import Element
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit_utils import conformers

__all__ = [
    "get_lsop",
    "rdkit_descriptors",
    "chemistry_descriptors",
    "distance_descriptors",
]

ALL_LSOP = [
    "cn",
    "sgl_bd",
    "bent",
    "tri_plan",
    "tri_plan_max",
    "reg_tri",
    "sq_plan",
    "sq_plan_max",
    "pent_plan",
    "pent_plan_max",
    "sq",
    "tet",
    "tet_max",
    "tri_pyr",
    "sq_pyr",
    "sq_pyr_legacy",
    "tri_bipyr",
    "sq_bipyr",
    "oct",
    "oct_legacy",
    "pent_pyr",
    "hex_pyr",
    "pent_bipyr",
    "hex_bipyr",
    "T",
    "cuboct",
    "cuboct_max",
    "see_saw_rect",
    "bcc",
    "q2",
    "q4",
    "q6",
    "oct_max",
    "hex_plan_max",
    "sq_face_cap_trig_pris",
]


def rdkit_descriptors(rdkit_mol, three_dimensional: bool = True):
    descriptors = {
        "min_partial_charge": Descriptors.MinPartialCharge(rdkit_mol),
        "max_partial_charge": Descriptors.MaxPartialCharge(rdkit_mol),
        "tpsa": rdMolDescriptors.CalcTPSA(rdkit_mol),
    }

    if three_dimensional:
        engine = conformers.ConformerGenerator(max_conformers=10)
        mol = engine.generate_conformers(rdkit_mol)
        descriptors["spherocity"] = rdMolDescriptors.CalcSpherocityIndex(mol)
        descriptors["eccentricity"] = rdMolDescriptors.CalcEccentricity(mol)
        descriptors["radius_of_gyration"] = rdMolDescriptors.CalcRadiusOfGyration(mol)
        descriptors["inertial_shape_factor"] = rdMolDescriptors.CalcInertialShapeFactor(
            mol
        )
        descriptors["npr1"] = rdMolDescriptors.CalcNPR1(mol)
        descriptors["npr2"] = rdMolDescriptors.CalcNPR2(mol)
        descriptors["sphericity"] = descriptors["npr1"] + descriptors["npr2"] - 1
        descriptors["rod_likeness"] = descriptors["npr2"] - descriptors["npr1"]
        descriptors["disc_likeness"] = 2 - 2 * descriptors["npr2"]
        descriptors["plane_best_fit"] = rdMolDescriptors.CalcPBF(mol)
    return descriptors


def summarize(df):
    descriptors = {}

    for col in df.columns:
        values = df[col]

        descriptors[col + "_mean"] = values.mean()
        descriptors[col + "_std"] = values.std()
        descriptors[col + "_max"] = values.max()
        descriptors[col + "_min"] = values.min()
        descriptors[col + "_median"] = values.median()

    return descriptors


def chemistry_descriptors(structure):
    """Calculate basic chemistry like electronegativity, EA"""
    results = []
    for i, site in enumerate(structure):
        results.append(get_chemistry_descriptors_for_site(site))

    return summarize(pd.DataFrame(results))


def get_chemistry_descriptors_for_site(site):
    elem = Element(str(site.specie))

    descriptors = {
        "X": elem.X,
        "number": elem.number,
        "row": elem.row,
        "group": elem.group,
        "atomic_radius": elem.atomic_radius,
        "vdw_radius": elem.van_der_waals_radius,
        "avg_ionic_radius": elem.average_ionic_radius,
        "avg_anionic_radius": elem.average_anionic_radius,
        "avg_cationic_radius": elem.average_cationic_radius,
        "mendeleev_no": elem.mendeleev_no,
    }

    return descriptors


def get_lsop(structure):
    if len(structure) > 1:
        lsop = LocalStructOrderParams(types=ALL_LSOP)

        descriptors = lsop.get_order_parameters(
            structure, 0, indices_neighs=np.arange(len(structure))
        )
    else:
        descriptors = np.zeros((len(ALL_LSOP),))
        descriptors[0] = 1

    return dict(zip(ALL_LSOP, descriptors))


def distance_descriptors(structure):
    try:
        distances = []
        for i, sitei in enumerate(structure):
            for j, sitej in enumerate(structure):
                if i < j:
                    distances.append(structure.get_distance(i, j))

        return {
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "mean_distance": np.mean(distances),
            "median_distance": np.mean(distances),
            "std_distance": np.std(distances),
        }
    except Exception:
        return {
            "min_distance": 0,
            "max_distance": 0,
            "mean_distance": 0,
            "median_distance": 0,
            "std_distance": 0,
        }