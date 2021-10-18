# -*- coding: utf-8 -*-
"""Collection of descriptor methods"""
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
from EFGs import mol2frag
from pymatgen.analysis.local_env import LocalStructOrderParams
from pymatgen.core import Element
from rdkit.Chem import Descriptors, GraphDescriptors, rdMolDescriptors

from .flexibility import kier_molecular_flexibility, n_conf20

__all__ = [
    "get_lsop",
    "rdkit_descriptors",
    "chemistry_descriptors",
    "distance_descriptors",
    "ALL_LSOP",
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


def try_except_nan(mol, calculator, exception_value=np.nan):
    try:
        value = calculator(mol)
    except Exception:  # pylint:disable=broad-except
        value = exception_value
    return value


@lru_cache()
def rdkit_descriptors(rdkit_mol, three_dimensional: bool = True):
    descriptors = {
        "min_partial_charge": try_except_nan(rdkit_mol, Descriptors.MinPartialCharge),
        "max_partial_charge": try_except_nan(rdkit_mol, Descriptors.MaxPartialCharge),
        "tpsa": try_except_nan(rdkit_mol, rdMolDescriptors.CalcTPSA),
        "kier_molecular_flexibility": try_except_nan(
            rdkit_mol, kier_molecular_flexibility
        ),
        "bertz_ct": try_except_nan(rdkit_mol, GraphDescriptors.BertzCT),
        "kappa_1": try_except_nan(rdkit_mol, GraphDescriptors.Kappa1),
        "kappa_2": try_except_nan(rdkit_mol, GraphDescriptors.Kappa2),
        "kappa_3": try_except_nan(rdkit_mol, GraphDescriptors.Kappa3),
        "fragments": try_except_nan(rdkit_mol, get_fragments, []),
    }

    if three_dimensional:
        try:
            from rdkit_utils import conformers  # pylint:disable=import-outside-toplevel

            engine = conformers.ConformerGenerator(max_conformers=20)
            mol = engine.generate_conformers(rdkit_mol)
        except ImportError:
            warnings.warn(
                "rdkit_utils is required to create conformers\
                     for 3D descriptor calculation."
            )
        except Exception:  # pylint:disable=broad-except
            mol = rdkit_mol
        descriptors["spherocity"] = try_except_nan(
            mol, rdMolDescriptors.CalcSpherocityIndex
        )
        descriptors["eccentricity"] = try_except_nan(
            mol, rdMolDescriptors.CalcEccentricity
        )
        descriptors["radius_of_gyration"] = try_except_nan(
            mol, rdMolDescriptors.CalcRadiusOfGyration
        )
        descriptors["inertial_shape_factor"] = try_except_nan(
            mol, rdMolDescriptors.CalcInertialShapeFactor
        )
        descriptors["npr1"] = try_except_nan(mol, rdMolDescriptors.CalcNPR1)
        descriptors["npr2"] = try_except_nan(mol, rdMolDescriptors.CalcNPR2)
        descriptors["sphericity"] = descriptors["npr1"] + descriptors["npr2"] - 1

        descriptors["rod_likeness"] = descriptors["npr2"] - descriptors["npr1"]

        descriptors["disc_likeness"] = 2 - 2 * descriptors["npr2"]
        descriptors["plane_best_fit"] = try_except_nan(mol, rdMolDescriptors.CalcPBF)
        descriptors["n_conf20"] = try_except_nan(mol, n_conf20)
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
    for site in structure:
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
        for i, _ in enumerate(structure):
            for j, _ in enumerate(structure):
                if i < j:
                    distances.append(structure.get_distance(i, j))

        return {
            "min_distance": np.min(distances),
            "max_distance": np.max(distances),
            "mean_distance": np.mean(distances),
            "median_distance": np.mean(distances),
            "std_distance": np.std(distances),
        }
    except Exception:  # pylint: disable=broad-except
        return {
            "min_distance": 0,
            "max_distance": 0,
            "mean_distance": 0,
            "median_distance": 0,
            "std_distance": 0,
        }


def get_fragments(rdkit_mol):
    frag_a, frag_b = mol2frag(rdkit_mol)
    return frag_a + frag_b
