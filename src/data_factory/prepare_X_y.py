# -*- coding: utf-8 -*-
#
# Written by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
from argparse import ArgumentParser
import numpy as np
from schnetpack.data import AtomsData
from typing import List

from src.data_factory.coulomb import Coulomb
from src.data_factory.atomic_data import AVAILABLE_PROPERTIES_QM7X, AVAILABLE_PROPERTIES_QM9

def prepare_X_y(
    dataset_path: str,
    label: List[str]
) -> None:

    print("\nGenerating data...")
    
    dataset = AtomsData(dataset_path)
    representer = Coulomb(flatten=True, permute=True)

    X, y = representer.transform(dataset=dataset, label=label)

    X_y_directory = os.path.join('/'.join(dataset_path.split('/')[:-2]), 'X_y')
    os.makedirs(X_y_directory, exist_ok=True)

    # Save the numpy array as binary files
    with open(os.path.join(X_y_directory , dataset_path.split('/')[-1].split('.')[0]+'_{}_X.npy'.format(representer.representer_name)), 'wb') as f: 
        np.save(f, X)
    with open(os.path.join(X_y_directory , dataset_path.split('/')[-1].split('.')[0]+'_{}_y.npy'.format(representer.representer_name)), 'wb') as f: 
        np.save(f, y)  

    print("\nData generated.\n")

def main(args) -> None:

    dataset = args.dataset_path.split('/')[2]

    if dataset == 'qm7x':
        label = AVAILABLE_PROPERTIES_QM7X[0:28]+['number_of_atoms'] # 'SRMSD', 'SMIT', 'EPBE0MBD',  'EDFTBMBD', 'EAT', 'EPBE0', 'EMBD', 'ETS',  'ENN',  'EKIN',  'ENE',  'EEE',  'EXC',  'EX',  'EC',  'EXX',  'EKSE',   'EH',  'EL',  'HLGAP', 'DIP', 'VDIP',  'VTQ',  'VIQ',  'VEQ',  'C6', 'POL', 'MTPOL', 'number_of_atoms'
    else:
        label = [AVAILABLE_PROPERTIES_QM9[i] for i in [5, 6, 7, 10, 11, 12, 13, 14]]+['number_of_atoms'] # 'homo', 'lumo', 'gap', 'energy_U0', 'energy_U', 'enthalpy_H', 'free_energy', 'heat_capacity', 'number_of_atoms'

    prepare_X_y(dataset_path=args.dataset_path, label=label)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./data/qm7x/db/full_20conf_noH.db")
    args = parser.parse_args()
    main(args)   
