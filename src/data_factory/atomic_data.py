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

ATOMIC_CHARGES = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    16: 'S',
    17: 'Cl'
}

# atom energies
EPBE0_atom = {
    6: -1027.592489146, 
    17: -12516.444619523, 
    1: -13.641404161,
    7: -1484.274819088, 
    8: -2039.734879322, 
    16: -10828.707468187
}

AVAILABLE_PROPERTIES_QM7X = [
    'SRMSD',
    'SMIT',
    'EPBE0MBD', 
    'EDFTBMBD',
    'EAT',
    'EPBE0',
    'EMBD',
    'ETS', 
    'ENN', 
    'EKIN', 
    'ENE', 
    'EEE', 
    'EXC', 
    'EX', 
    'EC', 
    'EXX', 
    'EKSE',  
    'EH', 
    'EL', 
    'HLGAP',
    'DIP',
    'VDIP', 
    'VTQ', 
    'VIQ', 
    'VEQ', 
    'C6',
    'POL',
    'MTPOL', 
    'KSE',
    'FORCE',
    'VDWFOR', 
    'PBE0FOR',
    'HVOL',
    'HRAT',
    'HCHG',
    'HDIP',
    'HVDIP',
    'ATC6',
    'ATPOL',
    'VDWR'
]

AVAILABLE_PROPERTIES_QM9 = [
    'rotational_constant_A', 
    'rotational_constant_B', 
    'rotational_constant_C', 
    'dipole_moment', 
    'isotropic_polarizability', 
    'homo', 
    'lumo', 
    'gap', 
    'electronic_spatial_extent', 
    'zpve', 
    'energy_U0', 
    'energy_U', 
    'enthalpy_H', 
    'free_energy', 
    'heat_capacity'
]

DATASET_SIZE_QM9 = 133885