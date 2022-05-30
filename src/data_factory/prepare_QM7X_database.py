# -*- coding: utf-8 -*-
#
# This code was taken from https://zenodo.org/record/4288677#.YaDRj7so9H6
# and refactored by Matthieu Sarkis, https://github.com/MatthieuSarkis
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from argparse import ArgumentParser
from ase.atoms import Atoms
import h5py
import numpy as np
import os
import schnetpack
from typing import List

from src.data_factory.atomic_data import AVAILABLE_PROPERTIES_QM7X

def removing_duplicates(IDs):
    """function to exclude duplicates from QM7-X dataset"""

    DupMols = []
    for line in open('data/qm7x/hdf5/DupMols.dat', 'r'):
        DupMols.append(line.rstrip('\n'))

    for IDconf in IDs:
        if IDconf in DupMols:
            IDs.remove(IDconf)
            stmp = IDconf[:-3]
            for ii in range(1,101):
                IDs.remove(stmp+'d'+str(ii))

    return IDs

def generate_db(
    data_path: str,
    database_name: str,
    available_properties: List[str],
    set_ids: List[str],
    max_number_configs: int = None,
    max_number_atoms: int = None,
    eliminate_hydrogen: bool = False,
    remove_duplicates: bool = True,
) -> None:
    """
    -'atNUM': Atomic numbers (N)
    -'atXYZ': Atoms coordinates [Ang] (Nx3)

    -'sRMSD': RMSD to optimized structure [Ang] (1)
    -'sMIT': Momente of inertia tensor [amu.Ang^2] (9)
    -'ePBE0+MBD': Total PBE0+MBD energy [eV] (1)
    -'eDFTB+MBD': Total DFTB+MBD energy [eV] (1)
    -'eAT': PBE0 atomization energy [eV] (1)
    -'ePBE0': PBE0 energy [eV] (1)
    -'eMBD': MBD energy [eV] (1)
    -'eTS': TS dispersion energy [eV] (1)
    -'eNN': Nuclear-nuclear repulsion energy [eV] (1)
    -'eKIN': Kinetic energy [eV] (1)
    -'eNE': Nuclear-electron attracttion [eV] (1)
    -'eEE': Classical coulomb energy (el-el) [eV] (1)
    -'eXC': Exchange-correlation energy [eV] (1)
    -'eX': Exchange energy [eV] (1)
    -'eC': Correlation energy [eV] (1)
    -'eXX': Exact exchange energy [eV] (1)
    -'eKSE': Sum of Kohn-Sham eigenvalues [eV] (1)
    -'eH': HOMO energy [eV] (1)
    -'eL': LUMO energy [eV] (1)
    -'HLgap': HOMO-LUMO gap [eV] (1)
    -'DIP': Total dipole moment [e.Ang] (1)
    -'vDIP': Dipole moment components [e.Ang] (3)
    -'vTQ': Total quadrupole moment components [e.Ang^2] (3)
    -'vIQ': Ionic quadrupole moment components [e.Ang^2] (3)
    -'vEQ': Electronic quadrupole moment components [eAng^2] (3)
    -'mC6': Molecular C6 coefficient [hartree.bohr^6] (computed using SCS) (1)
    -'mPOL': Molecular polarizability [bohr^3] (computed using SCS) (1)
    -'mTPOL': Molecular polarizability tensor [bohr^3] (9)

    -'KSE': Kohn-Sham eigenvalues [eV] (depends on the molecule)
    -'totFOR': Total PBE0+MBD atomic forces (unitary forces cleaned) [eV/Ang] (Nx3)
    -'vdwFOR': MBD atomic forces [eV/Ang] (Nx3)
    -'pbe0FOR': PBE0 atomic forces [eV/Ang] (Nx3)
    -'hVOL': Hirshfeld volumes [bohr^3] (N)
    -'hRAT': Hirshfeld ratios (N)
    -'hCHG': Hirshfeld charges [e] (N)
    -'hDIP': Hirshfeld dipole moments [e.bohr] (N)
    -'hVDIP': Components of Hirshfeld dipole moments [e.bohr] (Nx3)
    -'atC6': Atomic C6 coefficients [hartree.bohr^6] (N)
    -'atPOL': Atomic polarizabilities [bohr^3] (N)
    -'vdwR': van der Waals radii [bohr] (N)
    """

    path = os.path.join(data_path, 'db')
    os.makedirs(path, exist_ok=True)

    dataset = schnetpack.data.AtomsData(
        os.path.join(data_path, 'db', database_name), 
        available_properties=available_properties
    )

    # Buffers for molecules and properties
    atom_buffer = []
    property_buffer = []

    for setid in set_ids:
        
        fMOL = h5py.File(os.path.join(data_path, 'hdf5', '{}.hdf5'.format(setid)), 'r')
        
        # get IDs of HDF5 files and loop through
        mol_ids = list(fMOL.keys())
        
        for molid in mol_ids:
            
            print('Current molecule: ' + molid)
            
            # Each molecule comes in various configurations of its atoms. 
            conf_ids = list(fMOL[molid].keys())
            
            # We ignore molecules with to many atoms.
            if max_number_atoms is not None:
                Z = np.array(fMOL[molid][conf_ids[-1]]['atNUM'])
                if eliminate_hydrogen:
                    if Z[Z>1].size > max_number_atoms:
                        continue
                else:
                    if Z.size > max_number_atoms:
                        continue
            
            # We remove the duplicates.
            if remove_duplicates:
                conf_ids = removing_duplicates(conf_ids)
                
            # We select max_number_configs evenly spaced configurations,
            # including the equilibrium (the last in the list), 
            # in order to reduce the correlations as much as possible.
            if max_number_configs is not None:
                conf_ids = np.array(conf_ids)
                idx = np.round(np.linspace(0, len(conf_ids) - 1, max_number_configs - 1)).astype(int)
                conf_ids = [conf_ids[-1]] + list(conf_ids[idx])
        
            for confid in conf_ids:
                
                # Cartesion coordinates and atomic charges of atoms in the molecule.
                xyz = np.array(fMOL[molid][confid]['atXYZ'])
                Z = np.array(fMOL[molid][confid]['atNUM'])
                
                if eliminate_hydrogen:
                    xyz = xyz[Z>1]
                    Z = Z[Z>1]

                atom_buffer.append(Atoms(Z, xyz))
                
                # The dimension of the following properties is independent of the number of atoms.
                SRMSD = float(list(fMOL[molid][confid]['sRMSD'])[0])
                SMIT = list(fMOL[molid][confid]['sMIT'])
                EPBE0MBD = float(list(fMOL[molid][confid]['ePBE0+MBD'])[0])
                EDFTBMBD = float(list(fMOL[molid][confid]['eDFTB+MBD'])[0])
                EAT = float(list(fMOL[molid][confid]['eAT'])[0])    
                EPBE0 = float(list(fMOL[molid][confid]['ePBE0'])[0])
                EMBD = float(list(fMOL[molid][confid]['eMBD'])[0])
                ETS = float(list(fMOL[molid][confid]['eTS'])[0])
                ENN = float(list(fMOL[molid][confid]['eNN'])[0])
                EKIN = float(list(fMOL[molid][confid]['eKIN'])[0])
                ENE = float(list(fMOL[molid][confid]['eNE'])[0])
                EEE = float(list(fMOL[molid][confid]['eEE'])[0])
                EXC = float(list(fMOL[molid][confid]['eXC'])[0])
                EX = float(list(fMOL[molid][confid]['eX'])[0])
                EC = float(list(fMOL[molid][confid]['eC'])[0])
                EXX = float(list(fMOL[molid][confid]['eXX'])[0])
                EKSE = float(list(fMOL[molid][confid]['eKSE'])[0])
                EH = float(list(fMOL[molid][confid]['eH'])[0])
                EL = float(list(fMOL[molid][confid]['eL'])[0])
                HLGAP = float(list(fMOL[molid][confid]['HLgap'])[0])
                DIP = float(list(fMOL[molid][confid]['DIP'])[0])
                VDIP = list(fMOL[molid][confid]['vDIP'])
                VTQ = list(fMOL[molid][confid]['vTQ'])
                VIQ = list(fMOL[molid][confid]['vIQ'])
                VEQ = list(fMOL[molid][confid]['vEQ'])
                C6 = float(list(fMOL[molid][confid]['mC6'])[0])
                POL = float(list(fMOL[molid][confid]['mPOL'])[0])
                MTPOL = list(fMOL[molid][confid]['mTPOL'])

                # The dimension of the following properties scales with the number of atoms.
                KSE = float(list(fMOL[molid][confid]['KSE'])[0])
                FORCE = list(fMOL[molid][confid]['totFOR'])
                VDWFOR = list(fMOL[molid][confid]['vdwFOR']) 
                PBE0FOR = list(fMOL[molid][confid]['pbe0FOR'])
                HVOL = list(fMOL[molid][confid]['hVOL'])
                HRAT = list(fMOL[molid][confid]['hRAT'])
                HCHG = list(fMOL[molid][confid]['hCHG'])
                HDIP = list(fMOL[molid][confid]['hDIP'])
                HVDIP = list(fMOL[molid][confid]['hVDIP'])
                ATC6 = list(fMOL[molid][confid]['atC6'])
                ATPOL = list(fMOL[molid][confid]['atPOL'])
                VDWR = list(fMOL[molid][confid]['vdwR'])

                property_buffer.append({
                    'SRMSD': np.array([SRMSD]),
                    'SMIT': np.array(SMIT),
                    'EPBE0MBD': np.array([EPBE0MBD]),
                    'EDFTBMBD': np.array([EDFTBMBD]),
                    'EAT': np.array([EAT]),
                    'EPBE0': np.array([EPBE0]),
                    'EMBD': np.array([EMBD]),
                    'ETS': np.array([ETS]),
                    'ENN': np.array([ENN]),
                    'EKIN': np.array([EKIN]),
                    'ENE': np.array([ENE]),
                    'EEE': np.array([EEE]),
                    'EXC': np.array([EXC]),
                    'EX': np.array([EX]),
                    'EC': np.array([EC]),
                    'EXX': np.array([EXX]),
                    'EKSE': np.array([EKSE]),
                    'EH': np.array([EH]),
                    'EL': np.array([EL]),
                    'HLGAP': np.array([HLGAP]),
                    'DIP': np.array([DIP]),
                    'VDIP': np.array(VDIP),
                    'VTQ': np.array(VTQ),
                    'VIQ': np.array(VIQ),
                    'VEQ': np.array(VEQ),
                    'C6': np.array([C6]),
                    'POL': np.array([POL]),
                    'MTPOL': np.array(MTPOL),
                    'KSE': np.array([KSE]),
                    'FORCE': np.array(FORCE),
                    'VDWFOR': np.array(VDWFOR), 
                    'PBE0FOR': np.array(PBE0FOR),
                    'HVOL': np.array(HVOL),
                    'HRAT': np.array(HRAT),
                    'HCHG': np.array(HCHG),
                    'HDIP': np.array(HDIP),
                    'HVDIP': np.array(HVDIP),
                    'ATC6': np.array(ATC6),
                    'ATPOL': np.array(ATPOL),
                    'VDWR': np.array(VDWR),
                    'number_of_atoms': np.array([Z.size])
                })

    # Writting the database .db file.        
    print('=====================================================')
    print('  Gathering data done. Writing database...')
    dataset.add_systems(atom_buffer, property_buffer)
    print('Database "' + database_name + '" written.')
    print('=====================================================')

def main(args):

    data_path = 'data/qm7x'
    database_name = args.database_name

    max_number_configs = args.max_number_configs
    max_number_atoms = args.max_number_atoms

    # Various .hdf5 files:
    set_ids = ['1000', '2000', '3000', '4000', '5000', '6000', '7000', '8000']
    #set_ids = ['8000']
    
    generate_db(
        data_path=data_path,
        database_name=database_name,
        available_properties=AVAILABLE_PROPERTIES_QM7X+['number_of_atoms'],
        set_ids=set_ids,
        max_number_configs=max_number_configs,
        max_number_atoms=max_number_atoms,
        eliminate_hydrogen=args.eliminate_hydrogen,
        remove_duplicates=True
    )

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--database_name', type=str, default='equil_noH.db')
    parser.add_argument("--max_number_configs", type=int)
    parser.add_argument("--max_number_atoms", type=int)
    parser.add_argument('--eliminate_hydrogen', dest='eliminate_hydrogen', action='store_true')
    parser.add_argument('--no-eliminate_hydrogen', dest='eliminate_hydrogen', action='store_false')
    parser.set_defaults(eliminate_hydrogen=False)

    args = parser.parse_args()
    main(args) 


