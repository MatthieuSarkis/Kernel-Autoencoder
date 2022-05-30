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

import numpy as np
import torch
from schnetpack.data.atoms import AtomsData
from typing import Dict, List, Tuple

class Coulomb():
    r"""
    Class implementing the Coulomb representation of molecules. 
    """
      
    def __init__(
        self,
        flatten: bool = True,
        permute: bool = True,
    ) -> None:
        
        super(Coulomb, self).__init__()
        
        self.representer_name = "coulomb"
        self.max_size_molecule = 0
        self.flatten = flatten
        self.permute = permute

    def transform(
        self,
        dataset: AtomsData,
        label: List[str] = ['EAT'],
    ) -> Tuple[torch.tensor, torch.tensor]:
             
        self._set_max_size_molecule(dataset=dataset)  
    
        # Instanciating the Coulomb matrix tensor with correct padding, the label tensor and the extra properties tensor  
        shape_X = (len(dataset), self.max_size_molecule * (self.max_size_molecule + 1) // 2) if self.flatten else (len(dataset), self.max_size_molecule, self.max_size_molecule) 
        shape_y = (len(dataset), len(torch.cat([dataset[0][prop_name] for prop_name in label])))   # (number of molecules * dimension of label space)      
        X = torch.zeros(size=shape_X, dtype=torch.float)
        y = torch.zeros(size=shape_y, dtype=torch.float)
    
        for i in range(len(dataset)):
            
            print("Molecule configuration: {}/{}".format(i+1, len(dataset)))
            
            molecule = dataset[i]
            length_molecule = len(molecule['_atomic_numbers'])
            
            if self.flatten:
                X[i, :length_molecule*(length_molecule+1)//2] = torch.tensor(self._transform_single_molecule(molecule))
            else:
                X[i, :length_molecule, :length_molecule] = torch.tensor(self._transform_single_molecule(molecule))

            y[i] = torch.cat([molecule[prop_name] for prop_name in label])

        return X.numpy(), y.numpy()
             
    def _transform_single_molecule(
        self, 
        molecule: Dict[str, torch.tensor],
    ) -> Dict[Tuple, List]:
        r"""
        This function implements the main part of the algorithm. 
        It computes the Coulomb matrix of a given molecule.
        """
        
        xyz = molecule['_positions']
        Z = molecule['_atomic_numbers']
        molecule_length = len(Z)
        
        if self.permute:
            idx = torch.randperm(molecule_length)
            xyz = xyz[idx]
            Z = Z[idx]
         
        c = np.zeros((molecule_length, molecule_length), dtype=np.float32)
        
        for i in range(molecule_length):
            for j in range(molecule_length):
                
                if i == j:
                    c[i, j] = 0.5 * Z[i] ** 2.4
                    
                else:
                    distance = np.linalg.norm(xyz[i] - xyz[j])
                    c[i, j] = Z[i] * Z[j] / distance

        return c[np.triu_indices(molecule_length)] if self.flatten else c

    def _set_max_size_molecule(
        self,
        dataset: AtomsData,
    ) -> None:
        
        print("(Coulomb) Finding max size molecule.")

        self.max_size_molecule = 0
        
        for i in range(len(dataset)):
            
            molecule_length = len(dataset[i]['_atomic_numbers'])
            
            if molecule_length > self.max_size_molecule:
                self.max_size_molecule = molecule_length