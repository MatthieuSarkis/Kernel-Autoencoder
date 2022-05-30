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

import math
import numpy as np
from schnetpack.data.atoms import AtomsData
from typing import Optional

from src.data_factory.atomic_data import ATOMIC_CHARGES

# Data preprocessing
def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float = 0.2,
) -> tuple:
    
    dataset_size = y.shape[0]
    
    idx = np.random.permutation(X.shape[0])
    X = X[idx]
    y = y[idx]
    
    split_idx = int(dataset_size * (1 - validation_fraction))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    return X_train, y_train, X_val, y_val

class Normalizer():
    r"""the transformed data is in the range [0, 2\pi]"""

    def __init__(
        self
    ) -> None:

        self.min: Optional[float] = None
        self.max: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
    ) -> None:
        
        self.min = X.min()
        self.max = X.max()

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        
        X = 2 * math.pi * (X - self.min) / (self.max - self.min)

        return X

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        self.fit(X)
        X = self.transform(X)

        return X

    def inverse_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        
        X = (self.max - self.min) * X / (2 * math.pi) + self.min

        return X

class Standardizer():

    def __init__(
        self
    ) -> None:

        self.mu = None
        self._sigma = None

    def fit(
        self,
        X: np.array,
    ) -> None:
        
        self.mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)

    def transform(
        self,
        X: np.array,
    ) -> np.ndarray:
        
        X = (X - self.mu) / self._sigma

        return X

    def fit_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        self.fit(X)
        X = self.transform(X)

        return X

    def inverse_transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        
        X = self._sigma * X + self.mu

        return X

# Loss functions   
def rmae(
    y_true: np.ndarray, 
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean(np.abs((y_true - y_predicted) /  y_true))

def mse(
    y_true: np.ndarray, 
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean((y_true - y_predicted)**2)

def rmse(
    y_true: np.ndarray, 
    y_predicted: np.ndarray,
) -> np.ndarray:

    return 1 / y_true.shape[0] * np.sqrt(np.sum((y_true - y_predicted)**2))

def percentage_mae(
    y_true: np.ndarray, 
    y_predicted: np.ndarray,
) -> np.ndarray:

    return np.mean(np.abs(y_true - y_predicted) / (np.abs(y_true) + np.abs(y_predicted)))

def list_molecules_in_dataset(
    dataset: AtomsData,
    number_configs_per_molecule: int = 100,
) -> None:
    r"""Simply outputs the list of molecules present in the given dataset."""

    for i in range(1, len(dataset)):
        if i % number_configs_per_molecule == 0:
            print(i//number_configs_per_molecule, ''.join([ATOMIC_CHARGES[int(z)] for z in dataset[i]['_atomic_numbers']]))
 
