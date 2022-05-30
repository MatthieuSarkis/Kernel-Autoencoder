from datetime import datetime
import numpy as np
import os
import torch

from src.autoencoder.autoencoder import AutoEncoder
from src.autoencoder.config import config

def load_data(
    data_path: str,
    dataset_size: int
) -> torch.tensor:

    with open(data_path, 'rb') as f:
        X = np.load(f)
    
    idx = np.random.permutation(X.shape[0])
    X = X[idx][:dataset_size]

    return torch.tensor(X, dtype=torch.float32)

def main(config: dict) -> None:

    save_directory = os.path.join("saved_models", datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
    os.makedirs(save_directory, exist_ok=True)

    X = load_data(
        data_path=config['data_path'],
        dataset_size=config['dataset_size']
    )

    autoencoder = AutoEncoder(
        dataset_size=X.shape[0], 
        feature_dim=X.shape[1],
        latent_dim=config['latent_dim'],
        sigma_encoder=config['sigma_encoder'],
        sigma_decoder=config['sigma_decoder'],
        learning_rate=config['learning_rate'],
        save_dir=save_directory
    )

    autoencoder._train(
        X_train=X, 
        epochs=config['epochs']
    )


if __name__ == '__main__':

    main(config=config)