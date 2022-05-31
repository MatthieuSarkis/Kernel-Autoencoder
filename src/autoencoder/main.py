from argparse import ArgumentParser
from datetime import datetime
import numpy as np
import os
import torch

from src.autoencoder.autoencoder import AutoEncoder

def load_data(
    data_path: str,
    dataset_size: int
) -> torch.tensor:

    with open(data_path, 'rb') as f:
        X = np.load(f)
    
    idx = np.random.permutation(X.shape[0])
    X = X[idx][:dataset_size] / 255.0
    X = X.reshape((dataset_size, -1))

    return torch.tensor(X, dtype=torch.float32)

def main(args) -> None:

    config = {
        'data_path': os.path.join('./data', args.dataset, 'train_images.npy'),
        'dataset_size': 5000,
        'latent_dim': 20,
        'sigma_encoder': 1e-2,
        'sigma_decoder': 1e-2,
        'learning_rate': 1e-3,
        'epochs': 150,
    }

    save_directory = os.path.join("saved_models", args.dataset, datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))
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

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['mnist', 'fashion_mnist'], default='mnist')
    args = parser.parse_args()

    main(args)