import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Optional, Dict

from src.autoencoder.kernels import gaussian

class AutoEncoder(nn.Module):

    def __init__(
        self,
        dataset_size: int,
        feature_dim: int,
        latent_dim: int = 20,
        sigma_encoder: float = 1.0,
        sigma_decoder: float = 1.0,
        learning_rate: float = 1e-3,
        save_dir: str = './saved_models/autoencoder',
    ) -> None:

        super(AutoEncoder, self).__init__()

        self.sigma_encoder = sigma_encoder
        self.sigma_decoder = sigma_decoder

        self.alpha_encoder = nn.Parameter(torch.empty(size=(latent_dim, dataset_size), dtype=torch.float32))
        self.alpha_decoder = nn.Parameter(torch.empty(size=(dataset_size, feature_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.alpha_encoder)
        nn.init.xavier_uniform_(self.alpha_decoder)

        self.centers_encoder: Optional[torch.tensor] = None
        self.centers_decoder: Optional[torch.tensor] = None
        self.hash_gram: Dict[int, torch.tensor] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')

        self.prepare_logs_directories(save_dir=save_dir)

    def _train(
        self,
        X_train,
        epochs: int
    ) -> None:

        self.centers_encoder = X_train
        losses = {'epoch': [], 'loss': []}

        for epoch in tqdm(range(epochs)):

            z = self.encode(x=X_train)
            self.centers_decoder = z
            reconstructed = self.decode(z)

            self.optimizer.zero_grad()
            loss = self.criterion(reconstructed, X_train)
            loss.backward()
            self.optimizer.step()

            losses['epoch'].append(epoch)
            losses['loss'].append(loss.detach)

            if epoch % 10 == 0:
                self.plot_images(original_images=X_train[:64], epoch=epoch)

        torch.save(self, os.path.join(self.save_dir_model, 'model.pt'))

    def encode(
        self,
        x: torch.tensor
    ) -> torch.tensor:

        key = hash((x.detach().numpy().tobytes(), self.centers_encoder.detach().numpy().tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = torch.tensor(gaussian(X1=x.detach().numpy(), X2=self.centers_encoder.detach().numpy(), sigma=self.sigma_encoder), dtype=torch.float32)
        K = self.hash_gram[key]
        return torch.mm(K, self.alpha_encoder.t())

    def decode(
        self,
        z: torch.tensor
    ) -> torch.tensor:

        key = hash((z.detach().numpy().tobytes(), self.centers_decoder.detach().numpy().tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = torch.tensor(gaussian(X1=z.detach().numpy(), X2=self.centers_decoder.detach().numpy(), sigma=self.sigma_decoder), dtype=torch.float32)
        K = self.hash_gram[key]
        return torch.mm(K, self.alpha_decoder)

    def plot_images(
        self,
        original_images: torch.tensor,
        epoch: int
    ) -> None:

        image_size = int(original_images.shape[1]**0.5)

        reconstructed_images = self.decode(self.encode(original_images))
        original_images = original_images.view(original_images.shape[0], 1, image_size, image_size) * 255.0
        reconstructed_images = reconstructed_images.view(original_images.shape[0], 1, image_size, image_size) * 255.0

        with torch.no_grad():
            original_images_grid = torchvision.utils.make_grid(original_images)
            reconstructed_images_grid = torchvision.utils.make_grid(reconstructed_images)
            save_image(original_images_grid, os.path.join(self.save_dir_images, 'epoch={}_original.png'.format(epoch)))
            save_image(reconstructed_images_grid, os.path.join(self.save_dir_images, 'epoch={}_reconstructed.png'.format(epoch)))

    def prepare_logs_directories(
        self,
        save_dir: str
    ) -> None:

        self.save_dir = save_dir
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        self.save_dir_images = os.path.join(self.save_dir, 'images')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_model, exist_ok=True)
        os.makedirs(self.save_dir_images, exist_ok=True)


