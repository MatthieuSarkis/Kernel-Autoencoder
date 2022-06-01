import json
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.utils import save_image
from tqdm import tqdm
from typing import Optional, Dict

from src.autoencoder_v2.kernels import gaussian

class AutoEncoder(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 20,
        number_centers: int = 20,
        sigma_encoder: float = 1.0,
        sigma_decoder: float = 1.0,
        regularization: float = 1.0,
        learning_rate: float = 1e-3,
        save_dir: str = './saved_models/autoencoder',
    ) -> None:

        super(AutoEncoder, self).__init__()

        self.sigma_encoder = sigma_encoder
        self.sigma_decoder = sigma_decoder
        self.regularization = regularization
        self.number_centers = number_centers

        self.alpha_encoder = nn.Parameter(torch.empty(size=(latent_dim, number_centers), dtype=torch.float32))
        self.alpha_decoder = nn.Parameter(torch.empty(size=(number_centers, feature_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.alpha_encoder)
        nn.init.xavier_uniform_(self.alpha_decoder)

        self.centers_encoder = nn.Parameter(torch.empty(size=(number_centers, feature_dim), dtype=torch.float32))
        self.centers_decoder = nn.Parameter(torch.empty(size=(number_centers, latent_dim), dtype=torch.float32))

        self.hash_gram: Dict[int, torch.tensor] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss(reduction='mean')

        self.prepare_logs_directories(save_dir=save_dir)

    def _train(
        self,
        X_train: torch.tensor,
        X_test: torch.tensor,
        epochs: int
    ) -> None:

        # Initializing the centers
        permutation = torch.randperm(X_train.shape[0])
        indices_encoder = permutation[0: self.number_centers]
        indices_decoder = permutation[0: self.number_centers]
        self.centers_encoder = nn.Parameter(X_train[indices_encoder])
        self.centers_decoder = nn.Parameter(self.encode(X_train)[indices_decoder])

        losses = {'epoch': [], 'train_loss': [], 'test_loss': []}

        for epoch in tqdm(range(epochs)):

            reconstructed = self.decode(self.encode(x=X_train))

            l2_norm = sum(p.pow(2.0).sum() for p in self.parameters())

            self.optimizer.zero_grad()
            train_loss = self.criterion(reconstructed, X_train) + self.regularization * l2_norm
            #test_loss = self.criterion(self.decode(self.encode(X_test)), X_test)
            train_loss.backward()
            self.optimizer.step()

            losses['epoch'].append(epoch)
            losses['train_loss'].append(train_loss.item())
            #losses['test_loss'].append(test_loss.item())

            with open(os.path.join(self.save_dir, 'losses.json'), 'w') as f:
                json.dump(losses, f,  indent=4)

            if epoch % 10 == 0:
                self.plot_images(
                    train_original_images=X_train[:64], 
                    test_original_images=X_test[:64],
                    epoch=epoch
                )

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
        train_original_images: torch.tensor,
        test_original_images: torch.tensor,
        epoch: int,
    ) -> None:

        image_size = int(train_original_images.shape[1]**0.5)

        train_reconstructed_images = self.decode(self.encode(train_original_images))
        train_original_images = train_original_images.view(train_original_images.shape[0], 1, image_size, image_size) * 255.0
        train_reconstructed_images = train_reconstructed_images.view(train_original_images.shape[0], 1, image_size, image_size) * 255.0

        test_reconstructed_images = self.decode(self.encode(test_original_images))
        test_original_images = test_original_images.view(test_original_images.shape[0], 1, image_size, image_size) * 255.0
        test_reconstructed_images = test_reconstructed_images.view(test_original_images.shape[0], 1, image_size, image_size) * 255.0

        with torch.no_grad():

            train_original_images_grid = torchvision.utils.make_grid(train_original_images)
            train_reconstructed_images_grid = torchvision.utils.make_grid(train_reconstructed_images)
            save_image(train_original_images_grid, os.path.join(self.save_dir_images, 'epoch={}_original_train.png'.format(epoch)))
            save_image(train_reconstructed_images_grid, os.path.join(self.save_dir_images, 'epoch={}_reconstructed_train.png'.format(epoch)))

            test_original_images_grid = torchvision.utils.make_grid(test_original_images)
            test_reconstructed_images_grid = torchvision.utils.make_grid(test_reconstructed_images)
            save_image(test_original_images_grid, os.path.join(self.save_dir_images, 'epoch={}_original_test.png'.format(epoch)))
            save_image(test_reconstructed_images_grid, os.path.join(self.save_dir_images, 'epoch={}_reconstructed_test.png'.format(epoch)))

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