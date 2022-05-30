import os
import torch
import torch.nn as nn
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

    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:

        return self.decode(self.encode(x))

    def _train(
        self,
        X_train,
        epochs: int
    ) -> None:

        self.centers_encoder = X_train
        losses = {'epoch': [], 'loss': []}

        for i in tqdm(range(epochs)):

            z = self.encode(x=X_train)
            self.centers_decoder = z
            reconstructed = self.decode(z)

            self.optimizer.zero_grad()
            loss = self.criterion(reconstructed, X_train)
            loss.backward()
            self.optimizer.step()

            losses['epoch'].append(i)
            losses['loss'].append(loss.detach)

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

    def prepare_logs_directories(
        self,
        save_dir: str
    ) -> None:

        self.save_dir = save_dir
        self.save_dir_model = os.path.join(self.save_dir, 'model')
        self.save_dir_ckpts = os.path.join(self.save_dir_model, 'ckpts')
        self.save_dir_images = os.path.join(self.save_dir, 'images')
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir_model, exist_ok=True)
        os.makedirs(self.save_dir_ckpts, exist_ok=True)
        os.makedirs(self.save_dir_images, exist_ok=True)


