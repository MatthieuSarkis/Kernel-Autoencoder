import torch
import torch.nn as nn

from src.encoder import Encoder
from src.decoder import Decoder
from src.kernels import gaussian

class AutoEncoder(nn.Module):

    def __init__(
        self,
        latent_dim: int = 20,
        sigma_encoder: float = 1.0,
        sigma_decoder: float = 1.0
    ) -> None:

        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

        self.sigma_encoder = sigma_encoder
        self.sigma_decoder = sigma_decoder

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='sum')

    def _train(
        self,
        X_train,
        epochs: int
    ) -> None:

        loss_history = []
        self.encoder.dataset = X_train
        N = X_train.shape[0]

        for i in range(epochs):

            z = self.encoder(X_train)
            self.decoder.dataset = z
            x = self.decoder(z)



