import torch.nn as nn
import torch
from typing import Dict, Optional

from src.kernels import gaussian

class Encoder(nn.Module):

    def __init__(
        self,
        dataset_size: int,
        latent_dim: int
    ) -> None:

        super(Encoder, self).__init__()

        self.alpha = nn.Parameter(torch.empty(size=(latent_dim, dataset_size)))
        nn.init.xavier_uniform_(self.alpha)

        self.dataset: Optional[torch.tensor] = None
        self.hash_gram: Dict[int, torch.tensor] = {}

    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        
        key = hash((x.numpy().tobytes(), self.dataset.numpy().tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = torch.tensor(gaussian(x.numpy(), self.dataset.numpy()))
        K = self.hash_gram[key]

        return torch.mm(K, self.alpha.t())

        