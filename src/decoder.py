import torch.nn as nn
import torch
from typing import Dict, Optional

from src.kernels import gaussian

class Decoder(nn.Module):

    def __init__(
        self,
        dataset_size: int,
        feature_dim: int,
    ) -> None:

        super(Decoder, self).__init__()

        self.alpha = nn.Parameter(torch.empty(size=(dataset_size, feature_dim)))
        nn.init.xavier_uniform_(self.alpha)

        self.dataset: Optional[torch.tensor] = None
        self.hash_gram: Dict[int, torch.tensor] = {}

    def forward(
        self,
        z: torch.tensor
    ) -> torch.tensor:

        key = hash((z.numpy().tobytes(), self.dataset.numpy().tobytes()))
        if not key in self.hash_gram:
            self.hash_gram[key] = torch.tensor(gaussian(z.numpy(), self.dataset.numpy()))
        K = self.hash_gram[key]

        return torch.mm(K, self.alpha)