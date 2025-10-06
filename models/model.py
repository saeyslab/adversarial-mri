import torch

from pathlib import Path

class Model(torch.nn.Module):
    def __init__(self, weight_path: Path, device: torch.device = torch.device('cpu'), **kwargs):
        """
        Arguments:
            weight_path (Path): Path to the model weights.
            device (torch.device): Device to store the model on.
        """

        super().__init__(**kwargs)
        self.weight_path: Path = weight_path
        self.device = device

    def forward(self, images: torch.tensor) -> torch.tensor:
        pass
