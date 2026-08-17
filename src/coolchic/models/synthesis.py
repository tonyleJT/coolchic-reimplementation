import torch
import torch.nn as nn
import torch.nn.functional as F

class SynthesisMLP(nn.Module):
    """
    COOL-CHIC synthesis network.

    Each pixel receives a 7-dimensional feature vector from the
    hierarchical latent representation and maps it to RGB.
    """
    '''
    @brief Initialize the MLP Cool-Chic synthesis network.
    @Architecture: 7 -> 12 -> 12 -> 3.
    '''

    def __init__(self) -> None:
        super().__init__()
        # with bias (as output neurons): layer1: 7*12 + 12 = 96,
        # layer2: 12*12 + 12 = 156,
        # layer3: 12*3 + 3 = 39
        self.fc1 = nn.Linear(in_features=7, out_features=12)
        self.fc2 = nn.Linear(in_features=12, out_features=12)
        self.fc3 = nn.Linear(in_features=12, out_features=3)


    '''
    @brief Convert a dense seven-channel latent representation to RGB.
    - nn.Linear(7, 12) expects the last dim to contain the 7 features  
    
    @param[in] latent Dense latent tensor with shape [B, 7, H, W].
    @return RGB reconstruction tensor with shape [B, 3, H, W].
    '''

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B, 7, H, W], got {tuple(latent.shape)}."
            )

        if latent.shape[1] != 7:
            raise ValueError(
                f"Expected 7 latent channels, got {latent.shape[1]}."
            )

        batch_size, _, height, width = latent.shape
        # [B, 7, H, W] -> [B, H, W, 7] -> [B*H*W, 7]
        features = latent.permute(0, 2, 3, 1).reshape(-1, 7)

        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))
        rgb = self.fc3(features)  # [B*H*W, 3] RGB representation

        # [B*H*W, 3] -> [B, H, W, 3] -> [B, 3, H, W]
        reconstruction = rgb.reshape(
            batch_size,
            height,
            width,
            3,
        ).permute(0, 3, 1, 2)

        return reconstruction.contiguous()

