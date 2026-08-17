import torch
import torch.nn.functional as F
from torch import nn

class CausalContext(nn.Module):
    # 12 causal spaial neighbors used by the COOL-CHIC probability model.

    CONTEXT_SIZE = 12
    CONTEXT_RADIUS = 2

    # Context order:
    #   5 pixels from row -2
    #   5 pixels from row -1
    #   2 pixels from the current row
    CONTEXT_OFFSETS = (
        (-2, -2),
        (-2, -1),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, -2),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (-1, 2),
        (0, -2),
        (0, -1),
    )

    '''
    @brief Extract the twelve causal spatial neighbors of each latent symbol.
    
    @param[in] latent Native-resolution latent grid with shape [B, 1, H, W].
    @return Context tensor with shape [B, 12, H, W].
    '''

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4:
            raise ValueError(
                f"Expected latent shape [B, 1, H, W], got {tuple(latent.shape)}."
            )

        if latent.shape[1] != 1:
            raise ValueError(
                f"Expected one latent channel, got {latent.shape[1]}."
            )

        _, _, height, width = latent.shape

        # Reconstruction from left to right, top to bottom
        # The problem is that pixels near the top and the left edges don't have
        # those neighbors, that's why we have to pad
        padded = F.pad(
            latent,
            (
                self.CONTEXT_RADIUS,
                self.CONTEXT_RADIUS,
                self.CONTEXT_RADIUS,
                0,
            ),
            mode="constant",
            value=0.0,
        )

        context_maps = []
        for row_offset, column_offset in self.CONTEXT_OFFSETS:
            row_start = self.CONTEXT_RADIUS + row_offset
            column_start = self.CONTEXT_RADIUS + column_offset

            context_maps.append(
                padded[
                    :,
                    :,
                    row_start: row_start + height,
                    column_start: column_start + width,
                ]
            )

        return torch.cat(context_maps, dim=1)