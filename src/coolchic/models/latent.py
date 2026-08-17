import torch
import torch.nn.functional as F
from torch import nn


class HierarchicalLatents(nn.Module):
    # Seven-level hierarchical latent representation used by COOL-CHIC.

    NUM_LEVELS = 7  # Number of hierarchical latent levels

    '''
    @brief Create 7 trainable hierarchical laten grids.
    
    @param[in] height Target image height in pixels.
    @param[in] width Target image width in pixels.
    @return None.
    '''

    def __init__(self, height: int, width: int) -> None:
        super().__init__()

        if height < 0 or width < 0:
            raise ValueError("Height and width must be non-negative integers.")

        self.height = height
        self.width = width

        latent_grids = []
        for level in range(self.NUM_LEVELS):
            scale = 2**level

            # Ceiling division keeps non-divisible dimensions safe.
            latent_height = max(1, (height + scale -1) // scale)  # using max to avoid zero dimensions
            latent_width = max(1, (width + scale -1) // scale)

            # .zeros() initializes the parameter with zeros.
            # wrap with nn.Parameter is used to create a trainable parameter in the model
            # (allow the optimizer).
            latent = nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    latent_height,
                    latent_width,
                    dtype=torch.float32,
                )
            )
            latent_grids.append(latent)
        # nn.ParameterList is used to store a list of parameters in a module.
        # It allows the model to recognize and optimize all the parameters in the list during training.
        self.latent_grids = nn.ParameterList(latent_grids)


    '''
    @brief Upsample all latent grids and concatenate them channel-wise.
    
    @return Dense latent representation with shape [1, 7, H, W].
    '''

    def forward(self) -> torch.Tensor:
        upsampled_latents = []

        for latent in self.latent_grids:
            # F.interpolate is used to upsample the latent grid to the target image size.
            # mode='bilinear' specifies bilinear interpolation, which is suitable for continuous data.
            # align_corners=False ensures that the corner pixels are not aligned, which can help avoid artifacts.
            upsampled = F.interpolate(
                input=latent,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
            )
            upsampled_latents.append(upsampled)

        return torch.cat(upsampled_latents, dim=1)  # Concatenate along the channel dimension

    '''
    @brief Quantize all hierarchical latent grids according to the current model mode.
    - During training, uniform noise in [-0.5, 0.5] is added to each latent grid.
    - During evaluation, each latent value is rounded to the nearest integer.
    - round() during training is not good for optimizers, gradient descent ...
    
    @return List containing the seven quantized latent grids.
    '''

    def quantize(self) -> list[torch.Tensor]:
        quantized_latents = []

        for latent in self.latent_grids:
            if self.training:
                # Uniform noise approximates quantization while keeping the operation differentiable.
                # empty_like(): creates a tensor with the same shape, dtype,
                # and device as latent, but without initializing its values meaningfully.
                # Example: latent.shape = [1, 1, 128, 128] same noise.shape = [1, 1, 128, 128]
                noise = torch.empty_like(latent).uniform_(-0.5, 0.5)
                quantized = latent + noise

            else:
                # Evaluation uses actual integer-valued latent symbols.
                quantized = torch.round(latent)

            quantized_latents.append(quantized)

        return quantized_latents

    '''
    @brief Quantize, upsample, and concatenate all hierarchical latent grids.

    @return Dense quantized latent representation with shape [1, 7, H, W].
    '''

    def quantized_forward(self) -> torch.Tensor:
        quantized_latents = self.quantize()
        upsampled_latents = []

        for latent in quantized_latents:
            upsampled = F.interpolate(
                input=latent,
                size=(self.height, self.width),
                mode='bicubic',
                align_corners=False,
            )
            upsampled_latents.append(upsampled)

        return torch.cat(upsampled_latents, dim=1)