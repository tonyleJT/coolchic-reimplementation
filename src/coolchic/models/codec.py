import torch
import torch.nn.functional as F
from torch import nn

from coolchic.models.context import CausalContext
from coolchic.models.latent import HierarchicalLatents
from coolchic.models.probability import LaplaceProbabilityModel
from coolchic.models.synthesis import SynthesisMLP


class COOLCHICModel(nn.Module):
    # Integrated COOL-CHIC model for one image.

    '''
    # /**
    #  * @brief Create the integrated COOL-CHIC model.
    #  *
    #  * @param[in] height Target image height in pixels.
    #  * @param[in] width Target image width in pixels.
    #  * @return None.
    #  */
    '''

    def __init__(self, height: int, width: int) -> None:
        super().__init__()

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        self.height = height
        self.width = width

        self.latents = HierarchicalLatents(height, width)
        self.synthesis = SynthesisMLP()
        self.context = CausalContext()
        self.probability = LaplaceProbabilityModel()

    '''
    # /**
    #  * @brief Upsample seven quantized latent grids to image resolution.
    #  *
    #  * @param[in] quantized_latents Seven latent tensors at native resolutions.
    #  * @return Dense latent representation with shape [1, 7, H, W].
    #  */
    '''

    def _upsample_latents(
        self,
        quantized_latents: list[torch.Tensor],
    ) -> torch.Tensor:
        upsampled_latents = []

        for latent in quantized_latents:
            upsampled = F.interpolate(
                latent,
                size=(self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            upsampled_latents.append(upsampled)

        return torch.cat(upsampled_latents, dim=1)

    '''
    # /**
    #  * @brief Estimate the total entropy-model rate of all latent levels.
    #  *
    #  * @param[in] quantized_latents Seven latent tensors at native resolutions.
    #  * @return Estimated total latent rate in bits.
    #  */
    '''

    def _estimate_latent_bits(
        self,
        quantized_latents: list[torch.Tensor],
    ) -> torch.Tensor:
        latent_bits = quantized_latents[0].new_zeros(())

        for latent in quantized_latents:
            context = self.context(latent)

            level_bits, _ = self.probability.estimate_rate(
                latent,
                context,
                self.height,
                self.width,
            )

            latent_bits = latent_bits + level_bits

        return latent_bits

    '''
    # /**
    #  * @brief Run the integrated COOL-CHIC rate-distortion forward pass.
    #  *
    #  * @param[in] target Target RGB image with shape [1, 3, H, W].
    #  * @param[in] lambda_rd Rate-distortion tradeoff parameter.
    #  * @return Dictionary containing reconstruction, MSE, PSNR,
    #  *         estimated latent bits, estimated latent bpp, and loss.
    #  */
    '''

    def forward(
        self,
        target: torch.Tensor,
        lambda_rd: float,
    ) -> dict[str, torch.Tensor]:
        expected_shape = (1, 3, self.height, self.width)

        if tuple(target.shape) != expected_shape:
            raise ValueError(
                f"target must have shape {expected_shape}, "
                f"got {tuple(target.shape)}"
            )

        if lambda_rd < 0.0:
            raise ValueError("lambda_rd must be non-negative")

        # Quantize exactly once so reconstruction and rate use
        # the same noisy/rounded latent realization.
        quantized_latents = self.latents.quantize()

        dense_latents = self._upsample_latents(quantized_latents)
        reconstruction = self.synthesis(dense_latents)

        # Training keeps the synthesis output unclamped so gradients
        # are not killed outside [0, 1]. Evaluation uses valid RGB.
        if not self.training:
            reconstruction = reconstruction.clamp(0.0, 1.0)

        mse = F.mse_loss(reconstruction, target)

        psnr = 10.0 * torch.log10(
            1.0 / mse.clamp_min(1e-12)
        )

        latent_bits = self._estimate_latent_bits(quantized_latents)
        latent_bpp = latent_bits / (self.height * self.width)

        loss = mse + lambda_rd * latent_bpp

        return {
            "reconstruction": reconstruction,
            "mse": mse,
            "psnr": psnr,
            "latent_bits": latent_bits,
            "latent_bpp": latent_bpp,
            "loss": loss,
        }