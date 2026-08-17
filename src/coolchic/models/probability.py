import torch
import torch.nn.functional as F
from torch import nn


class LaplaceProbabilityModel(nn.Module):
    # Auto-regressive Laplace probability model used to estimate latent rate.

    CONTEXT_SIZE = 12
    HIDDEN_SIZE = 12
    MIN_SCALE = 1e-6
    MIN_PROBABILITY = 1e-9

    '''
    @brief Create the COOL-CHIC probability MLP.

    The network maps 12 causal context values to the mean and scale
    parameters of one Laplace distribution.

    Architecture:
        12 -> 12 -> 12 -> 2

    The two outputs represent:
        mu
        raw_sigma
    '''
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(self.CONTEXT_SIZE, self.HIDDEN_SIZE)
        self.fc2 = nn.Linear(self.HIDDEN_SIZE, self.HIDDEN_SIZE)
        self.fc3 = nn.Linear(self.HIDDEN_SIZE, 2)

    '''
    @brief Predict Laplace parameters from causal contexts.

    @param[in] context Causal context tensor with shape [B, 12, H, W].
    @return mu Laplace mean tensor with shape [B, 1, H, W].
    @return sigma Positive Laplace scale tensor with shape [B, 1, H, W].
    '''

    def forward(self, context):
        if context.ndim != 4:
            raise ValueError(
                f"Expected context shape [B, 12, H, W], got {tuple(context.shape)}."
            )

        if context.shape[1] != self.CONTEXT_SIZE:
            raise ValueError(
                f"Expected {self.CONTEXT_SIZE} context channels, "
                f"got {context.shape[1]}."
            )

        # Move the 12 context values to the last dimension so nn.Linear
        # independently processes every spatial position.
        features = context.permute(0, 2, 3, 1)

        features = F.relu(self.fc1(features))
        features = F.relu(self.fc2(features))

        parameters = self.fc3(features)

        mu = parameters[..., 0].unsqueeze(1)

        raw_sigma = parameters[..., 1].unsqueeze(1)
        sigma = F.softplus(raw_sigma) + self.MIN_SCALE

        return mu, sigma

    '''
    @brief Evaluate the cumulative distribution function of a Laplace distribution.

    @param[in] value Values at which the CDF is evaluated.
    @param[in] mu Laplace mean.
    @param[in] sigma Positive Laplace scale.
    @return Laplace CDF evaluated at value.
    '''

    @staticmethod
    def _laplace_cdf(value, mu, sigma):
        normalized = (value - mu) / sigma

        # Clamp exponential arguments to non-positive values.
        # This avoids exponential overflow for extreme latent values.
        left = 0.5 * torch.exp(
            torch.clamp(normalized, max=0.0)
        )

        right = 1.0 - 0.5 * torch.exp(
            torch.clamp(-normalized, max=0.0)
        )

        return torch.where(normalized < 0.0, left, right)

    '''
    @brief Calculate discretized Laplace probability mass for latent symbols.

    Each symbol represents a quantization bin from y - 0.5 to y + 0.5.

    @param[in] latent Latent symbols with shape [B, 1, H, W].
    @param[in] mu Laplace mean with shape [B, 1, H, W].
    @param[in] sigma Laplace scale with shape [B, 1, H, W].
    @return Probability mass for every latent symbol.
    '''

    def probability_mass(self, latent, mu, sigma):
        if latent.shape != mu.shape or latent.shape != sigma.shape:
            raise ValueError(
                "latent, mu, and sigma must have identical shapes."
            )

        upper_cdf = self._laplace_cdf(
            latent + 0.5,
            mu,
            sigma,
        )

        lower_cdf = self._laplace_cdf(
            latent - 0.5,
            mu,
            sigma,
        )

        probability = upper_cdf - lower_cdf

        # Prevent log2(0) and protect against tiny floating-point errors.
        probability = probability.clamp(
            min=self.MIN_PROBABILITY,
            max=1.0,
        )

        return probability

    '''
    @brief Estimate latent rate in bits and bits per image pixel.

    @param[in] latent Latent symbols with shape [B, 1, H_latent, W_latent].
    @param[in] context Causal contexts with shape [B, 12, H_latent, W_latent].
    @param[in] image_height Original image height.
    @param[in] image_width Original image width.
    @return latent_bits Estimated number of latent bits.
    @return latent_bpp Estimated latent bits per original image pixel.
    '''

    def estimate_rate(
        self,
        latent,
        context,
        image_height,
        image_width,
    ):
        if image_height <= 0 or image_width <= 0:
            raise ValueError("Image height and width must be positive.")

        mu, sigma = self(context)

        probability = self.probability_mass(
            latent,
            mu,
            sigma,
        )

        bits_per_symbol = -torch.log2(probability)

        latent_bits = bits_per_symbol.sum()

        num_image_pixels = image_height * image_width
        latent_bpp = latent_bits / num_image_pixels

        return latent_bits, latent_bpp