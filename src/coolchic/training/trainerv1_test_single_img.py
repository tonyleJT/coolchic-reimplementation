import torch
from torch import nn


'''
@brief Overfit a COOL-CHIC model to one image.

@param[in,out] model Integrated COOL-CHIC model to optimize.
@param[in] target Target RGB image with shape [1, 3, H, W].
@param[in] lambda_rd Rate-distortion tradeoff parameter.
@param[in] steps Number of optimization steps.
@param[in] learning_rate Adam learning rate.
@param[in] log_every Number of steps between progress messages.
@return Final model output evaluated with rounded latent variables.
'''

def train_single_image(
    model: nn.Module,
    target: torch.Tensor,
    lambda_rd: float,
    steps: int,
    learning_rate: float,
    log_every: int = 100,
) -> dict[str, torch.Tensor]:
    if steps <= 0:
        raise ValueError("steps must be positive.")

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if lambda_rd < 0.0:
        raise ValueError("lambda_rd must be non-negative.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    model.train()

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)

        output = model(
            target=target,
            lambda_rd=lambda_rd,
        )

        output["loss"].backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"step={step:4d} | "
                f"loss={output['loss'].item():.6f} | "
                f"mse={output['mse'].item():.6f} | "
                f"psnr={output['psnr'].item():.3f} | "
                f"latent_bpp={output['latent_bpp'].item():.6f}"
            )

    # Final evaluation uses rounded latents instead of training noise.
    model.eval()

    with torch.no_grad():
        final_output = model(
            target=target,
            lambda_rd=lambda_rd,
        )

    print(
        "final-rounded | "
        f"loss={final_output['loss'].item():.6f} | "
        f"mse={final_output['mse'].item():.6f} | "
        f"psnr={final_output['psnr'].item():.3f} | "
        f"latent_bpp={final_output['latent_bpp'].item():.6f}"
    )

    return final_output