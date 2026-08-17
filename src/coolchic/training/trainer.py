import torch
from torch import nn

from coolchic.models.probability import LaplaceProbabilityModel
from coolchic.models.synthesis import SynthesisMLP


MLP_QUANTIZATION_STEPS = (1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5)
MIN_MLP_SCALE = 1.0e-6
MIN_MLP_PROBABILITY = 1.0e-9


'''
@brief Optimize one COOL-CHIC model directly on one image.

@param[in] model Integrated COOL-CHIC model.
@param[in] image Target image tensor with shape [1, 3, H, W].
@param[in] steps Number of optimization steps.
@param[in] learning_rate Adam learning rate.
@param[in] lambda_rd Rate-distortion weight.
@param[in] log_every Number of steps between progress messages.
@return Final evaluation output using rounded latents.
'''
def train_single_image(
    model: nn.Module,
    image: torch.Tensor,
    steps: int,
    learning_rate: float,
    lambda_rd: float,
    log_every: int = 100,
) -> dict:
    if steps <= 0:
        raise ValueError("steps must be positive.")

    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    if lambda_rd < 0.0:
        raise ValueError("lambda_rd must be non-negative.")

    if log_every <= 0:
        raise ValueError("log_every must be positive.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for step in range(1, steps + 1):
        model.train()

        optimizer.zero_grad(set_to_none=True)

        output = model(
            image,
            lambda_rd=lambda_rd,
        )

        output["loss"].backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(
                f"step={step:05d} | "
                f"loss={output['loss'].item():.6f} | "
                f"mse={output['mse'].item():.6f} | "
                f"psnr={output['psnr'].item():.3f} | "
                f"latent_bpp={output['latent_bpp'].item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        final_output = model(
            image,
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


'''
@brief Find the synthesis and probability MLPs inside the integrated model.

@param[in] model Integrated COOL-CHIC model.
@return Tuple containing the synthesis MLP and probability MLP.
'''
def _find_mlp_modules(
    model: nn.Module,
) -> tuple[SynthesisMLP, LaplaceProbabilityModel]:
    synthesis_model = None
    probability_model = None

    for module in model.modules():
        if synthesis_model is None and isinstance(module, SynthesisMLP):
            synthesis_model = module

        if probability_model is None and isinstance(
            module,
            LaplaceProbabilityModel,
        ):
            probability_model = module

    if synthesis_model is None:
        raise RuntimeError(
            "SynthesisMLP was not found inside the COOL-CHIC model."
        )

    if probability_model is None:
        raise RuntimeError(
            "LaplaceProbabilityModel was not found inside the COOL-CHIC model."
        )

    return synthesis_model, probability_model


'''
@brief Copy all trainable parameters from one module.

@param[in] module PyTorch module whose parameters are copied.
@return Detached parameter tensors in module parameter order.
'''
def _copy_parameters(
    module: nn.Module,
) -> list[torch.Tensor]:
    return [
        parameter.detach().clone()
        for parameter in module.parameters()
    ]


'''
@brief Restore parameters previously copied from a module.

@param[in,out] module PyTorch module whose parameters are restored.
@param[in] saved_parameters Saved parameter tensors in module parameter order.
@return None.
'''
def _restore_parameters(
    module: nn.Module,
    saved_parameters: list[torch.Tensor],
) -> None:
    with torch.no_grad():
        for parameter, saved_parameter in zip(
            module.parameters(),
            saved_parameters,
        ):
            parameter.copy_(saved_parameter)


'''
@brief Quantize one module in place with a uniform scalar quantizer.

@param[in,out] module PyTorch module to quantize.
@param[in] quantization_step Scalar quantization step.
@return None.
'''
def _quantize_module(
    module: nn.Module,
    quantization_step: float,
) -> None:
    if quantization_step <= 0.0:
        raise ValueError(
            "quantization_step must be positive."
        )

    with torch.no_grad():
        for parameter in module.parameters():
            quantized_parameter = (
                torch.round(
                    parameter / quantization_step
                )
                * quantization_step
            )

            parameter.copy_(quantized_parameter)


'''
@brief Estimate entropy-model bits for one quantized MLP.

Each quantized parameter is represented by the integer symbol
q = round(parameter / delta). The symbols are modeled using a
zero-mean Laplace distribution whose scale is their population
standard deviation. Probability mass is integrated over q +/- 0.5.

@param[in] module Quantized PyTorch module.
@param[in] quantization_step Quantization step used for the module.
@return Estimated number of bits for all module parameters.
'''
def _estimate_module_bits(
    module: nn.Module,
    quantization_step: float,
) -> torch.Tensor:
    symbols = torch.cat(
        [
            torch.round(
                parameter.detach().reshape(-1)
                / quantization_step
            )
            for parameter in module.parameters()
        ]
    )

    scale = symbols.std(
        unbiased=False,
    ).clamp_min(
        MIN_MLP_SCALE,
    )

    distribution = torch.distributions.Laplace(
        loc=torch.zeros(
            (),
            device=symbols.device,
            dtype=symbols.dtype,
        ),
        scale=scale,
    )

    probability = (
        distribution.cdf(
            symbols + 0.5
        )
        - distribution.cdf(
            symbols - 0.5
        )
    ).clamp_min(
        MIN_MLP_PROBABILITY
    )

    return (
        -torch.log2(probability)
    ).sum()


'''
@brief Quantize both COOL-CHIC MLPs and select their quantization steps.

The synthesis step is selected first because synthesis parameters
affect distortion but not latent probability. The probability-model
step is selected second because it affects latent rate but not
reconstruction. The final objective is:

MSE + lambda * estimated_total_bpp

@param[in,out] model Trained integrated COOL-CHIC model.
@param[in] image Target image tensor with shape [1, 3, H, W].
@param[in] lambda_rd Rate-distortion weight.
@param[in] quantization_steps Candidate scalar quantization steps.
@return Final evaluation output using rounded latents and quantized MLPs.
'''
def quantize_mlp_parameters(
    model: nn.Module,
    image: torch.Tensor,
    lambda_rd: float,
    quantization_steps: tuple[float, ...] = MLP_QUANTIZATION_STEPS,
) -> dict:
    if lambda_rd < 0.0:
        raise ValueError(
            "lambda_rd must be non-negative."
        )

    if not quantization_steps:
        raise ValueError(
            "quantization_steps must not be empty."
        )

    synthesis_model, probability_model = _find_mlp_modules(
        model
    )

    synthesis_full_precision = _copy_parameters(
        synthesis_model
    )

    probability_full_precision = _copy_parameters(
        probability_model
    )

    num_pixels = (
        image.shape[-2]
        * image.shape[-1]
    )

    model.eval()

    # ---------------------------------------------------------
    # Search synthesis MLP quantization step: delta_theta.
    # ---------------------------------------------------------

    best_theta_step = None
    best_theta_cost = float("inf")

    for theta_step in quantization_steps:
        _restore_parameters(
            synthesis_model,
            synthesis_full_precision,
        )

        _restore_parameters(
            probability_model,
            probability_full_precision,
        )

        _quantize_module(
            synthesis_model,
            theta_step,
        )

        theta_bits = _estimate_module_bits(
            synthesis_model,
            theta_step,
        )

        with torch.no_grad():
            output = model(
                image,
                lambda_rd=lambda_rd,
            )

        # The probability-model contribution is constant
        # during the theta search, so it does not affect
        # which theta step minimizes the objective.
        theta_search_bpp = (
            output["latent_bits"]
            + theta_bits
        ) / num_pixels

        theta_cost = (
            output["mse"]
            + lambda_rd * theta_search_bpp
        ).item()

        print(
            "theta-search | "
            f"step={theta_step:.0e} | "
            f"mse={output['mse'].item():.6f} | "
            f"theta_bits={theta_bits.item():.3f} | "
            f"cost={theta_cost:.6f}"
        )

        if theta_cost < best_theta_cost:
            best_theta_cost = theta_cost
            best_theta_step = theta_step

    # Keep the best synthesis quantization permanently.
    _restore_parameters(
        synthesis_model,
        synthesis_full_precision,
    )

    _restore_parameters(
        probability_model,
        probability_full_precision,
    )

    _quantize_module(
        synthesis_model,
        best_theta_step,
    )

    theta_bits = _estimate_module_bits(
        synthesis_model,
        best_theta_step,
    )

    # ---------------------------------------------------------
    # Search probability MLP quantization step: delta_psi.
    # ---------------------------------------------------------

    best_psi_step = None
    best_psi_cost = float("inf")

    for psi_step in quantization_steps:
        _restore_parameters(
            probability_model,
            probability_full_precision,
        )

        _quantize_module(
            probability_model,
            psi_step,
        )

        psi_bits = _estimate_module_bits(
            probability_model,
            psi_step,
        )

        with torch.no_grad():
            output = model(
                image,
                lambda_rd=lambda_rd,
            )

        total_estimated_bits = (
            output["latent_bits"]
            + theta_bits
            + psi_bits
        )

        estimated_bpp = (
            total_estimated_bits
            / num_pixels
        )

        psi_cost = (
            output["mse"]
            + lambda_rd * estimated_bpp
        ).item()

        print(
            "psi-search   | "
            f"step={psi_step:.0e} | "
            f"latent_bits={output['latent_bits'].item():.3f} | "
            f"psi_bits={psi_bits.item():.3f} | "
            f"cost={psi_cost:.6f}"
        )

        if psi_cost < best_psi_cost:
            best_psi_cost = psi_cost
            best_psi_step = psi_step

    # Keep the best probability-model quantization.
    _restore_parameters(
        probability_model,
        probability_full_precision,
    )

    _quantize_module(
        probability_model,
        best_psi_step,
    )

    # ---------------------------------------------------------
    # Final rounded-latent + quantized-MLP evaluation.
    # ---------------------------------------------------------

    theta_bits = _estimate_module_bits(
        synthesis_model,
        best_theta_step,
    )

    psi_bits = _estimate_module_bits(
        probability_model,
        best_psi_step,
    )

    with torch.no_grad():
        final_output = model(
            image,
            lambda_rd=lambda_rd,
        )

    final_output = dict(
        final_output
    )

    mlp_bits = (
        theta_bits
        + psi_bits
    )

    total_estimated_bits = (
        final_output["latent_bits"]
        + mlp_bits
    )

    estimated_bpp = (
        total_estimated_bits
        / num_pixels
    )

    final_loss = (
        final_output["mse"]
        + lambda_rd * estimated_bpp
    )

    final_output.update(
        {
            "loss": final_loss,
            "theta_bits": theta_bits,
            "psi_bits": psi_bits,
            "mlp_bits": mlp_bits,
            "total_estimated_bits": total_estimated_bits,
            "estimated_bpp": estimated_bpp,
            "theta_quantization_step": best_theta_step,
            "psi_quantization_step": best_psi_step,
        }
    )

    mlp_bpp = (
        mlp_bits
        / num_pixels
    )

    print(
        "final-quantized | "
        f"theta_step={best_theta_step:.0e} | "
        f"psi_step={best_psi_step:.0e} | "
        f"mse={final_output['mse'].item():.6f} | "
        f"psnr={final_output['psnr'].item():.3f} | "
        f"latent_bpp={final_output['latent_bpp'].item():.6f} | "
        f"mlp_bpp={mlp_bpp.item():.6f} | "
        f"estimated_bpp={estimated_bpp.item():.6f}"
    )

    return final_output