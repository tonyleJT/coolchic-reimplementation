from pathlib import Path
import torch
import numpy as np
from PIL import Image

'''
@brief Load an RGB image as a normalized Pytorch tensor.

@param path: Path to the image file.
@return: Image tensor with shape [1, 3, H, W], dtype float32, in [0, 1].
'''

def load_image(path: str | Path) -> torch.Tensor:
    # Load an image as a [1, 3, H, W] float32 RGB tensor in [0, 1].
    with Image.open(path) as image:
        image = image.convert('RGB')
        array = np.array(image, dtype=np.float32, copy=True)
    array /= 255.0

    tensor = torch.from_numpy (array)
    tensor = tensor.permute(2,0,1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
    # unsqueeze(0) adds a batch dimension at the front, resulting in shape [1, 3, H, W].

    return tensor.contiguous()


'''
@brief Save a reconstructed RGB tensor as a PNG image.
- detach() removes the tensor from PyTorch's computation graph (no need gradient
for this image anymore).

@param image Image tensor with shape [1, 3, H, W].
@param path Path where the PNG image will be saved.
@return None.
'''

def save_image(image: torch.Tensor, path: str | Path) -> None:
    # Save a [1, 3, H, W] RGB tensor as an 8-bit PNG.
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError(
            "Expected image shape [1, 3, H, W], "
            f"but received {tuple(image.shape)}."
        )

    image = image.detach().to(device='cpu', dtype=torch.float32)
    image = image.clamp(0.0, 1.0)  # Ensure values are in [0, 1]

    array = (
        image.squeeze(0)  # Remove batch dimension, shape [3, H, W]
        .permute(1, 2, 0)  # [3, H, W] -> [H, W, 3]
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .numpy()
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    Image.fromarray(array).save(output_path)