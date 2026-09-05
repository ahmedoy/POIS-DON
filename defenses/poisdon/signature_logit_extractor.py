"""
Code is adapted from:
https://github.com/vimal-isi-edu/trigs/blob/main/trigs/generate_model_signature.py

This software is Copyright © 2024 The University of Southern California.
All Rights Reserved.

Permission to use, copy, modify, and distribute this software and its
documentation for educational, research and non-profit purposes, without
fee, and without a written agreement is hereby granted, provided that the
above copyright notice, this paragraph and the following three paragraphs
appear in all copies.

Permission to make commercial use of this software may be obtained by contacting:
USC Stevens Center for Innovation
University of Southern California
1150 S. Olive Street, Suite 2300
Los Angeles, CA 90115, USA

This software program and documentation are copyrighted by The University of Southern California. The software program and documentation are supplied "as is", without any accompanying services from USC. USC does not warrant that the operation of the program will be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.

IN NO EVENT SHALL THE UNIVERSITY OF SOUTHERN CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF SOUTHERN CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. THE UNIVERSITY OF SOUTHERN CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN "AS IS" BASIS, AND THE UNIVERSITY OF SOUTHERN CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.

"""

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as ttf
from tqdm.auto import tqdm

from defenses.poisdon.optimization_configs import OptimizationConfig
from defenses.utils import defense_utils
from image_datasets.base_image_dataset_config import ImageDatasetConfigBase
from schema.defense_experiment import DefenseExperiment


def tv(img: torch.Tensor) -> torch.Tensor:
    # If passed a 3D tensor ([B, H, W]), convert to ([B, 1, H, W])
    if img.ndim == 3:
        img = img.unsqueeze(1)

    if img.ndim != 4:
        raise RuntimeError(
            f"Expected input `img` to be a 3D or 4D tensor, but got shape {list(img.shape)}"
        )

    diff1 = img[..., 1:, :] - img[..., :-1, :]
    diff2 = img[..., :, 1:] - img[..., :, :-1]

    # Sum across non-batch dimensions
    res1 = diff1.abs().sum(dim=list(range(1, img.ndim)))
    res2 = diff2.abs().sum(dim=list(range(1, img.ndim)))

    return res1 + res2


def images_to_tensors(
    imgs: np.ndarray,
    mean: tuple[float, float, float] | tuple[float],
    std: tuple[float, float, float] | tuple[float],
) -> torch.Tensor:
    ims_as_arrs = np.float32(imgs)

    # 1. Convert NHWC (B, H, W, C) -> NCHW (B, C, H, W)
    ims_as_arrs = ims_as_arrs.transpose(0, 3, 1, 2)  # type: ignore
    ims_as_arrs /= 255.0

    # 2. Reshape mean/std to (1, C, 1, 1) to match NCHW broadcasting
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std_arr = np.array(std, dtype=np.float32).reshape(1, -1, 1, 1)

    # 3. normalization
    ims_as_arrs = (ims_as_arrs - mean_arr) / std_arr

    return torch.from_numpy(ims_as_arrs)


def clamp_and_normalize_img_tensor(
    img_tensor: torch.Tensor,  # (B, C, H, W)
    mean: tuple[float, float, float] | tuple[float],
    std: tuple[float, float, float] | tuple[float],
):

    # 2. Reshape mean/std to (1, C, 1, 1) to match NCHW broadcasting
    mean_arr = torch.tensor(
        mean, dtype=img_tensor.dtype, device=img_tensor.device
    ).view(1, -1, 1, 1)

    std_arr = torch.tensor(std, dtype=img_tensor.dtype, device=img_tensor.device).view(
        1, -1, 1, 1
    )

    # 3. denormalization
    img_tensor = img_tensor * std_arr + mean_arr

    # 4. clamping
    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

    # 5. renormalize
    img_tensor = (img_tensor - mean_arr) / std_arr
    return img_tensor


def rfft2d_freqs(h: int, w: int) -> torch.Tensor:
    """Calculates 2D frequency magnitudes for 2D real FFT spectrum scaling."""
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.rfftfreq(w)[None, :]
    return torch.sqrt(fy**2 + fx**2)


def fourier_to_spatial(
    spectrum: torch.Tensor,
    height: int,
    width: int,
    decay_power: float = 1.0,
    sd: float = 0.01,
) -> torch.Tensor:
    """
    Transforms Fourier frequency spectrum parameters into a valid [0, 1] spatial image tensor.

    Args:
        spectrum: Tensor of shape (B, C, H, W//2 + 1, 2) containing real and imaginary components.
        height: Spatial height of destination image.
        width: Spatial width of destination image.
        decay_power: Power factor for low-pass frequency scaling (1.0 = standard 1/f).
        sd: Standard deviation for scaling the complex components.
    """
    # 1. Scale spectrum by frequency decay factor (low-pass filter)
    freqs = rfft2d_freqs(height, width).to(spectrum.device)
    scale = (
        1.0
        / torch.maximum(
            freqs, torch.tensor(1.0 / max(height, width), device=spectrum.device)
        )
        ** decay_power
    )
    scale = scale * (height * width) ** 0.5  # Preserve energy normalization

    # 2. Convert real/imaginary tensor components to complex tensor
    complex_spectrum = torch.complex(spectrum[..., 0], spectrum[..., 1]) * scale * sd

    # 3. Inverse 2D Real FFT to bring image into spatial domain
    spatial_raw = torch.fft.irfft2(complex_spectrum, s=(height, width))

    # 4. Map unbounded spatial values into strictly [0, 1] space via Sigmoid
    spatial_bounded = torch.sigmoid(spatial_raw)

    return spatial_bounded


def standardize_tensor(
    tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Standardizes a [0, 1] spatial image batch with dataset mean and std."""
    return (tensor - mean) / std


class SpectrumParameterization(torch.nn.Module):
    """Encapsulates learnable Fourier spectrum parameters."""

    def __init__(self, batch_size: int, channels: int, height: int, width: int):
        super().__init__()
        self.height = height
        self.width = width
        # Shape for rfft2 output: (B, C, H, W // 2 + 1, 2) for real and imaginary parts
        freq_w = width // 2 + 1

        # Initialize Fourier parameters with zero-mean unit variance noise
        init_spectrum = torch.randn(batch_size, channels, height, freq_w, 2)
        self.spectrum = torch.nn.Parameter(init_spectrum)

    def forward(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        # Generate [0, 1] spatial image
        img_01 = fourier_to_spatial(self.spectrum, self.height, self.width)
        # Normalize for model forward pass
        return standardize_tensor(img_01, mean, std)


class SignatureLogitsExtractor:
    def __init__(
        self,
        poisdon_config: DefenseExperiment,
        dataset_config: ImageDatasetConfigBase,
        optimization_config: OptimizationConfig,
    ):
        self.dataset_config = dataset_config
        self.poisdon_config = poisdon_config
        self.optimization_config = optimization_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.optimization_config.optimizer_type == "ADAM":
            self.optimizer_class = torch.optim.Adam
        elif self.optimization_config.optimizer_type == "SGD":
            self.optimizer_class = torch.optim.SGD

        poisdon_mode: Literal["min", "max", "both"] = (
            self.poisdon_config.defense_config["poisdon_mode"]
        )

        if poisdon_mode == "both":
            self.poisdon_modes = ("min", "max")
        else:
            self.poisdon_modes = (poisdon_mode,)

        self.use_standardization: bool = self.poisdon_config.defense_config[
            "poisdon_mode"
        ]

    def get_features(
        self,
        model_path_list: list[Path],
    ) -> dict[Literal["min", "max"], list[torch.Tensor]]:
        eps = 1e-7
        model_features: dict[Literal["min", "max"], list[torch.Tensor]] = {
            poisdon_mode: [] for poisdon_mode in self.poisdon_modes
        }
        rng = np.random.default_rng(seed=self.poisdon_config.seed)
        init_image = np.uint8(
            rng.uniform(
                0,
                255,
                (
                    self.dataset_config.img_width,
                    self.dataset_config.img_height,
                    self.dataset_config.num_channels_input,
                ),
            )
        )

        for poisdon_mode in self.poisdon_modes:
            if poisdon_mode == "min":
                is_ascent = True
            elif poisdon_mode == "max":
                is_ascent = False
            else:
                raise ValueError(
                    "Invalid value for poisdon mode. Please use either 'both', 'min', or 'max'"
                )

            for model_path in tqdm(
                model_path_list, desc="Extracting Signatures", unit="models"
            ):
                model, eval_transform, model_artifact_json_data = (
                    defense_utils.load_model(model_path)
                )

                if model is None:
                    raise ValueError(
                        f"Failed to read model checkpoint saved at: {model_path}"
                    )
                model = model.to(self.device)
                logits_tensor = self._get_single_model_logits_original_method(
                    model, eval_transform, init_image, ascent=is_ascent
                )

                if poisdon_mode == "min":
                    if self.use_standardization:
                        # Standardize each row before softmax.
                        logits_std = (
                            logits_tensor - logits_tensor.mean(dim=1, keepdim=True)
                        ) / (logits_tensor.std(dim=1, keepdim=True) + eps)
                        feature = torch.softmax(logits_std, dim=1).mean(dim=0)
                    else:
                        feature = torch.softmax(logits_tensor, dim=1).mean(dim=0)
                else:
                    # Extract the diagonal entries (convert to float32 for numerical stability)
                    raw_diag = torch.diagonal(logits_tensor, dim1=0, dim2=1).to(
                        torch.float32
                    )
                    if self.use_standardization:
                        feature = (raw_diag - raw_diag.mean()) / (raw_diag.std() + eps)
                    else:
                        feature = raw_diag

                model_features[poisdon_mode].append(feature)

                del model
                torch.cuda.empty_cache()

        return model_features

    # Poisdon implementation in the paper
    # TODO modify to use eval_transforms (including mean and std) specific to the model instead of self.optimization_config. Should be doable with defense_utils.load_model
    def _get_single_model_logits_original_method(
        self,
        model: torch.nn.Module,
        eval_transform,
        init_image: np.uint8,
        ascent: bool,
    ) -> torch.Tensor:
        model.eval()

        # CxC logit tensor used by poisdon
        num_classes = self.dataset_config.num_classes
        logits_tensor = torch.zeros((num_classes, num_classes), dtype=torch.float32)

        blur_ks = 2 * self.optimization_config.blur_half_kernel_width + 1

        target_classes = [c for c in range(num_classes)]

        created_images = np.tile(init_image, [len(target_classes), 1, 1, 1])
        processed_images = images_to_tensors(
            created_images,
            mean=self.optimization_config.mean,
            std=self.optimization_config.std,
        )

        processed_images = processed_images.to(self.device)
        processed_images.requires_grad = True
        targets = torch.as_tensor(target_classes, device=self.device)
        indices = torch.arange(end=len(target_classes), device=self.device)
        optimizer = self.optimizer_class(
            [processed_images],
            lr=self.optimization_config.lr,
            weight_decay=self.optimization_config.weight_decay,
        )
        min_losses = [float("inf")] * len(target_classes)
        best_images = processed_images.clone()

        for i in range(self.optimization_config.iterations):
            # Process image and return variable
            # implement gaussian blurring every blur_freq iteration to improve output
            if (
                self.optimization_config.blur_freq
                and i % self.optimization_config.blur_freq == 0
            ):
                processed_images = ttf.gaussian_blur(
                    processed_images.detach(),
                    [
                        blur_ks,
                        blur_ks,
                    ],
                    self.optimization_config.blur_sigma,  # type: ignore
                )
                processed_images.requires_grad = True
                optimizer = self.optimizer_class(
                    [processed_images],
                    lr=self.optimization_config.lr,
                    weight_decay=self.optimization_config.weight_decay,
                )
            # Forward
            output = model(processed_images)

            # Target specific class
            if self.optimization_config.loss_type == "ce":
                class_loss = F.cross_entropy(output, targets, reduction="none")
            elif self.optimization_config.loss_type == "logit":
                class_loss = -output[indices, targets]
            else:
                raise ValueError(
                    f"Unsupported loss type: {self.optimization_config.loss_type}"
                )
            if ascent:
                class_loss = -class_loss

            tv_loss = (
                (self.optimization_config.lambda_tv * tv(processed_images))
                if self.optimization_config.lambda_tv > 0.0
                else 0.0
            )

            total_loss = class_loss + tv_loss

            for li, loss in enumerate(total_loss):
                if loss.item() < min_losses[li]:
                    min_losses[li] = loss.item()
                    best_images[li] = processed_images[li].detach().clone()
                    logits_tensor[li] = output[li].clone().cpu().detach()

            optimizer.zero_grad()

            # Backward
            total_loss.sum().backward()

            if self.optimization_config.normalize_grad:
                for gi in range(len(target_classes)):
                    # shape: (C, H, W)
                    img_grad = processed_images.grad[gi]  # type: ignore

                    # keepdim=True produces shape (C, 1, 1)
                    norm = torch.norm(img_grad, dim=[1, 2], keepdim=True)

                    # Direct in-place division with no permute needed
                    processed_images.grad[gi] /= norm + 1e-8  # type: ignore

            if self.optimization_config.clipping_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    processed_images, self.optimization_config.clipping_val
                )

            # Update image
            optimizer.step()

            if (
                self.optimization_config.clamp_pixels_freq
                and i % self.optimization_config.clamp_pixels_freq == 0
            ):
                tmp_images = clamp_and_normalize_img_tensor(
                    processed_images.detach(),
                    mean=self.optimization_config.mean,
                    std=self.optimization_config.std,
                )
                processed_images = tmp_images.contiguous()
                processed_images.requires_grad = True
                optimizer = self.optimizer_class(
                    [processed_images],
                    lr=self.optimization_config.lr,
                    weight_decay=self.optimization_config.weight_decay,
                )

        return logits_tensor

    # TODO Review improved signature logit optimization method
    def _get_single_model_logits_fourier_sigmoid(
        self,
        model: torch.nn.Module,
        eval_transform,
        init_image: np.uint8,
        ascent: bool,
    ) -> torch.Tensor:
        model.eval()

        num_classes = self.dataset_config.num_classes
        logits_tensor = torch.zeros((num_classes, num_classes), dtype=torch.float32)

        target_classes = list(range(num_classes))
        batch_size = len(target_classes)

        # 1. Setup spatial dimensions and normalization tensors
        channels, height, width = (  # type: ignore
            init_image.shape[:3] if init_image.ndim == 4 else init_image.shape
        )

        mean_tensor = torch.tensor(
            self.optimization_config.mean, device=self.device
        ).view(1, -1, 1, 1)
        std_tensor = torch.tensor(
            self.optimization_config.std, device=self.device
        ).view(1, -1, 1, 1)

        # 2. Instantiate learnable Fourier spectrum parameterization
        spectrum_param = SpectrumParameterization(
            batch_size=batch_size, channels=channels, height=height, width=width
        ).to(self.device)

        # 3. Setup optimizer for Fourier parameters (Adam works best in frequency space)
        optimizer = torch.optim.Adam(
            spectrum_param.parameters(),
            lr=self.optimization_config.lr,
            weight_decay=self.optimization_config.weight_decay,
        )

        targets = torch.as_tensor(target_classes, device=self.device)
        indices = torch.arange(end=batch_size, device=self.device)

        min_losses = [float("inf")] * batch_size

        for i in range(self.optimization_config.iterations):
            # Generate spatial images from frequency spectrum parameters smoothly mapped via Sigmoid
            processed_images = spectrum_param(mean_tensor, std_tensor)

            # Forward pass
            output = model(processed_images)

            # Loss Calculation
            if self.optimization_config.loss_type == "ce":
                class_loss = F.cross_entropy(output, targets, reduction="none")
            elif self.optimization_config.loss_type == "logit":
                class_loss = -output[indices, targets]
            else:
                raise ValueError(
                    f"Unsupported loss type: {self.optimization_config.loss_type}"
                )

            if ascent:
                class_loss = -class_loss

            # In Fourier space, total_loss is simply class_loss (TV loss is no longer required)
            total_loss = class_loss

            # Track best results
            for li, loss_val in enumerate(total_loss):
                if loss_val.item() < min_losses[li]:
                    min_losses[li] = loss_val.item()
                    logits_tensor[li] = output[li].clone().cpu().detach()

            # Optimization Step
            optimizer.zero_grad()
            total_loss.sum().backward()

            if (
                self.optimization_config.normalize_grad
                and spectrum_param.spectrum.grad is not None
            ):
                # Gradient normalization directly in frequency space
                grad_norm = torch.norm(
                    spectrum_param.spectrum.grad, dim=[1, 2, 3, 4], keepdim=True
                )
                spectrum_param.spectrum.grad /= grad_norm + 1e-8

            if self.optimization_config.clipping_val > 0:
                torch.nn.utils.clip_grad_norm_(
                    spectrum_param.parameters(), self.optimization_config.clipping_val
                )

            optimizer.step()

        return logits_tensor
