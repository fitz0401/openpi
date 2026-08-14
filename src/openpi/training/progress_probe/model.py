"""Vision-language Progress Regression Lite v3 with a frozen CLIP backbone."""

from __future__ import annotations

import pathlib
from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as functional
from transformers import AutoTokenizer
from transformers import CLIPModel


class ClipBackbone(Protocol):
    output_dim: int

    def encode_image(self, images: torch.Tensor) -> torch.Tensor: ...

    def encode_text(self, instructions: list[str], device: torch.device) -> torch.Tensor: ...


class FrozenClipVitB32(nn.Module):
    """Reloadable frozen CLIP; its parameters are absent from probe checkpoints."""

    output_dim = 512

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        self.model_name = model_name
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.clip.requires_grad_(requires_grad=False)
        self.clip.eval()
        self.register_buffer("image_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None])

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002 - matches torch.nn.Module.train
        super().train(mode)
        self.clip.eval()
        return self

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        pixels = functional.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
        pixels = (pixels - self.image_mean) / self.image_std
        return self.clip.get_image_features(pixel_values=pixels)

    @torch.no_grad()
    def encode_text(self, instructions: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(instructions, padding=True, truncation=True, return_tensors="pt").to(device)
        return self.clip.get_text_features(**tokens)


class ProgressRegressionVisionLanguageLite(nn.Module):
    """Shared scalar progress regressor with no proprioception branch."""

    def __init__(self, clip: ClipBackbone) -> None:
        super().__init__()
        self.clip = clip
        self.image_projection = nn.Linear(clip.output_dim, 128)
        self.text_projection = nn.Linear(clip.output_dim, 128)
        self.progress_head = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, images: torch.Tensor, instructions: list[str]) -> torch.Tensor:
        image = self.image_projection(self.clip.encode_image(images))
        text = self.text_projection(self.clip.encode_text(instructions, images.device))
        fused = torch.cat((image, text, image * text), dim=-1)
        return torch.sigmoid(self.progress_head(fused).squeeze(-1))

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value for name, value in self.state_dict().items() if not name.startswith("clip.")}

    def load_trainable_checkpoint(self, path: str | pathlib.Path, *, map_location: str | torch.device = "cpu") -> dict:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        incompatible = self.load_state_dict(payload["trainable_state_dict"], strict=False)
        unexpected = [name for name in incompatible.unexpected_keys if not name.startswith("clip.")]
        missing = [name for name in incompatible.missing_keys if not name.startswith("clip.")]
        if unexpected or missing:
            raise RuntimeError(f"Invalid probe checkpoint: missing={missing}, unexpected={unexpected}")
        return payload

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_vision_language_regression_model(model_name: str) -> ProgressRegressionVisionLanguageLite:
    return ProgressRegressionVisionLanguageLite(FrozenClipVitB32(model_name))
