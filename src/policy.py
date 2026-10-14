"""DINOv3 attention modules for seven-dimensional continuous robot actions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import torch
from torch import Tensor, nn
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DINOv3VisionEncoder(nn.Module):
    """Encodes a BCHW image batch with DINOv3 patch-token self-attention.

    The DINOv3 backbone is frozen by default. Set ``freeze_backbone=False`` to
    fine-tune it jointly with the attention layers.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        num_attention_heads: int = 6,
        freeze_backbone: bool = True,
        backbone: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if backbone is not None:
            self.backbone = backbone
        elif checkpoint_path is not None:
            self.backbone = self._load_local_backbone(model_name, checkpoint_path)
        else:
            self.backbone = torch.hub.load("facebookresearch/dinov3", model_name)
        self.embedding_dim = self._embedding_dim()
        if self.embedding_dim % num_attention_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_attention_heads; "
                f"got {self.embedding_dim} and {num_attention_heads}."
            )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=num_attention_heads,
            batch_first=True,
        )
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

        if freeze_backbone:
            self.backbone.requires_grad_(False)

    @staticmethod
    def _load_local_backbone(model_name: str, checkpoint_path: str) -> nn.Module:
        """Load a local DINOv3 checkpoint without requesting remote weights."""
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"DINOv3 checkpoint not found: {checkpoint}. Set DINO_CHECKPOINT "
                "to a local pretrained DINOv3 checkpoint."
            )

        backbone = torch.hub.load(
            "facebookresearch/dinov3", model_name, pretrained=False
        )
        state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Expected a state dictionary in DINOv3 checkpoint {checkpoint}."
            )
        incompatible = backbone.load_state_dict(state_dict, strict=False)
        unexpected_core_keys = [
            key
            for key in incompatible.unexpected_keys
            if not key.startswith("projectors.")
        ]
        if incompatible.missing_keys or unexpected_core_keys:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is incompatible with {model_name}: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={unexpected_core_keys}."
            )
        return backbone

    def _embedding_dim(self) -> int:
        for attribute in ("embed_dim", "num_features"):
            value = getattr(self.backbone, attribute, None)
            if isinstance(value, int):
                return value
        raise AttributeError(
            "Could not determine the DINOv3 embedding dimension. The backbone "
            "must expose an integer 'embed_dim' or 'num_features' attribute."
        )

    def forward(self, images: Tensor) -> Tensor:
        """Return one attended embedding per normalized BCHW image."""
        features = self.backbone_features(images)
        patch_tokens = self._patch_tokens(features)
        attended_tokens, _ = self.attention(
            patch_tokens, patch_tokens, patch_tokens, need_weights=False
        )
        return attended_tokens.mean(dim=1)

    def backbone_features(self, images: Tensor) -> Any:
        """Return normalized DINOv3 backbone features for uint8 BCHW images."""
        return self.backbone.forward_features(
            (images.float() / 255.0 - self.image_mean) / self.image_std
        )

    def global_embedding(self, images: Tensor) -> Tensor:
        """Return the pretrained DINOv3 global embedding without policy attention."""
        features = self.backbone_features(images)
        if isinstance(features, dict):
            class_tokens = features.get("x_norm_clstoken")
            if isinstance(class_tokens, Tensor) and class_tokens.ndim == 2:
                return class_tokens
        return self._patch_tokens(features).mean(dim=1)

    @staticmethod
    def _patch_tokens(features: Any) -> Tensor:
        if isinstance(features, Tensor):
            if features.ndim != 3:
                raise ValueError(
                    "DINOv3 tensor features must have shape (batch, tokens, embedding)."
                )
            return features
        if isinstance(features, dict):
            tokens = features.get("x_norm_patchtokens")
            if isinstance(tokens, Tensor) and tokens.ndim == 3:
                return tokens
        raise TypeError(
            "DINOv3 forward_features must return patch tokens as a tensor or "
            "in the 'x_norm_patchtokens' dictionary entry."
        )


class DINOv3VisuomotorExtractor(BaseFeaturesExtractor):
    """Encode LIBERO's camera views, proprioception, and elapsed step count."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        model_name: str = "dinov3_vits16",
        num_attention_heads: int = 6,
        freeze_backbone: bool = True,
        backbone: Optional[nn.Module] = None,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        for key in ("agentview", "wrist", "proprio", "step"):
            if key not in observation_space.spaces:
                raise KeyError(f"Expected '{key}' in the observation space.")
        proprio_shape = observation_space["proprio"].shape
        if proprio_shape is None or len(proprio_shape) != 1:
            raise ValueError("'proprio' must be a one-dimensional observation.")
        step_shape = observation_space["step"].shape
        if step_shape != (1,):
            raise ValueError("'step' must be a one-dimensional scalar observation.")

        vision_encoder = DINOv3VisionEncoder(
            model_name=model_name,
            num_attention_heads=num_attention_heads,
            freeze_backbone=freeze_backbone,
            backbone=backbone,
            checkpoint_path=checkpoint_path,
        )
        features_dim = 2 * vision_encoder.embedding_dim + proprio_shape[0] + 1
        super().__init__(observation_space, features_dim=features_dim)
        self.vision_encoder = vision_encoder

    def forward(self, observations: dict[str, Tensor]) -> Tensor:
        agentview, wrist = observations["agentview"], observations["wrist"]
        if agentview.shape != wrist.shape:
            raise ValueError(
                "agentview and wrist observations must have identical shapes; "
                f"got {tuple(agentview.shape)} and {tuple(wrist.shape)}."
            )
        image_embeddings = self.vision_encoder(torch.cat((agentview, wrist), dim=0))
        agentview_embedding, wrist_embedding = image_embeddings.chunk(2, dim=0)
        return torch.cat(
            (
                agentview_embedding,
                wrist_embedding,
                observations["proprio"].float(),
                observations["step"].float(),
            ),
            dim=1,
        )


class DINOv3VisuomotorPolicy(MultiInputActorCriticPolicy):
    """PPO actor-critic policy using DINOv3 visual features.

    Stable-Baselines3 supplies the Gaussian 7-D action head and value head;
    this policy supplies their shared DINOv3 attention representation.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("features_extractor_class", DINOv3VisuomotorExtractor)
        kwargs.setdefault("net_arch", {"pi": [256], "vf": [256]})
        kwargs.setdefault("normalize_images", False)
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule: Any) -> None:
        # SB3's orthogonal init rewrites every Linear/Conv2d in the extractor, including the pretrained backbone.
        backbone = self.features_extractor.vision_encoder.backbone
        pretrained = {
            name: value.clone() for name, value in backbone.state_dict().items()
        }
        super()._build(lr_schedule)
        backbone.load_state_dict(pretrained)
