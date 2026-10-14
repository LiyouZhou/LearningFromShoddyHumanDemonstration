"""Timestamp-aligned DINOv3 cosine-similarity reward for vectorized environments."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from policy import DINOv3VisionEncoder


class DINOv3SimilarityReward(VecEnvWrapper):
    """Reward time-aligned DINOv3 similarity with an additional success bonus."""

    def __init__(
        self,
        venv: VecEnv,
        human_embeddings: np.ndarray,
        control_timestep: float,
        similarity_weight: float = 1.0,
        success_weight: float = 100.0,
        gripper_weight: float = 0.5,
        z_weight: float = 2.0,
    ) -> None:
        if control_timestep <= 0:
            raise ValueError(
                f"control_timestep must be positive, got {control_timestep}."
            )
        super().__init__(venv)
        self.similarity_weight = similarity_weight
        self.success_weight = success_weight
        self.gripper_weight = gripper_weight
        self.z_weight = z_weight
        self.control_timestep = control_timestep
        self.timestamps = np.asarray(human_embeddings["timestamp"], dtype=np.float64)
        self.human_embeddings = np.asarray(
            human_embeddings["embedding"],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        self._elapsed_seconds = np.zeros(self.num_envs, dtype=np.float64)
        self._encoder: DINOv3VisionEncoder | None = None
        self._normalized_human_embeddings: torch.Tensor | None = None

        self._gripper_supervision = np.asanyarray(
            human_embeddings["gripper_state"],
            dtype=np.float32,
            copy=True,
            order="C",
        )

        self._z_supervision = np.asanyarray(
            human_embeddings["object_scale"],
            dtype=np.float32,
            copy=True,
            order="C",
        )
        self.z_min = 0.92
        self.z_max = 1.2

    def set_encoder(self, encoder: DINOv3VisionEncoder) -> None:
        """Share PPO's DINOv3 backbone instead of allocating another model."""
        self._encoder = encoder
        device = next(encoder.parameters()).device
        human_embeddings = torch.as_tensor(self.human_embeddings, device=device)
        self._normalized_human_embeddings = F.normalize(human_embeddings, dim=1)

    def reset(self) -> dict[str, np.ndarray]:
        self._elapsed_seconds.fill(0.0)
        return self.venv.reset()

    def step_wait(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        observations, _, dones, infos = self.venv.step_wait()
        self._elapsed_seconds += self.control_timestep
        reward_images = observations["wrist"].copy()
        for index, done in enumerate(dones):
            if done:
                terminal_observation = infos[index].get("terminal_observation")
                if terminal_observation is None:
                    raise RuntimeError(
                        "A completed vectorized environment did not provide its "
                        "terminal observation for imitation reward scoring."
                    )
                reward_images[index] = terminal_observation["wrist"]
        rewards, indices = self._similarity_rewards(reward_images)
        gripper_rewards = self._gripper_reward(
            np.asarray([observations["proprio"].squeeze()])
        )
        z_rewards = self._z_reward(np.asarray([observations["proprio"].squeeze()]))
        for info, timestamp, index, similarity, gripper_reward, z_reward in zip(
            infos,
            self._elapsed_seconds,
            indices,
            rewards,
            gripper_rewards,
            z_rewards,
        ):
            success_reward = float(bool(info.get("is_success", False)))
            reward = float(similarity) + success_reward
            info["human_timestamp"] = float(timestamp)
            info["human_frame_index"] = int(index)
            info["similarity_reward"] = float(similarity)
            info["success_reward"] = success_reward
            info["imitation_reward"] = float(reward)
            info["gripper_reward"] = float(gripper_reward)
            info["z_reward"] = float(z_reward)
            info["z_target"] = float(self._z_supervision[index])

        rewards = np.exp(rewards) - 1
        rewards *= self.similarity_weight
        rewards += (
            np.asarray([info["success_reward"] for info in infos], dtype=np.float32)
            * self.success_weight
            + np.asarray([info["gripper_reward"] for info in infos], dtype=np.float32)
            * self.gripper_weight
            + np.asarray([info["z_reward"] for info in infos], dtype=np.float32)
            * self.z_weight
        )
        self._elapsed_seconds[dones] = 0.0
        return observations, rewards, dones, infos

    def _similarity_rewards(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._encoder is None or self._normalized_human_embeddings is None:
            raise RuntimeError(
                "Set the DINOv3 encoder before stepping the environment."
            )
        indices = self._nearest_indices(self._elapsed_seconds)
        image_tensor = torch.as_tensor(
            images, device=self._normalized_human_embeddings.device
        )
        if image_tensor.ndim != 4:
            raise ValueError(
                f"Expected a batched image tensor, got {image_tensor.shape}."
            )
        if image_tensor.shape[-1] == 3:
            image_tensor = image_tensor.permute(0, 3, 1, 2)
        elif image_tensor.shape[1] != 3:
            raise ValueError(
                "Expected agentview images with three channels in BCHW or BHWC layout; "
                f"got {tuple(image_tensor.shape)}."
            )
        self._encoder.backbone.eval()
        with torch.inference_mode():
            robot_embeddings = F.normalize(
                self._encoder.global_embedding(image_tensor), dim=1
            )
            targets = self._normalized_human_embeddings[
                torch.as_tensor(indices, device=robot_embeddings.device)
            ]
            similarity = F.cosine_similarity(robot_embeddings, targets, dim=1)
            rewards = similarity
        return rewards.cpu().numpy().astype(np.float32), indices

    def _nearest_indices(self, timestamps: np.ndarray) -> np.ndarray:
        right = np.searchsorted(self.timestamps, timestamps, side="left")
        right = np.clip(right, 0, len(self.timestamps) - 1)
        left = np.clip(right - 1, 0, len(self.timestamps) - 1)
        choose_left = np.abs(timestamps - self.timestamps[left]) <= np.abs(
            self.timestamps[right] - timestamps
        )
        return np.where(choose_left, left, right)

    def _gripper_reward(self, robot_state: np.ndarray) -> np.ndarray:
        robot_state = np.asarray(robot_state).squeeze()
        if robot_state.ndim != 2:
            robot_state = robot_state.reshape(1, -1)
        indices = self._nearest_indices(self._elapsed_seconds)
        gripper_target = self._gripper_supervision[
            torch.as_tensor(indices, device=self._gripper_supervision.device)
        ]

        # Gripper state is the last element of the proprioceptive observation
        gripper_state = robot_state[:, -1]
        gripper_open = np.abs(gripper_state) > 1.8e-2

        gripper_err = np.abs(gripper_open - gripper_target)

        # Reward is higher when the gripper state matches the target
        return np.exp(-gripper_err * 5).astype(np.float32)

    def _z_reward(self, robot_state: np.ndarray) -> np.ndarray:
        indices = self._nearest_indices(self._elapsed_seconds)
        robot_state = np.asarray(robot_state).squeeze()
        if robot_state.ndim != 2:
            robot_state = robot_state.reshape(1, -1)
        z_targets = self._z_supervision[
            torch.as_tensor(indices, device=self._z_supervision.device)
        ]

        # Z position is the third-to-last element of the proprioceptive observation
        z_position = robot_state[:, 2]
        z_position = (z_position - self.z_min) / (self.z_max - self.z_min)
        z_err = np.abs(z_position - z_targets)

        return np.exp(-z_err).astype(np.float32)
