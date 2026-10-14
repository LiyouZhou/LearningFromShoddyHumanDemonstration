#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
import json
import os
import subprocess

import cv2
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from encode_human_video import load_video_embeddings
from imitation_reward import DINOv3SimilarityReward
from policy import DINOv3VisuomotorPolicy

IMG_SIZE = 224
CONTROL_FREQ = 20
PROPRIO_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


class LiberoEnv(gym.Env):
    """Gymnasium wrapper around a single LIBERO task with image + proprio observations."""

    def __init__(self, suite="libero_goal", task_id=8, max_steps=100, seed=0):
        task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        bddl = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=IMG_SIZE,
            camera_widths=IMG_SIZE,
            control_freq=CONTROL_FREQ,
        )
        self.env.seed(seed)
        self.max_steps = max_steps
        self.task_description = task.language
        self._t = 0

        img = spaces.Box(0, 255, (IMG_SIZE, IMG_SIZE, 3), np.uint8)
        proprio_dim = 3 + 4 + 2
        self.observation_space = spaces.Dict(
            {
                "agentview": img,
                "wrist": img,
                "proprio": spaces.Box(-np.inf, np.inf, (proprio_dim,), np.float32),
                "step": spaces.Box(0, max_steps, (1,), np.float32),
            }
        )
        # Agent controls XYZ and gripper; rotational deltas stay fixed at zero.
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)

    def _obs(self, raw):
        wrist = np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1])

        return {
            "agentview": np.ascontiguousarray(raw["agentview_image"][::-1]),
            "wrist": wrist,
            "proprio": np.concatenate([raw[k] for k in PROPRIO_KEYS]).astype(
                np.float32
            ),
            "step": np.array([self._t], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        raw = self.env.reset()
        # Let objects settle
        for _ in range(5):
            raw, *_ = self.env.step(np.zeros(7))
        return self._obs(raw), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (4,):
            raise ValueError(f"Expected a four-dimensional action, got {action.shape}.")
        libero_action = np.zeros(7, dtype=np.float64)
        libero_action[:3] = action[:3]
        libero_action[6] = action[3]
        raw, _, done, info = self.env.step(libero_action)
        self._t += 1
        success = bool(self.env.check_success())
        terminated = success or bool(done)
        truncated = self._t >= self.max_steps and not terminated
        info["is_success"] = success
        return self._obs(raw), 0.0, terminated, truncated, info

    def close(self):
        self.env.close()


def make_env(rank, suite, task_id):
    return lambda: LiberoEnv(suite=suite, task_id=task_id, seed=rank)


class ConfigLogCallback(BaseCallback):
    """Show the run config in TensorBoard's HParams and Text tabs."""

    def __init__(self, config):
        super().__init__()
        self.config = config

    def _on_training_start(self):
        hparams = {
            k: v if isinstance(v, (bool, int, float, str)) else str(v)
            for k, v in self.config.items()
        }
        self.logger.record(
            "hparams",
            HParam(hparams, {"rollout/ep_rew_mean": 0.0}),
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.record(
            "config",
            "```json\n" + json.dumps(self.config, indent=2) + "\n```",
            exclude=("stdout", "log", "json", "csv"),
        )
        self.logger.dump(0)

    def _on_step(self):
        return True


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class InfoLogCallback(BaseCallback):
    """Append every env's info dict at every training step to infos.jsonl."""

    # Image-sized payloads that would bloat the log.
    SKIP_KEYS = ("terminal_observation",)

    def __init__(self, run_dir):
        super().__init__()
        self.path = os.path.join(run_dir, "infos.jsonl")
        self.file = None

    def _on_training_start(self):
        self.file = open(self.path, "a", buffering=1)

    def _on_step(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        for env_idx, info in enumerate(self.locals["infos"]):
            record = {
                "num_timesteps": self.num_timesteps,
                "env": env_idx,
                "reward": float(rewards[env_idx]),
                "done": bool(dones[env_idx]),
                **{k: v for k, v in info.items() if k not in self.SKIP_KEYS},
            }
            self.file.write(json.dumps(record, default=_json_default) + "\n")
        return True

    def _on_training_end(self):
        if self.file is not None:
            self.file.close()


class VideoEvalCallback(BaseCallback):
    def __init__(self, eval_env, run_dir, eval_freq=50_000, n_eval_episodes=3):
        super().__init__()
        if eval_freq <= 0 or n_eval_episodes <= 0:
            raise ValueError("eval_freq and n_eval_episodes must be positive.")
        self.eval_env = eval_env
        self.video_root = os.path.join(run_dir, "eval_videos")
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.next_eval = eval_freq

    def _on_step(self):
        if self.num_timesteps < self.next_eval:
            return True
        while self.next_eval <= self.num_timesteps:
            self.next_eval += self.eval_freq
        self._evaluate()
        return True

    def _evaluate(self):
        step_dir = os.path.join(self.video_root, f"step_{self.num_timesteps:09d}")
        os.makedirs(step_dir, exist_ok=True)
        csv_path = os.path.join(step_dir, "results.csv")
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "episode",
                    "is_success",
                    "imitation_reward",
                    "gripper_reward",
                    "z_reward",
                    "z_target",
                    "robot_state",
                ]
            )
            successes = 0

            for episode in range(self.n_eval_episodes):
                observation = self.eval_env.reset()
                video_path = os.path.join(step_dir, f"episode_{episode:02d}.mp4")
                temporary_video_path = os.path.splitext(video_path)[0] + "_opencv.mp4"
                first_frame = self._video_frame(observation)
                height, width = first_frame.shape[:2]
                writer = cv2.VideoWriter(
                    temporary_video_path,
                    cv2.VideoWriter.fourcc(*"mp4v"),
                    CONTROL_FREQ,
                    (width, height),
                )
                success = False
                try:
                    try:
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"Could not open evaluation video writer: {temporary_video_path}"
                            )
                        writer.write(first_frame)
                        dones = [False]
                        while not (dones[0]):
                            action, _ = self.model.predict(
                                observation, deterministic=True
                            )
                            observation, rewards, dones, infos = self.eval_env.step(
                                action
                            )
                            info = infos[0]
                            writer.write(self._video_frame(observation))
                            success = rewards[0] > 100
                            csv_writer.writerow(
                                [
                                    episode,
                                    int(success),
                                    info.get("imitation_reward", -0.0),
                                    info.get("gripper_reward", -0.0),
                                    info.get("z_reward", -0.0),
                                    info.get("z_target", -0.0),
                                    str(observation["proprio"][0]),
                                ]
                            )
                    finally:
                        writer.release()

                    subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            temporary_video_path,
                            "-c:v",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            "-an",
                            video_path,
                        ],
                        check=True,
                    )
                finally:
                    if os.path.exists(temporary_video_path):
                        os.remove(temporary_video_path)

                successes += int(success)

        print(
            f"Evaluation at {self.num_timesteps} steps: "
            f"{successes}/{self.n_eval_episodes} successful; videos: {step_dir}"
        )

    @staticmethod
    def _video_frame(observation):
        views = np.concatenate(
            (observation["agentview"][0], observation["wrist"][0]), axis=1
        )
        return cv2.cvtColor(views, cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train an imitation-reward LIBERO policy."
    )
    parser.add_argument("--suite", type=str, default="libero_goal")
    parser.add_argument("--task-id", type=int, default=8)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--dino-checkpoint",
        type=str,
        default=os.environ.get(
            "DINO_CHECKPOINT",
            os.path.join(torch.hub.get_dir(), "checkpoints", "EUPE-ViT-B.pt"),
        ),
    )
    parser.add_argument(
        "--human-demo-embeddings",
        type=str,
        default=os.environ.get("HUMAN_DEMO_EMBEDDINGS", "human_demo_embeddings.npy"),
    )
    parser.add_argument(
        "--run-dir", type=str, default=os.environ.get("RUN_DIR", "./runs")
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(os.environ.get("TOTAL_TIMESTEPS", 500_000)),
    )
    parser.add_argument(
        "--ckpt-freq", type=int, default=int(os.environ.get("CKPT_FREQ", 10_000))
    )
    parser.add_argument(
        "--eval-freq", type=int, default=int(os.environ.get("EVAL_FREQ", 10_000))
    )

    parser.add_argument("--similarity-weight", type=float, default=1.0)
    parser.add_argument("--success-weight", type=float, default=100.0)
    parser.add_argument("--gripper-weight", type=float, default=0.5)
    parser.add_argument("--z-weight", type=float, default=2.0)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.environ.get("LEARNING_RATE", 3e-4)),
    )
    parser.add_argument(
        "--pretrained-policy",
        type=str,
        default=None,
        help="Path to a saved PPO .zip whose weights initialise the policy.",
    )

    args = parser.parse_args()

    suite = args.suite
    task_id = args.task_id
    n_envs = args.n_envs
    dino_checkpoint = args.dino_checkpoint
    human_demo_embeddings_path = args.human_demo_embeddings
    human_demo_embeddings = load_video_embeddings(human_demo_embeddings_path)
    print(
        f"Loaded {len(human_demo_embeddings)} human demonstration embeddings from "
        f"{human_demo_embeddings_path}."
    )

    imitation_reward_env = DINOv3SimilarityReward(
        SubprocVecEnv([make_env(i, suite, task_id) for i in range(n_envs)]),
        human_embeddings=human_demo_embeddings,
        control_timestep=1.0 / CONTROL_FREQ,
        similarity_weight=args.similarity_weight,
        success_weight=args.success_weight,
        gripper_weight=args.gripper_weight,
        z_weight=args.z_weight,
    )
    vec_env = VecMonitor(imitation_reward_env)
    run_dir = args.run_dir
    if run_dir == "./runs":
        subdir = datetime.now().strftime("%d%m%y_%H%M%S")
        run_dir = os.path.join(run_dir, subdir)
    os.makedirs(run_dir, exist_ok=True)
    config = {**vars(args), "run_dir": run_dir, "control_freq": CONTROL_FREQ}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    eval_callback = VideoEvalCallback(
        imitation_reward_env, run_dir, eval_freq=args.eval_freq
    )

    model = PPO(
        DINOv3VisuomotorPolicy,
        vec_env,
        policy_kwargs={
            "features_extractor_kwargs": {
                "model_name": "dinov3_vitb16",
                "freeze_backbone": True,
                "checkpoint_path": dino_checkpoint,
            },
        },
        n_steps=256,
        batch_size=128,
        learning_rate=args.learning_rate,
        ent_coef=args.ent_coef,
        verbose=1,
        tensorboard_log=run_dir,
        device="auto",
    )
    if args.pretrained_policy:
        model.set_parameters(args.pretrained_policy, exact_match=False)
        print(f"Loaded pre-trained policy from {args.pretrained_policy}.")
    imitation_reward_env.set_encoder(model.policy.features_extractor.vision_encoder)
    ckpt = CheckpointCallback(
        save_freq=args.ckpt_freq // n_envs,
        save_path=run_dir,
        name_prefix="ppo_libero",
    )
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[
                ConfigLogCallback(config),
                InfoLogCallback(run_dir),
                ckpt,
                eval_callback,
            ],
            progress_bar=False,
        )
        model.save("ppo_libero_visuomotor")
    finally:
        vec_env.close()
