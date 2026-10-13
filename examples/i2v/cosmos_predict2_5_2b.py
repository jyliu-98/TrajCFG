#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained TrajCFG inference for Cosmos-Predict2.5-2B image-to-video.

Run this script in an installed Cosmos-Predict2.5 environment. The official
Video2WorldInference path still handles input preparation, conditioning,
offloading, and decoding; only its denoising sampler is replaced by TrajCFG.
"""

from __future__ import annotations

import argparse
import json
import types
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
import tqdm
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir
from einops import rearrange

from cosmos_predict2._src.imaginaire.flags import INTERNAL
from cosmos_predict2._src.imaginaire.utils import misc
from cosmos_predict2._src.imaginaire.utils.context_parallel import (
    broadcast_split_tensor,
    cat_outputs_cp,
    find_split,
)
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2.config import (
    DEFAULT_NEGATIVE_PROMPT,
    InferenceArguments,
    InferenceType,
    SetupArguments,
    is_rank0,
)


MODEL = "2B/post-trained"
CHECKPOINT_PATH = "checkpoints/Cosmos-Predict2.5-2B"
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_IMAGE = str(REPO_ROOT / "assets/i2v/cosmos_predict2_5.jpeg")
PROMPT = "A close-up view of a golden, translucent honeycomb-like substance, likely honeycomb or a similar gelatinous material, set against a stark black background. The honeycomb's intricate, honeycomb-like structure is highlighted by the lighting, which emphasizes its glossy, reflective quality. A thin stream of honey is seen dripping from the top left corner, cascading down onto the honeycomb's surface. The honey appears thick and viscous, clinging to itself as it flows, creating small droplets and ripples upon contact with the textured surface below. The honey's movement is slow and deliberate, showcasing the fluidity and translucency of the material as it interacts with the honeycomb. The camera remains steady throughout, focusing on the interaction between the dripping honey and the honeycomb, capturing the mesmerizing beauty and complexity of the scene. As the video progresses, the honey continues to drip and spread across the honeycomb, creating dynamic patterns and textures. The honeycomb's intricate structure and the glossy, reflective quality of the material remain prominently displayed, maintaining the mesmerizing and delicate nature of the interaction until the end of the video."
OUTPUT = "outputs/i2v_cosmos_predict2_5_2b.mp4"

NUM_FRAMES = 81
NUM_INFERENCE_STEPS = 35
FPS = 16
SEED = 0

BASE_SCALE = 5.0
MIN_SCALE = 4.0
MAX_SCALE = 9.0
RHO_LO = 0.90
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", default=CHECKPOINT_PATH)
    parser.add_argument("--image", default=INPUT_IMAGE)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--resolution", default="none", help="Use H,W or none for the model default.")
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--enable-guardrails", action="store_true")
    return parser.parse_args()


def resolve_checkpoint_path(path_string: str) -> Path:
    """Resolve a model repository directory to its post-trained checkpoint."""
    path = Path(path_string).expanduser()
    if path.is_file():
        return path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint path does not exist: {path}. Pass the downloaded Cosmos-Predict2.5-2B "
            "repository or its concrete *_ema_bf16.pt file with --checkpoint-path."
        )

    search_roots = [path / "base" / "post-trained", path]
    candidates: list[Path] = []
    for root in search_roots:
        if root.is_dir():
            candidates.extend(sorted(root.glob("*_ema_bf16.pt")))
            candidates.extend(sorted(root.glob("*.pt")))
    if not candidates:
        raise FileNotFoundError(f"No .pt checkpoint found under {path}.")
    return candidates[0].resolve()


def resolve_input_image(image: str, output_dir: Path) -> Path:
    """Download the default project-page image, or validate a local image path."""
    if image.startswith(("http://", "https://")):
        cache_path = output_dir / "inputs" / "pai_g_cosmos.jpeg"
        if is_rank0() and not cache_path.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(image, cache_path)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return cache_path.resolve()

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Input image not found: {path}")
    return path


class TrajectoryFeedback:
    """Eqs. (6)-(11): token-wise adjacent/stride-2 trajectory feedback."""

    def __init__(self, *, base_scale: float, min_scale: float, max_scale: float, rho_lo: float):
        if not 0.0 <= min_scale < max_scale:
            raise ValueError("Expected 0 <= min_scale < max_scale.")
        if not -1.0 <= rho_lo < 1.0:
            raise ValueError("rho_lo must be in [-1, 1).")
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rho_lo = rho_lo
        self.velocities: deque[torch.Tensor] = deque(maxlen=3)
        self.timesteps: deque[float] = deque(maxlen=3)
        self.step_stats: list[dict[str, Any]] = []

    @staticmethod
    def flatten_tokens(velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim != 5:
            raise ValueError(f"Expected [B, C, T, H, W], got {tuple(velocity.shape)}.")
        return velocity.detach().permute(0, 2, 3, 4, 1).reshape(velocity.shape[0], -1, velocity.shape[1])

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = a.float()
        b = b.float()
        numerator = (a * b).sum(dim=-1)
        denominator = a.norm(dim=-1).clamp_min(NORM_FLOOR) * b.norm(dim=-1).clamp_min(NORM_FLOOR)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def map_to_scale(self, cosine: torch.Tensor) -> torch.Tensor:
        fraction = ((1.0 - cosine) / (1.0 - self.rho_lo)).clamp(0.0, 1.0)
        return self.min_scale + (self.max_scale - self.min_scale) * fraction

    @staticmethod
    def latent_shape(token_scale: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return token_scale.reshape(
            velocity.shape[0], velocity.shape[2], velocity.shape[3], velocity.shape[4]
        ).unsqueeze(1)

    def scale_for_step(self, step: int, velocity: torch.Tensor) -> tuple[torch.Tensor, str]:
        if step < 2:
            scale = torch.full(
                (velocity.shape[0], 1, velocity.shape[2], velocity.shape[3], velocity.shape[4]),
                self.base_scale,
                device=velocity.device,
                dtype=velocity.dtype,
            )
            return scale, "base"

        q_short = self.cosine(self.velocities[-2], self.velocities[-1])
        short_scale = self.map_to_scale(q_short)
        token_scale = short_scale
        source = "adjacent"

        if len(self.velocities) == 3:
            q_stride2 = self.cosine(self.velocities[-3], self.velocities[-1])
            delta_short = abs(self.timesteps[-1] - self.timesteps[-2])
            delta_stride2 = abs(self.timesteps[-1] - self.timesteps[-3])
            if delta_short > 0.0 and delta_stride2 > 0.0:
                angle = torch.acos(q_stride2)
                q_stride2 = torch.cos(angle * (delta_short / delta_stride2)).clamp(-1.0, 1.0)
            stride2_scale = self.map_to_scale(q_stride2)
            token_scale = SHORT_WEIGHT * short_scale + STRIDE2_WEIGHT * stride2_scale
            source = "adjacent+stride2"

        scale = self.latent_shape(token_scale, velocity).to(device=velocity.device, dtype=velocity.dtype)
        return scale, source

    def update(self, guided_velocity: torch.Tensor, timestep: torch.Tensor, scale: torch.Tensor, source: str) -> None:
        self.velocities.append(self.flatten_tokens(guided_velocity))
        self.timesteps.append(float(timestep.detach().cpu().flatten()[0].item()))
        scale_float = scale.detach().float()
        self.step_stats.append(
            {
                "step": len(self.step_stats),
                "timestep": self.timesteps[-1],
                "source": source,
                "min": float(scale_float.min().item()),
                "max": float(scale_float.max().item()),
                "mean": float(scale_float.mean().item()),
            }
        )


def make_trajcfg_velocity_fn(
    model: Any,
    data_batch: Dict,
    feedback: TrajectoryFeedback,
    *,
    is_negative_prompt: bool,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build Cosmos' denoiser callback with token-wise TrajCFG scales."""
    num_conditional_frames = data_batch.get("num_conditional_frames", 0)
    if is_negative_prompt:
        condition, uncondition = model.conditioner.get_condition_with_negative_prompt(data_batch)
    else:
        condition, uncondition = model.conditioner.get_condition_uncondition(data_batch)

    is_image_batch = model.is_image_batch(data_batch)
    data_type = DataType.IMAGE if is_image_batch else DataType.VIDEO
    condition = condition.edit_data_type(data_type)
    uncondition = uncondition.edit_data_type(data_type)
    _, x0, _ = model.get_data_and_condition(data_batch)

    if hasattr(condition, "set_video_condition"):
        condition = condition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=model.config.conditional_frames_probs,
        )
        uncondition = uncondition.set_video_condition(
            gt_frames=x0,
            random_min_num_conditional_frames=model.config.min_num_conditional_frames,
            random_max_num_conditional_frames=model.config.max_num_conditional_frames,
            num_conditional_frames=num_conditional_frames,
            conditional_frames_probs=model.config.conditional_frames_probs,
        )
        condition = condition.edit_for_inference(
            is_cfg_conditional=True, num_conditional_frames=num_conditional_frames
        )
        uncondition = uncondition.edit_for_inference(
            is_cfg_conditional=False, num_conditional_frames=num_conditional_frames
        )

    _, condition, _, _ = model.broadcast_split_for_model_parallelsim(x0, condition, None, None)
    _, uncondition, _, _ = model.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
    step_index = 0

    def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        nonlocal step_index
        conditional_velocity = model.denoise(noise, noise_x, timestep, condition)
        unconditional_velocity = model.denoise(noise, noise_x, timestep, uncondition)
        scale, source = feedback.scale_for_step(step_index, conditional_velocity)

        # Cosmos-Predict2.5 uses cond + s * (cond - uncond).
        guided_velocity = conditional_velocity + scale * (conditional_velocity - unconditional_velocity)
        feedback.update(guided_velocity, timestep, scale, source)
        step_index += 1
        return guided_velocity

    return velocity_fn


@torch.no_grad()
def generate_samples_with_trajcfg(
    model: Any,
    data_batch: Dict,
    *,
    seed: int,
    n_sample: int | None,
    is_negative_prompt: bool,
    num_steps: int,
    feedback: TrajectoryFeedback,
    shift: float = 5.0,
    state_shape: Tuple | None = None,
) -> torch.Tensor:
    """Cosmos rectified-flow sampling loop with a TrajCFG velocity callback."""
    model._normalize_video_databatch_inplace(data_batch)
    model._augment_image_dim_inplace(data_batch)
    is_image_batch = model.is_image_batch(data_batch)
    input_key = model.input_image_key if is_image_batch else model.input_data_key
    if n_sample is None:
        n_sample = data_batch[input_key].shape[0]
    if state_shape is None:
        pixel_frames, height, width = data_batch[input_key].shape[-3:]
        state_shape = [
            model.config.state_ch,
            model.tokenizer.get_latent_num_frames(pixel_frames),
            height // model.tokenizer.spatial_compression_factor,
            width // model.tokenizer.spatial_compression_factor,
        ]

    noise = misc.arch_invariant_rand(
        (n_sample,) + tuple(state_shape), torch.float32, model.tensor_kwargs["device"], seed
    )
    generator = torch.Generator(device=model.tensor_kwargs["device"])
    generator.manual_seed(seed)
    model.sample_scheduler.set_timesteps(
        num_steps,
        device=model.tensor_kwargs["device"],
        shift=shift,
        use_kerras_sigma=model.config.use_kerras_sigma_at_inference,
    )

    velocity_fn = make_trajcfg_velocity_fn(
        model, data_batch, feedback, is_negative_prompt=is_negative_prompt
    )
    use_spatial_split = False
    if model.net.is_context_parallel_enabled:
        cp_size = len(torch.distributed.get_process_group_ranks(model.get_context_parallel_group()))
        n_views = noise.shape[2] // model.get_num_video_latent_frames()
        state_t = noise.shape[2] // n_views
        use_spatial_split = cp_size > state_t or state_t % cp_size != 0
        after_split_shape = None
        if use_spatial_split:
            after_split_shape = find_split(noise.shape, cp_size, view_factor=n_views)
            after_split_shape = torch.Size([after_split_shape[0] * n_views, *after_split_shape[1:]])
            noise = rearrange(noise, "b c t h w -> b c (t h w)")
        noise = broadcast_split_tensor(noise, seq_dim=2, process_group=model.get_context_parallel_group())
        if use_spatial_split:
            noise = rearrange(
                noise,
                "b c (t h w) -> b c t h w",
                t=after_split_shape[0],
                h=after_split_shape[1],
            )

    latents = noise
    timesteps = model.sample_scheduler.timesteps
    iterator = timesteps if INTERNAL else tqdm.tqdm(timesteps, desc="Generating samples", total=len(timesteps))
    for timestep_value in iterator:
        timestep = torch.stack([timestep_value])
        velocity = velocity_fn(noise, latents, timestep.unsqueeze(0))
        latents = model.sample_scheduler.step(
            velocity.unsqueeze(0),
            timestep_value,
            latents.unsqueeze(0),
            return_dict=False,
            generator=generator,
        )[0].squeeze(0)

    if model.net.is_context_parallel_enabled:
        if use_spatial_split:
            latents = rearrange(latents, "b c t h w -> b c (t h w)")
        latents = cat_outputs_cp(latents, seq_dim=2, cp_group=model.get_context_parallel_group())
        if use_spatial_split:
            latents = rearrange(
                latents,
                "b c (t h w) -> b c t h w",
                t=state_shape[1],
                h=state_shape[2],
            )
    return latents


def main(args: argparse.Namespace) -> None:
    output_path = Path(args.output).expanduser().resolve()
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint_path)
    input_image = resolve_input_image(args.image, output_path.parent)

    setup_args = SetupArguments.model_validate(
        {
            "output_dir": output_path.parent,
            "model": MODEL,
            "checkpoint_path": str(checkpoint_path),
            "disable_guardrails": not args.enable_guardrails,
            "offload_diffusion_model": args.cpu_offload,
            "offload_text_encoder": args.cpu_offload,
            "offload_tokenizer": args.cpu_offload,
            "offload_guardrail_models": True,
        }
    )
    init_output_dir(setup_args.output_dir, profile=False)

    from cosmos_predict2.inference import Inference

    inference = Inference(setup_args)
    feedback = TrajectoryFeedback(
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )
    original_sampler = inference.pipe.model.generate_samples_from_batch

    def trajcfg_sampler(
        model_self: Any,
        data_batch: Dict,
        guidance: float = BASE_SCALE,
        seed: int = SEED,
        state_shape: Tuple | None = None,
        n_sample: int | None = None,
        is_negative_prompt: bool = False,
        num_steps: int = NUM_INFERENCE_STEPS,
        shift: float = 5.0,
        **_kwargs: Any,
    ) -> torch.Tensor:
        del guidance
        return generate_samples_with_trajcfg(
            model_self,
            data_batch,
            seed=seed,
            n_sample=n_sample,
            is_negative_prompt=is_negative_prompt,
            num_steps=num_steps,
            feedback=feedback,
            shift=shift,
            state_shape=state_shape,
        )

    inference.pipe.model.generate_samples_from_batch = types.MethodType(
        trajcfg_sampler, inference.pipe.model
    )
    sample = InferenceArguments.model_validate(
        {
            "inference_type": InferenceType.IMAGE2WORLD.value,
            "name": output_path.stem,
            "prompt": args.prompt,
            "input_path": input_image,
            "negative_prompt": args.negative_prompt,
            "resolution": args.resolution,
            "num_output_frames": args.num_frames,
            "num_steps": args.num_inference_steps,
            # Cosmos validates this legacy scalar field even though the patched
            # sampler below ignores it in favor of the token-wise scale map.
            "guidance": max(0, min(7, round(args.base_scale))),
            "seed": args.seed,
        }
    )

    try:
        generated = inference.generate([sample], output_dir=output_path.parent)
    finally:
        inference.pipe.model.generate_samples_from_batch = original_sampler

    if is_rank0():
        metadata = {
            "model": "Cosmos-Predict2.5-2B",
            "checkpoint_path": str(checkpoint_path),
            "input_image": str(input_image),
            "prompt": args.prompt,
            "negative_prompt": args.negative_prompt,
            "seed": args.seed,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "fps": FPS,
            "trajcfg": {
                "base_scale": args.base_scale,
                "min_scale": args.min_scale,
                "max_scale": args.max_scale,
                "rho_lo": args.rho_lo,
                "adjacent_weight": SHORT_WEIGHT,
                "stride2_weight": STRIDE2_WEIGHT,
                "cfg_formula": "cond + scale * (cond - uncond)",
                "step_stats": feedback.step_stats,
            },
            "outputs": generated,
        }
        output_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved video to {output_path}")


if __name__ == "__main__":
    init_environment()
    try:
        main(parse_args())
    finally:
        cleanup_environment()
