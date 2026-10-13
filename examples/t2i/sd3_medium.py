"""Self-contained TrajCFG inference for Stable Diffusion 3 Medium."""

from __future__ import annotations

import argparse
import inspect
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import StableDiffusion3Pipeline

try:
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
except ImportError:
    retrieve_timesteps = None

try:
    from diffusers.utils.torch_utils import randn_tensor
except ImportError:
    randn_tensor = None


MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
PROMPT = "a photo of a dog right of a teddy bear"
OUTPUT = "outputs/t2i_sd3_medium.png"

HEIGHT = 1024
WIDTH = 1024
NUM_INFERENCE_STEPS = 28
SEED = 42

BASE_SCALE = 7.0
MIN_SCALE = 6.0
MAX_SCALE = 9.0
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--no-norm-rescale", action="store_true")
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    return parser.parse_args()


class TrajectoryFeedback:
    """Token-wise adjacent/stride-2 feedback for spatial SD3 latents."""

    def __init__(self, *, base_scale: float, min_scale: float, max_scale: float, rho_lo: float) -> None:
        if not 0.0 <= min_scale <= max_scale:
            raise ValueError("Expected 0 <= min_scale <= max_scale.")
        if not -1.0 <= rho_lo < 1.0:
            raise ValueError("rho_lo must be in [-1, 1).")
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rho_lo = rho_lo
        self.velocities: deque[torch.Tensor] = deque(maxlen=3)
        self.timesteps: deque[float] = deque(maxlen=3)
        self.stats: list[dict[str, Any]] = []

    @staticmethod
    def flatten_tokens(velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim != 4:
            raise ValueError(f"Expected [B, C, H, W], got {tuple(velocity.shape)}.")
        return velocity.detach().float().cpu().permute(0, 2, 3, 1).reshape(
            velocity.shape[0], -1, velocity.shape[1]
        )

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        numerator = (a * b).sum(dim=-1)
        denominator = a.norm(dim=-1).clamp_min(NORM_FLOOR) * b.norm(dim=-1).clamp_min(NORM_FLOOR)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def map_to_scale(self, cosine: torch.Tensor) -> torch.Tensor:
        fraction = ((1.0 - cosine) / (1.0 - self.rho_lo)).clamp(0.0, 1.0)
        return self.min_scale + (self.max_scale - self.min_scale) * fraction

    def scale_for_step(self, step: int, prediction: torch.Tensor) -> tuple[torch.Tensor, str]:
        if step < 2:
            shape = (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3])
            return torch.full(
                shape, self.base_scale, device=prediction.device, dtype=prediction.dtype
            ), "base"

        adjacent = self.map_to_scale(self.cosine(self.velocities[-2], self.velocities[-1]))
        token_scale = adjacent
        source = "adjacent"
        if len(self.velocities) == 3:
            stride2_cosine = self.cosine(self.velocities[-3], self.velocities[-1])
            delta_adjacent = abs(self.timesteps[-1] - self.timesteps[-2])
            delta_stride2 = abs(self.timesteps[-1] - self.timesteps[-3])
            if delta_adjacent > 0.0 and delta_stride2 > 0.0:
                stride2_cosine = torch.cos(
                    torch.acos(stride2_cosine) * (delta_adjacent / delta_stride2)
                ).clamp(-1.0, 1.0)
            stride2 = self.map_to_scale(stride2_cosine)
            token_scale = SHORT_WEIGHT * adjacent + STRIDE2_WEIGHT * stride2
            source = "adjacent+stride2"

        scale = token_scale.reshape(
            prediction.shape[0], prediction.shape[2], prediction.shape[3]
        ).unsqueeze(1)
        return scale.to(device=prediction.device, dtype=prediction.dtype), source

    def update(
        self,
        guided_velocity: torch.Tensor,
        timestep: torch.Tensor,
        scale: torch.Tensor,
        source: str,
    ) -> None:
        self.velocities.append(self.flatten_tokens(guided_velocity))
        self.timesteps.append(float(timestep.detach().float().cpu().flatten()[0].item()))
        value = scale.detach().float()
        self.stats.append(
            {
                "step": len(self.stats),
                "timestep": self.timesteps[-1],
                "source": source,
                "min": float(value.min().item()),
                "max": float(value.max().item()),
                "mean": float(value.mean().item()),
            }
        )


def call_supported(function: Any, **kwargs: Any) -> Any:
    target = function.forward if isinstance(function, torch.nn.Module) else function
    parameters = inspect.signature(target).parameters
    return function(**{name: value for name, value in kwargs.items() if name in parameters})


def encode_prompt(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    negative_prompt: str | None,
    device: torch.device,
    max_sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    encoded = call_supported(
        pipe.encode_prompt,
        prompt=prompt,
        prompt_2=None,
        prompt_3=None,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        negative_prompt_3=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        max_sequence_length=max_sequence_length,
    )
    if len(encoded) != 4:
        raise RuntimeError(f"Unexpected SD3 encode_prompt output length: {len(encoded)}")
    return encoded


def prepare_latents(
    pipe: StableDiffusion3Pipeline,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    channels = pipe.transformer.config.in_channels
    if hasattr(pipe, "prepare_latents"):
        return pipe.prepare_latents(1, channels, height, width, dtype, device, generator, None)
    if randn_tensor is None:
        raise RuntimeError("The installed diffusers version exposes no SD3 latent initializer.")
    scale = getattr(pipe, "vae_scale_factor", 8)
    return randn_tensor(
        (1, channels, height // scale, width // scale),
        generator=generator,
        device=device,
        dtype=dtype,
    )


def get_timesteps(
    pipe: StableDiffusion3Pipeline, steps: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    if retrieve_timesteps is not None:
        sigmas = np.linspace(1.0, 1.0 / steps, steps)
        return retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas)
    pipe.scheduler.set_timesteps(steps, device=device)
    return pipe.scheduler.timesteps, steps


def transformer_forward(
    pipe: StableDiffusion3Pipeline,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
) -> torch.Tensor:
    return call_supported(
        pipe.transformer,
        hidden_states=latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        pooled_projections=pooled_prompt_embeds,
        joint_attention_kwargs=getattr(pipe, "joint_attention_kwargs", None),
        return_dict=False,
    )[0]


@torch.no_grad()
def generate(pipe: StableDiffusion3Pipeline, args: argparse.Namespace, generator: torch.Generator):
    device = pipe._execution_device
    pipe._guidance_scale = args.base_scale
    pipe._clip_skip = None
    pipe._joint_attention_kwargs = None
    pipe._interrupt = False
    positive, negative, pooled_positive, pooled_negative = encode_prompt(
        pipe, args.prompt, args.negative_prompt, device, args.max_sequence_length
    )
    latents = prepare_latents(
        pipe, args.height, args.width, positive.dtype, device, generator
    )
    timesteps, steps = get_timesteps(pipe, args.num_inference_steps, device)
    pipe._num_timesteps = len(timesteps)
    feedback = TrajectoryFeedback(
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )

    with pipe.progress_bar(total=steps) as progress_bar:
        for step, timestep in enumerate(timesteps):
            model_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
            conditional = transformer_forward(
                pipe, latents, model_timestep, positive, pooled_positive
            )
            unconditional = transformer_forward(
                pipe, latents, model_timestep, negative, pooled_negative
            )
            scale, source = feedback.scale_for_step(step, conditional)
            guided = unconditional + scale * (conditional - unconditional)
            if not args.no_norm_rescale:
                conditional_norm = torch.norm(conditional, dim=1, keepdim=True)
                guided_norm = torch.norm(guided, dim=1, keepdim=True).clamp_min(NORM_FLOOR)
                guided = guided * (conditional_norm / guided_norm)
            feedback.update(guided, timestep, scale, source)

            latent_dtype = latents.dtype
            latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]
            if latents.dtype != latent_dtype:
                latents = latents.to(latent_dtype)
            progress_bar.update()

    latents = latents.to(pipe.vae.dtype)
    scaling_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
    shift_factor = getattr(pipe.vae.config, "shift_factor", 0.0)
    image = pipe.vae.decode(latents / scaling_factor + shift_factor, return_dict=False)[0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    pipe.maybe_free_model_hooks()
    return image, feedback.stats


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("SD3-Medium inference requires a CUDA device.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.torch_dtype]
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=dtype)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    else:
        pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image, diagnostics = generate(pipe, args, generator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "backbone": "Stable Diffusion 3 Medium",
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "output": str(args.output),
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "trajcfg": {
            "base_scale": args.base_scale,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "rho_lo": args.rho_lo,
            "adjacent_weight": SHORT_WEIGHT,
            "stride2_weight": STRIDE2_WEIGHT,
            "norm_rescale": not args.no_norm_rescale,
        },
        "per_step": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
