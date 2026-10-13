"""Self-contained TrajCFG inference for Qwen-Image-2512."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from diffusers import QwenImagePipeline

try:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps
except ImportError:
    calculate_shift = None
    retrieve_timesteps = None

try:
    from diffusers.utils.torch_utils import randn_tensor
except ImportError:
    randn_tensor = None


MODEL_ID = "/path/to/Qwen-Image-2512"
PROMPT = "a photo of three baseball bats"
NEGATIVE_PROMPT = (
    "低分辨率，低画质，肢体畸形，手指畸形，画面过饱和，蜡像感，人脸无细节，"
    "过度光滑，画面具有AI感。构图混乱。文字模糊，扭曲。"
)
OUTPUT = "outputs/t2i_qwen_image.png"

HEIGHT = 1328
WIDTH = 1328
NUM_INFERENCE_STEPS = 30
SEED = 42

BASE_SCALE = 5.0
MIN_SCALE = 4.0
MAX_SCALE = 9.0
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--embedded-guidance-scale", type=float, default=None)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    return parser.parse_args()


class TrajectoryFeedback:
    """Token-wise adjacent/stride-2 feedback for packed Qwen image latents."""

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
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        numerator = (a * b).sum(dim=-1)
        denominator = a.norm(dim=-1).clamp_min(NORM_FLOOR) * b.norm(dim=-1).clamp_min(NORM_FLOOR)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def map_to_scale(self, cosine: torch.Tensor) -> torch.Tensor:
        fraction = ((1.0 - cosine) / (1.0 - self.rho_lo)).clamp(0.0, 1.0)
        return self.min_scale + (self.max_scale - self.min_scale) * fraction

    def scale_for_step(self, step: int, prediction: torch.Tensor) -> tuple[torch.Tensor, str]:
        if prediction.ndim != 3:
            raise ValueError(f"Expected packed latent shape [B, L, C], got {tuple(prediction.shape)}.")
        if step < 2:
            return torch.full(
                (prediction.shape[0], prediction.shape[1], 1),
                self.base_scale,
                device=prediction.device,
                dtype=prediction.dtype,
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

        return token_scale.unsqueeze(-1).to(
            device=prediction.device, dtype=prediction.dtype
        ), source

    def update(
        self,
        guided_velocity: torch.Tensor,
        timestep: torch.Tensor,
        scale: torch.Tensor,
        source: str,
    ) -> None:
        self.velocities.append(guided_velocity.detach().float().cpu())
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


def prepare_latents(
    pipe: QwenImagePipeline,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    channels = pipe.transformer.config.in_channels // 4
    if hasattr(pipe, "prepare_latents"):
        return pipe.prepare_latents(1, channels, height, width, dtype, device, generator, None)
    if randn_tensor is None:
        raise RuntimeError("The installed diffusers version exposes no Qwen latent initializer.")
    scale = pipe.vae_scale_factor
    latent_height = 2 * (height // (scale * 2))
    latent_width = 2 * (width // (scale * 2))
    latent = randn_tensor(
        (1, 1, channels, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return pipe._pack_latents(latent, 1, channels, latent_height, latent_width)


def get_timesteps(
    pipe: QwenImagePipeline,
    steps: int,
    device: torch.device,
    image_seq_len: int,
) -> tuple[torch.Tensor, int]:
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    if calculate_shift is None or retrieve_timesteps is None:
        pipe.scheduler.set_timesteps(steps, device=device)
        return pipe.scheduler.timesteps, steps
    config = pipe.scheduler.config
    shift = calculate_shift(
        image_seq_len,
        config.get("base_image_seq_len", 256),
        config.get("max_image_seq_len", 4096),
        config.get("base_shift", 0.5),
        config.get("max_shift", 1.15),
    )
    return retrieve_timesteps(pipe.scheduler, steps, device, sigmas=sigmas, mu=shift)


@torch.no_grad()
def generate(pipe: QwenImagePipeline, args: argparse.Namespace, generator: torch.Generator):
    device = pipe._execution_device
    pipe.check_inputs(
        args.prompt,
        args.height,
        args.width,
        negative_prompt=args.negative_prompt,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=args.max_sequence_length,
    )
    pipe._guidance_scale = args.embedded_guidance_scale
    pipe._attention_kwargs = {}
    pipe._current_timestep = None
    pipe._interrupt = False

    positive, positive_mask = pipe.encode_prompt(
        prompt=args.prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    negative, negative_mask = pipe.encode_prompt(
        prompt=args.negative_prompt,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    latents = prepare_latents(
        pipe, args.height, args.width, positive.dtype, device, generator
    )
    image_shapes = [[(
        1,
        args.height // pipe.vae_scale_factor // 2,
        args.width // pipe.vae_scale_factor // 2,
    )]]
    timesteps, steps = get_timesteps(
        pipe, args.num_inference_steps, device, latents.shape[1]
    )
    pipe._num_timesteps = len(timesteps)
    pipe.scheduler.set_begin_index(0)

    if pipe.transformer.config.guidance_embeds:
        if args.embedded_guidance_scale is None:
            raise ValueError("This checkpoint requires --embedded-guidance-scale.")
        guidance = torch.full(
            (latents.shape[0],),
            args.embedded_guidance_scale,
            device=device,
            dtype=torch.float32,
        )
    else:
        guidance = None

    feedback = TrajectoryFeedback(
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )
    with pipe.progress_bar(total=steps) as progress_bar:
        for step, timestep in enumerate(timesteps):
            pipe._current_timestep = timestep
            model_timestep = timestep.expand(latents.shape[0]).to(latents.dtype) / 1000
            with pipe.transformer.cache_context("cond"):
                conditional = pipe.transformer(
                    hidden_states=latents,
                    timestep=model_timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=positive_mask,
                    encoder_hidden_states=positive,
                    img_shapes=image_shapes,
                    attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]
            with pipe.transformer.cache_context("uncond"):
                unconditional = pipe.transformer(
                    hidden_states=latents,
                    timestep=model_timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_mask,
                    encoder_hidden_states=negative,
                    img_shapes=image_shapes,
                    attention_kwargs=pipe.attention_kwargs,
                    return_dict=False,
                )[0]

            scale, source = feedback.scale_for_step(step, conditional)
            guided = unconditional + scale * (conditional - unconditional)
            conditional_norm = torch.norm(conditional, dim=-1, keepdim=True)
            guided_norm = torch.norm(guided, dim=-1, keepdim=True).clamp_min(NORM_FLOOR)
            guided = guided * (conditional_norm / guided_norm)
            feedback.update(guided, timestep, scale, source)

            latent_dtype = latents.dtype
            latents = pipe.scheduler.step(guided, timestep, latents, return_dict=False)[0]
            if latents.dtype != latent_dtype:
                latents = latents.to(latent_dtype)
            progress_bar.update()

    pipe._current_timestep = None
    latents = pipe._unpack_latents(
        latents, args.height, args.width, pipe.vae_scale_factor
    ).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    inverse_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1, pipe.vae.config.z_dim, 1, 1, 1
    ).to(latents.device, latents.dtype)
    image = pipe.vae.decode(latents / inverse_std + mean, return_dict=False)[0][:, :, 0]
    image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    pipe.maybe_free_model_hooks()
    return image, feedback.stats


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Qwen-Image inference requires a CUDA device.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    pipe = QwenImagePipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    else:
        pipe = pipe.to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image, diagnostics = generate(pipe, args, generator)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    metadata = {
        "backbone": "Qwen-Image-2512",
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
            "norm_rescale": True,
        },
        "per_step": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
