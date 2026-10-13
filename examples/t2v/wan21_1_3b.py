"""Self-contained TrajCFG inference for Wan2.1-T2V-1.3B."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


MODEL_ID = "/path/to/Wan2.1-T2V-1.3B-Diffusers"
PROMPT = "A car drives through a large puddle, splashing water high onto the sidewalk and drenching a nearby fire hydrant."
NEGATIVE_PROMPT = "low quality"
OUTPUT = "outputs/t2v_wan21_1_3b.mp4"

HEIGHT = 480
WIDTH = 832
NUM_FRAMES = 81
NUM_INFERENCE_STEPS = 40
FPS = 16
SEED = 0

BASE_SCALE = 6
MIN_SCALE = 5
MAX_SCALE = 10
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
LONG_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a text-conditioned video with TrajCFG.")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=NEGATIVE_PROMPT)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    return parser.parse_args()


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
    def latent_shape(token_scale: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        return token_scale.reshape(
            latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4]
        ).unsqueeze(1)

    def scale_for_step(self, step: int, latents: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if step < 2:
            scale = torch.full(
                (latents.shape[0], 1, latents.shape[2], latents.shape[3], latents.shape[4]),
                self.base_scale,
                device=latents.device,
                dtype=latents.dtype,
            )
            return scale, {"source": "base", "min": self.base_scale, "max": self.base_scale}

        q_short = self.cosine(self.velocities[-2], self.velocities[-1])
        short_scale = self.map_to_scale(q_short)
        source = "adjacent"

        if len(self.velocities) == 3:
            q_long = self.cosine(self.velocities[-3], self.velocities[-1])
            delta_short = abs(self.timesteps[-1] - self.timesteps[-2])
            delta_long = abs(self.timesteps[-1] - self.timesteps[-3])
            if delta_short > 0.0 and delta_long > 0.0:
                q_long = torch.cos(torch.acos(q_long) * (delta_short / delta_long)).clamp(-1.0, 1.0)
            long_scale = self.map_to_scale(q_long)
            token_scale = SHORT_WEIGHT * short_scale + LONG_WEIGHT * long_scale
            source = "adjacent+stride2"
        else:
            token_scale = short_scale

        scale = self.latent_shape(token_scale, latents).to(latents.device, latents.dtype)
        stats = scale.detach().float()
        return scale, {
            "source": source,
            "min": float(stats.min().item()),
            "max": float(stats.max().item()),
            "mean": float(stats.mean().item()),
        }

    def update(self, guided_velocity: torch.Tensor, timestep: torch.Tensor) -> None:
        self.velocities.append(self.flatten_tokens(guided_velocity))
        self.timesteps.append(float(timestep.detach().cpu().flatten()[0].item()))


def load_pipeline(model_id: str, device: str, cpu_offload: bool) -> WanPipeline:
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    if cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=int(device.split(":")[1]))
        return pipe
    return pipe.to(device)


def adjust_shape(pipe: WanPipeline, height: int, width: int, num_frames: int) -> tuple[int, int, int]:
    num_frames = num_frames // pipe.vae_scale_factor_temporal * pipe.vae_scale_factor_temporal + 1
    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    patch_size = transformer.config.patch_size
    h_multiple = pipe.vae_scale_factor_spatial * patch_size[1]
    w_multiple = pipe.vae_scale_factor_spatial * patch_size[2]
    return height // h_multiple * h_multiple, width // w_multiple * w_multiple, max(num_frames, 1)


@torch.no_grad()
def generate(pipe: WanPipeline, args: argparse.Namespace, generator: torch.Generator):
    pipe.check_inputs(args.prompt, args.negative_prompt, args.height, args.width, None, None, ["latents"], None)
    height, width, num_frames = adjust_shape(pipe, args.height, args.width, args.num_frames)

    forced_scale = max(args.base_scale, 1.0001)
    pipe._guidance_scale = forced_scale
    pipe._guidance_scale_2 = forced_scale if pipe.config.boundary_ratio is not None else None
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False

    device = pipe._execution_device
    transformer = pipe.transformer if pipe.transformer is not None else pipe.transformer_2
    transformer_dtype = transformer.dtype
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=device,
    )
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

    pipe.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(
        1,
        transformer.config.in_channels,
        height,
        width,
        num_frames,
        torch.float32,
        device,
        generator,
        None,
    )
    mask = torch.ones_like(latents, dtype=torch.float32)
    pipe._num_timesteps = len(timesteps)
    pipe.scheduler.set_begin_index(0)
    boundary = (
        pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
        if pipe.config.boundary_ratio is not None
        else None
    )
    feedback = TrajectoryFeedback(
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )
    diagnostics = []

    with pipe.progress_bar(total=args.num_inference_steps) as progress_bar:
        for step, timestep in enumerate(timesteps):
            pipe._current_timestep = timestep
            current_model = pipe.transformer if boundary is None or timestep >= boundary else pipe.transformer_2
            latent_input = latents.to(transformer_dtype)
            if pipe.config.expand_timesteps:
                temp_ts = (mask[0][0][:, ::2, ::2] * timestep).flatten()
                model_timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                model_timestep = timestep.expand(latents.shape[0])

            with current_model.cache_context("cond"):
                velocity_cond = current_model(
                    hidden_states=latent_input,
                    timestep=model_timestep,
                    encoder_hidden_states=prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]
            with current_model.cache_context("uncond"):
                velocity_ref = current_model(
                    hidden_states=latent_input,
                    timestep=model_timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    attention_kwargs=None,
                    return_dict=False,
                )[0]

            scale, stats = feedback.scale_for_step(step, latents)
            guided_velocity = velocity_ref + scale * (velocity_cond - velocity_ref)
            feedback.update(guided_velocity, timestep)
            diagnostics.append({"step": step, "timestep": feedback.timesteps[-1], **stats})
            latents = pipe.scheduler.step(guided_velocity, timestep, latents, return_dict=False)[0]
            progress_bar.update()

    pipe._current_timestep = None
    latents = latents.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latents_mean = latents_mean.to(latents.device, latents.dtype)
    latents_std = latents_std.to(latents.device, latents.dtype)
    video = pipe.vae.decode(latents / latents_std + latents_mean, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type="np")[0]
    pipe.maybe_free_model_hooks()
    return frames, diagnostics


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required.")
    device = f"cuda:{args.gpu}"
    pipe = load_pipeline(args.model_id, device, args.cpu_offload)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    frames, diagnostics = generate(pipe, args, generator)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(output), fps=args.fps)
    metadata = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "requested_shape": [args.num_frames, args.height, args.width],
        "actual_shape": [len(frames), frames[0].shape[0], frames[0].shape[1]],
        "num_inference_steps": args.num_inference_steps,
        "trajcfg": {
            "base_scale": args.base_scale,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "rho_lo": args.rho_lo,
            "short_weight": SHORT_WEIGHT,
            "long_weight": LONG_WEIGHT,
        },
        "scale_stats_per_step": diagnostics,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved video to {output}")


if __name__ == "__main__":
    main()
