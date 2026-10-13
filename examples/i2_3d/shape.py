"""Self-contained TrajCFG inference for Hunyuan3D-2 shape generation."""

from __future__ import annotations

import argparse
import json
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline


MODEL_PATH = "tencent/Hunyuan3D-2"
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_IMAGE = str(REPO_ROOT / "assets/i2_3d/386.jpeg")
OUTPUT = "outputs/i2_3d_shape.glb"

NUM_INFERENCE_STEPS = 50
SEED = 0

BASE_SCALE = 7.0
MIN_SCALE = 5.0
MAX_SCALE = 9.0
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--image", default=INPUT_IMAGE)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument(
        "--keep-background",
        action="store_true",
        help="Keep the original image alpha instead of removing a near-white background.",
    )
    return parser.parse_args()


def load_image(source: str, remove_white_background: bool = True) -> Image.Image:
    if source.startswith(("http://", "https://")):
        with urlopen(source, timeout=60) as response:
            image = Image.open(BytesIO(response.read())).convert("RGBA")
    else:
        image = Image.open(source).convert("RGBA")

    if not remove_white_background:
        return image

    rgba = np.asarray(image).copy()
    rgb = rgba[..., :3].astype(np.float32)
    corners = np.concatenate(
        [rgb[:12, :12].reshape(-1, 3), rgb[:12, -12:].reshape(-1, 3),
         rgb[-12:, :12].reshape(-1, 3), rgb[-12:, -12:].reshape(-1, 3)]
    )
    background = np.median(corners, axis=0)
    foreground = np.linalg.norm(rgb - background, axis=-1) > 18.0
    rgba[..., 3] = np.where(foreground, rgba[..., 3], 0)
    return Image.fromarray(rgba, mode="RGBA")


class TrajectoryFeedback:
    """Token-wise adjacent/stride-2 trajectory feedback for shape latents."""

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

        return token_scale.unsqueeze(-1).to(device=prediction.device, dtype=prediction.dtype), source

    def update(self, velocity: torch.Tensor, timestep: torch.Tensor, scale: torch.Tensor, source: str) -> None:
        self.velocities.append(velocity.detach().float().cpu())
        value = float(timestep.detach().float().cpu().flatten()[0].item())
        self.timesteps.append(value)
        scale_float = scale.detach().float()
        self.stats.append(
            {
                "step": len(self.stats),
                "timestep": value,
                "source": source,
                "min": float(scale_float.min().item()),
                "max": float(scale_float.max().item()),
                "mean": float(scale_float.mean().item()),
                "std": float(scale_float.std(unbiased=False).item()),
            }
        )


@torch.inference_mode()
def generate_shape(
    pipe: Hunyuan3DDiTFlowMatchingPipeline,
    image: Image.Image,
    feedback: TrajectoryFeedback,
    *,
    num_inference_steps: int,
    generator: torch.Generator,
    octree_resolution: int,
    num_chunks: int,
    mc_level: float,
):
    pipe.set_surface_extractor(None)
    cond_inputs = pipe.prepare_image(image)
    prepared_image = cond_inputs.pop("image")
    cond = pipe.encode_cond(
        image=prepared_image,
        additional_cond_inputs=cond_inputs,
        do_classifier_free_guidance=True,
        dual_guidance=False,
    )

    sigmas = np.linspace(0, 1, num_inference_steps)
    pipe.scheduler.set_timesteps(sigmas=sigmas, device=pipe.device)
    timesteps = pipe.scheduler.timesteps
    latents = pipe.prepare_latents(prepared_image.shape[0], pipe.dtype, pipe.device, generator)

    for step, timestep in enumerate(tqdm(timesteps, desc="Shape sampling")):
        latent_input = torch.cat([latents] * 2)
        normalized_timestep = (
            timestep.expand(latent_input.shape[0]).to(latents.dtype)
            / pipe.scheduler.config.num_train_timesteps
        )
        prediction = pipe.model(latent_input, normalized_timestep, cond, guidance=None)
        prediction_cond, prediction_uncond = prediction.chunk(2)
        scale, source = feedback.scale_for_step(step, prediction_uncond)
        guided_velocity = prediction_uncond + scale * (prediction_cond - prediction_uncond)
        feedback.update(guided_velocity, timestep, scale, source)
        latents = pipe.scheduler.step(guided_velocity, timestep, latents).prev_sample

    return pipe._export(
        latents,
        "trimesh",
        1.01,
        mc_level,
        num_chunks,
        octree_resolution,
        None,
        enable_pbar=True,
    )[0]


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 3:
        raise ValueError("--num-inference-steps must be at least 3.")
    if min(args.base_scale, args.min_scale, args.max_scale) < 1.0:
        raise ValueError("CFG scales must be at least 1.0.")
    if args.min_scale > args.max_scale:
        raise ValueError("--min-scale must not exceed --max-scale.")

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(args.model_path, device=device)
    image = load_image(args.image, remove_white_background=not args.keep_background)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    feedback = TrajectoryFeedback(
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )

    mesh = generate_shape(
        pipe,
        image,
        feedback,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
        mc_level=args.mc_level,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)

    metadata = {
        "task": "image-to-3d shape generation",
        "model_path": args.model_path,
        "image": args.image,
        "output": str(args.output),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "base_scale": args.base_scale,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "rho_lo": args.rho_lo,
        "fusion_weights": [SHORT_WEIGHT, STRIDE2_WEIGHT],
        "scale_stats": feedback.stats,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
