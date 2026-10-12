"""Self-contained TrajCFG inference for Hunyuan3D-2 texture generation."""

from __future__ import annotations

import argparse
import inspect
import json
import types
from collections import deque
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import numpy as np
import torch
import trimesh
from PIL import Image

from hy3dgen.texgen import Hunyuan3DPaintPipeline


MODEL_PATH = "tencent/Hunyuan3D-2"
PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0"
INPUT_IMAGE = (
    "https://raw.githubusercontent.com/jyliu-98/trajcfg-project-page/main/"
    "assets/gallery/i2-3d/386.jpeg"
)
INPUT_MESH = "outputs/i2_3d_shape.glb"
OUTPUT = "outputs/i2_3d_texture.glb"

NUM_INFERENCE_STEPS = 50
SEED = 0

BASE_SCALE = 2.0
MIN_SCALE = 2.0
MAX_SCALE = 3.0
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--paint-subfolder", default=PAINT_SUBFOLDER)
    parser.add_argument("--image", default=INPUT_IMAGE)
    parser.add_argument("--mesh", type=Path, default=Path(INPUT_MESH))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
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


def retrieve_timesteps(scheduler, num_inference_steps, device, timesteps=None, sigmas=None):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of timesteps or sigmas can be passed.")
    parameters = inspect.signature(scheduler.set_timesteps).parameters
    if timesteps is not None:
        if "timesteps" not in parameters:
            raise ValueError(f"{scheduler.__class__.__name__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device)
        num_inference_steps = len(scheduler.timesteps)
    elif sigmas is not None:
        if "sigmas" not in parameters:
            raise ValueError(f"{scheduler.__class__.__name__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device)
        num_inference_steps = len(scheduler.timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device)
    return scheduler.timesteps, num_inference_steps


def flat_views_to_grouped(tensor: torch.Tensor, num_views: int) -> torch.Tensor:
    if tensor.shape[0] % num_views != 0:
        raise ValueError(f"Cannot group {tensor.shape[0]} samples into {num_views} views.")
    return tensor.reshape(tensor.shape[0] // num_views, num_views, *tensor.shape[1:])


def grouped_views_to_flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:])


class TrajectoryFeedback:
    """Token-wise adjacent/stride-2 trajectory feedback for texture latents."""

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
            return torch.full(
                (prediction.shape[0], 1, prediction.shape[2], prediction.shape[3]),
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

        scale = token_scale.reshape(
            prediction.shape[0], prediction.shape[2], prediction.shape[3]
        ).unsqueeze(1)
        return scale.to(device=prediction.device, dtype=prediction.dtype), source

    def update(self, velocity: torch.Tensor, timestep: torch.Tensor, scale: torch.Tensor, source: str) -> None:
        self.velocities.append(self.flatten_tokens(velocity))
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


@torch.no_grad()
def trajcfg_paint_denoise(
    self,
    prompt=None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    timesteps=None,
    sigmas=None,
    guidance_scale: float = BASE_SCALE,
    negative_prompt=None,
    num_images_per_prompt: int = 1,
    eta: float = 0.0,
    generator=None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image=None,
    ip_adapter_image_embeds=None,
    output_type: str = "pil",
    return_dict: bool = True,
    cross_attention_kwargs=None,
    guidance_rescale: float = 0.0,
    clip_skip: Optional[int] = None,
    callback_on_step_end=None,
    callback_on_step_end_tensor_inputs=("latents",),
    **kwargs,
):
    if self.is_turbo:
        raise ValueError("TrajCFG requires the non-turbo Hunyuan3D paint pipeline.")

    options = self._trajcfg_options
    num_inference_steps = options["num_inference_steps"]
    feedback = TrajectoryFeedback(
        base_scale=options["base_scale"],
        min_scale=options["min_scale"],
        max_scale=options["max_scale"],
        rho_lo=options["rho_lo"],
    )

    callback = kwargs.pop("callback", None)
    callback_steps = kwargs.pop("callback_steps", None) or 1
    if hasattr(callback_on_step_end, "tensor_inputs"):
        callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    self.check_inputs(
        prompt,
        height,
        width,
        callback_steps,
        negative_prompt,
        prompt_embeds,
        negative_prompt_embeds,
        ip_adapter_image,
        ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs,
    )

    self._guidance_scale = max(options["base_scale"], 1.0001)
    self._guidance_rescale = guidance_rescale
    self._clip_skip = clip_skip
    self._cross_attention_kwargs = cross_attention_kwargs
    self._interrupt = False
    if not self.do_classifier_free_guidance:
        raise ValueError("TrajCFG requires classifier-free guidance.")

    if isinstance(prompt, str):
        batch_size = 1
    elif isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    lora_scale = self.cross_attention_kwargs.get("scale") if self.cross_attention_kwargs else None
    prompt_embeds, negative_prompt_embeds = self.encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        False,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=lora_scale,
        clip_skip=self.clip_skip,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
            False,
        )

    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        timesteps,
        sigmas,
    )
    num_channels_latents = self.unet.config.in_channels
    num_views = kwargs["num_in_batch"]
    latents = self.prepare_latents(
        batch_size * num_views,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
    added_cond_kwargs = {"image_embeds": image_embeds} if image_embeds is not None else None

    timestep_cond = None
    if self.unet.config.time_cond_proj_dim is not None:
        scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
        timestep_cond = self.get_guidance_scale_embedding(
            scale_tensor,
            embedding_dim=self.unet.config.time_cond_proj_dim,
        ).to(device=device, dtype=latents.dtype)

    num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
    self._num_timesteps = len(timesteps)
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for step, timestep in enumerate(timesteps):
            if self.interrupt:
                continue

            grouped_latents = flat_views_to_grouped(latents, num_views)
            latent_input = torch.cat([grouped_latents] * 2)
            latent_input = grouped_views_to_flat(latent_input)
            latent_input = self.scheduler.scale_model_input(latent_input, timestep)
            latent_input = flat_views_to_grouped(latent_input, num_views)
            prediction = self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=self.cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                **kwargs,
            )[0]
            prediction_uncond, prediction_cond = prediction.chunk(2)
            scale, source = feedback.scale_for_step(step, prediction_uncond)
            guided_velocity = prediction_uncond + scale * (prediction_cond - prediction_uncond)

            if guidance_rescale > 0.0:
                dims = list(range(1, guided_velocity.ndim))
                std_cond = prediction_cond.std(dim=dims, keepdim=True)
                std_guided = guided_velocity.std(dim=dims, keepdim=True).clamp_min(NORM_FLOOR)
                rescaled = guided_velocity * (std_cond / std_guided)
                guided_velocity = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided_velocity

            feedback.update(guided_velocity, timestep, scale, source)
            latents = grouped_views_to_flat(grouped_latents)
            latents = self.scheduler.step(
                guided_velocity,
                timestep,
                latents[:, :num_channels_latents],
                **extra_step_kwargs,
                return_dict=False,
            )[0]

            if callback_on_step_end is not None:
                callback_kwargs = {name: locals()[name] for name in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, step, timestep, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            if step == len(timesteps) - 1 or (
                (step + 1) > num_warmup_steps and (step + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
                if callback is not None and step % callback_steps == 0:
                    callback(step // self.scheduler.order, timestep, latents)

    if output_type != "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor,
            return_dict=False,
            generator=generator,
        )[0]
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
    else:
        image = latents
        has_nsfw_concept = None

    do_denormalize = (
        [True] * image.shape[0]
        if has_nsfw_concept is None
        else [not has_nsfw for has_nsfw in has_nsfw_concept]
    )
    image = self.image_processor.postprocess(
        image,
        output_type=output_type,
        do_denormalize=do_denormalize,
    )
    self._trajcfg_stats = feedback.stats
    self.maybe_free_model_hooks()

    if not return_dict:
        return image, has_nsfw_concept
    return types.SimpleNamespace(images=image, nsfw_content_detected=has_nsfw_concept)


def install_trajcfg(pipeline: Hunyuan3DPaintPipeline, args: argparse.Namespace) -> None:
    paint_pipe = pipeline.models["multiview_model"].pipeline
    if paint_pipe.is_turbo:
        raise ValueError("Use the non-turbo hunyuan3d-paint-v2-0 subfolder for TrajCFG.")
    paint_pipe.denoise = types.MethodType(trajcfg_paint_denoise, paint_pipe)
    paint_pipe._trajcfg_options = {
        "num_inference_steps": args.num_inference_steps,
        "base_scale": args.base_scale,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "rho_lo": args.rho_lo,
    }


def generate_multiviews(multiview_model, images, controls, camera_info, *, steps: int, seed: int):
    multiview_model.seed_everything(seed)
    images = [image.resize((multiview_model.view_size, multiview_model.view_size)) for image in images]
    controls = list(controls)
    for index, control in enumerate(controls):
        control = control.resize((multiview_model.view_size, multiview_model.view_size))
        if control.mode == "L":
            control = control.point(lambda value: 255 if value > 1 else 0, mode="1")
        controls[index] = control

    paint_pipe = multiview_model.pipeline
    generator = torch.Generator(device=paint_pipe.device).manual_seed(seed)
    num_views = len(controls) // 2
    result = paint_pipe(
        images,
        generator=generator,
        width=multiview_model.view_size,
        height=multiview_model.view_size,
        num_in_batch=num_views,
        camera_info_gen=[camera_info],
        camera_info_ref=[[0]],
        normal_imgs=[[controls[index] for index in range(num_views)]],
        position_imgs=[[controls[index + num_views] for index in range(num_views)]],
        num_inference_steps=steps,
        guidance_scale=max(paint_pipe._trajcfg_options["base_scale"], 1.0001),
    )
    return result.images, paint_pipe._trajcfg_stats


@torch.no_grad()
def generate_texture(
    pipeline: Hunyuan3DPaintPipeline,
    mesh: trimesh.Trimesh,
    image: Image.Image,
    *,
    steps: int,
    seed: int,
):
    from hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap

    image = pipeline.models["delight_model"](pipeline.recenter_image(image))
    mesh = mesh_uv_wrap(mesh)
    pipeline.render.load_mesh(mesh)

    elevations = pipeline.config.candidate_camera_elevs
    azimuths = pipeline.config.candidate_camera_azims
    weights = pipeline.config.candidate_view_weights
    normal_maps = pipeline.render_normal_multiview(elevations, azimuths, use_abs_coor=True)
    position_maps = pipeline.render_position_multiview(elevations, azimuths)
    camera_info = [
        (((azimuth // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[elevation]
        + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elevation]
        for azimuth, elevation in zip(azimuths, elevations)
    ]

    multiviews, stats = generate_multiviews(
        pipeline.models["multiview_model"],
        [image],
        normal_maps + position_maps,
        camera_info,
        steps=steps,
        seed=seed,
    )
    multiviews = [
        view.resize((pipeline.config.render_size, pipeline.config.render_size))
        for view in multiviews
    ]
    texture, mask = pipeline.bake_from_multiview(
        multiviews,
        elevations,
        azimuths,
        weights,
        method=pipeline.config.merge_method,
    )
    mask = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)
    texture = pipeline.texture_inpaint(texture, mask)
    pipeline.render.set_texture(texture)
    return pipeline.render.save_mesh(), stats


def main() -> None:
    args = parse_args()
    if args.paint_subfolder != PAINT_SUBFOLDER:
        raise ValueError(f"TrajCFG requires --paint-subfolder {PAINT_SUBFOLDER}.")
    if args.num_inference_steps < 3:
        raise ValueError("--num-inference-steps must be at least 3.")
    if min(args.base_scale, args.min_scale, args.max_scale) < 1.0:
        raise ValueError("CFG scales must be at least 1.0.")
    if args.min_scale > args.max_scale:
        raise ValueError("--min-scale must not exceed --max-scale.")

    torch.cuda.set_device(args.gpu)
    pipeline = Hunyuan3DPaintPipeline.from_pretrained(
        args.model_path,
        subfolder=args.paint_subfolder,
    )
    install_trajcfg(pipeline, args)
    image = load_image(args.image, remove_white_background=not args.keep_background)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    textured_mesh, stats = generate_texture(
        pipeline,
        mesh,
        image,
        steps=args.num_inference_steps,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    textured_mesh.export(args.output)
    metadata = {
        "task": "image-to-3d texture generation",
        "model_path": args.model_path,
        "paint_subfolder": args.paint_subfolder,
        "image": args.image,
        "mesh": str(args.mesh),
        "output": str(args.output),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "base_scale": args.base_scale,
        "min_scale": args.min_scale,
        "max_scale": args.max_scale,
        "rho_lo": args.rho_lo,
        "fusion_weights": [SHORT_WEIGHT, STRIDE2_WEIGHT],
        "scale_stats": stats,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
