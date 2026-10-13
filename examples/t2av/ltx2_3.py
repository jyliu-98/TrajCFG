"""Self-contained TrajCFG inference for LTX-2.3 text-to-audio-video generation."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import torch

import ltx_core.model.transformer.attention as ltx_attention
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.types import Audio, VideoPixelShape
from ltx_core.utils import to_velocity
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, LTX_2_3_PARAMS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser, _guided_denoise
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.quantization_factory import QuantizationKind
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


PROMPT = (
    "In a medium shot with a somber, cinematic style, a young man in a rain-dampened jacket stands "
    "anxiously in a dimly lit apartment hallway. He raises a hesitant hand and knocks three times on "
    "a dark wooden door. The sound design features the sharp, echoing sound of the knocks, followed by "
    "the click of a lock turning and the slow creak of the door opening. His friend appears in the "
    "doorway, a look of concern on his face."
)
OUTPUT = "outputs/t2av_ltx2_3.mp4"

NUM_FRAMES = 97
NUM_INFERENCE_STEPS = 30
FRAME_RATE = 24.0

BASE_SCALE = 5.0
MIN_SCALE = 4.0
MAX_SCALE = 9.0
RHO_LO = 0.90
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/path/to/LTX-2.3"),
        help="Directory containing LTX-2.3 weights.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--spatial-upsampler-path", type=Path, default=None)
    parser.add_argument("--distilled-lora-path", type=Path, default=None)
    parser.add_argument("--gemma-root", type=Path, default=Path("/path/to/gemma-3-12b"))
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--height", type=int, default=LTX_2_3_PARAMS.stage_2_height)
    parser.add_argument("--width", type=int, default=LTX_2_3_PARAMS.stage_2_width)
    parser.add_argument("--num-frames", type=int, default=NUM_FRAMES)
    parser.add_argument("--frame-rate", type=float, default=FRAME_RATE)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--seed", type=int, default=LTX_2_3_PARAMS.seed)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--quantization", choices=("none", "fp8-cast", "fp8-scaled-mm"), default="none")
    parser.add_argument("--offload", choices=("none", "cpu", "disk"), default="none")
    parser.add_argument("--allocator-trim-strategy", choices=("trim", "defer"), default="defer")
    parser.add_argument("--enable-cudnn-sdpa", action="store_true")
    return parser.parse_args()


class TrajectoryFeedbackGuider:
    """LTX guider with token-wise TrajCFG and the official auxiliary guidance terms."""

    def __init__(
        self,
        *,
        modality: str,
        params: MultiModalGuiderParams,
        negative_context: torch.Tensor,
        base_scale: float,
        min_scale: float,
        max_scale: float,
        rho_lo: float,
    ) -> None:
        if not 0.0 <= min_scale <= max_scale:
            raise ValueError("Expected 0 <= min_scale <= max_scale.")
        if not -1.0 <= rho_lo < 1.0:
            raise ValueError("rho_lo must be in [-1, 1).")
        self.modality = modality
        self.params = params
        self.negative_context = negative_context
        self.base_scale = base_scale
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.rho_lo = rho_lo
        self.step_index = 0
        self.current_latent: torch.Tensor | None = None
        self.current_sigma: torch.Tensor | None = None
        self.velocities: deque[torch.Tensor] = deque(maxlen=3)
        self.timesteps: deque[float] = deque(maxlen=3)
        self.stats: list[dict[str, Any]] = []

    def set_step(self, *, step_index: int, latent: torch.Tensor | None, sigma: torch.Tensor) -> None:
        self.step_index = step_index
        self.current_latent = latent
        self.current_sigma = sigma

    def do_unconditional_generation(self) -> bool:
        return True

    def do_perturbed_generation(self) -> bool:
        return not math.isclose(self.params.stg_scale, 0.0)

    def do_isolated_modality_generation(self) -> bool:
        return not math.isclose(self.params.modality_scale, 1.0)

    def should_skip_step(self, step: int) -> bool:
        if self.params.skip_step == 0:
            return False
        return step % (self.params.skip_step + 1) != 0

    @staticmethod
    def flatten_tokens(velocity: torch.Tensor) -> torch.Tensor:
        if velocity.ndim < 3:
            raise ValueError(f"Expected a batched latent tensor, got {tuple(velocity.shape)}.")
        order = [0, *range(2, velocity.ndim), 1]
        return velocity.detach().float().cpu().permute(*order).reshape(
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

    @staticmethod
    def latent_shape(token_scale: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        return token_scale.reshape(latent.shape[0], *latent.shape[2:]).unsqueeze(1)

    def scale_for_step(self, conditional: torch.Tensor) -> tuple[torch.Tensor, str]:
        if self.step_index < 2:
            shape = (conditional.shape[0], 1, *conditional.shape[2:])
            return torch.full(
                shape, self.base_scale, device=conditional.device, dtype=conditional.dtype
            ), "base"

        adjacent_cosine = self.cosine(self.velocities[-2], self.velocities[-1])
        adjacent_scale = self.map_to_scale(adjacent_cosine)
        token_scale = adjacent_scale
        source = "adjacent"

        if len(self.velocities) == 3:
            stride2_cosine = self.cosine(self.velocities[-3], self.velocities[-1])
            delta_adjacent = abs(self.timesteps[-1] - self.timesteps[-2])
            delta_stride2 = abs(self.timesteps[-1] - self.timesteps[-3])
            if delta_adjacent > 0.0 and delta_stride2 > 0.0:
                stride2_cosine = torch.cos(
                    torch.acos(stride2_cosine) * (delta_adjacent / delta_stride2)
                ).clamp(-1.0, 1.0)
            stride2_scale = self.map_to_scale(stride2_cosine)
            token_scale = SHORT_WEIGHT * adjacent_scale + STRIDE2_WEIGHT * stride2_scale
            source = "adjacent+stride2"

        scale = self.latent_shape(token_scale, conditional)
        return scale.to(device=conditional.device, dtype=conditional.dtype), source

    def calculate(
        self,
        cond: torch.Tensor,
        uncond_text: torch.Tensor | float,
        uncond_perturbed: torch.Tensor | float,
        uncond_modality: torch.Tensor | float,
    ) -> torch.Tensor:
        dtype = cond.dtype
        cond_f = cond.float()
        text_f = uncond_text.float() if isinstance(uncond_text, torch.Tensor) else uncond_text
        perturbed_f = uncond_perturbed.float() if isinstance(uncond_perturbed, torch.Tensor) else uncond_perturbed
        modality_f = uncond_modality.float() if isinstance(uncond_modality, torch.Tensor) else uncond_modality
        scale, source = self.scale_for_step(cond_f)

        guided = (
            cond_f
            + (scale - 1.0) * (cond_f - text_f)
            + self.params.stg_scale * (cond_f - perturbed_f)
            + (self.params.modality_scale - 1.0) * (cond_f - modality_f)
        )
        if self.params.rescale_scale != 0:
            factor = cond_f.std() / guided.std().clamp_min(NORM_FLOOR)
            factor = self.params.rescale_scale * factor + (1.0 - self.params.rescale_scale)
            guided = guided * factor

        if self.current_latent is None or self.current_sigma is None:
            raise RuntimeError("The denoiser must set the current latent and sigma before guidance.")
        velocity = to_velocity(self.current_latent, self.current_sigma, guided)
        self.velocities.append(self.flatten_tokens(velocity))
        timestep = float(self.current_sigma.detach().float().cpu().flatten()[0].item())
        self.timesteps.append(timestep)
        scale_value = scale.detach().float()
        self.stats.append(
            {
                "step": self.step_index,
                "timestep": timestep,
                "source": source,
                "min": float(scale_value.min().item()),
                "max": float(scale_value.max().item()),
                "mean": float(scale_value.mean().item()),
            }
        )
        return guided.to(dtype)


class TrajCFGDenoiser:
    def __init__(
        self,
        *,
        video_context: torch.Tensor,
        audio_context: torch.Tensor,
        video_guider: TrajectoryFeedbackGuider,
        audio_guider: TrajectoryFeedbackGuider,
    ) -> None:
        self.video_context = video_context
        self.audio_context = audio_context
        self.video_guider = video_guider
        self.audio_guider = audio_guider
        self.last_video: torch.Tensor | None = None
        self.last_audio: torch.Tensor | None = None

    def __call__(self, transformer, video_state, audio_state, sigmas: torch.Tensor, step_index: int):
        sigma = sigmas[step_index]
        self.video_guider.set_step(
            step_index=step_index,
            latent=video_state.latent if video_state is not None else None,
            sigma=sigma,
        )
        self.audio_guider.set_step(
            step_index=step_index,
            latent=audio_state.latent if audio_state is not None else None,
            sigma=sigma,
        )
        video_result, audio_result = _guided_denoise(
            transformer=transformer,
            video_state=video_state,
            audio_state=audio_state,
            sigma=sigma,
            video_guider=self.video_guider,
            audio_guider=self.audio_guider,
            v_context=self.video_context,
            a_context=self.audio_context,
            last_denoised_video=self.last_video,
            last_denoised_audio=self.last_audio,
            step_index=step_index,
            force_uncond_pass=True,
        )
        self.last_video = video_result.denoised if video_result is not None else None
        self.last_audio = audio_result.denoised if audio_result is not None else None
        return video_result, audio_result


class TrajCFGLTXPipeline(TI2VidTwoStagesPipeline):
    """Apply TrajCFG in stage 1 and retain the official stage-2 refinement."""

    def __call__(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        tiling_config: TilingConfig,
        max_batch_size: int,
        base_scale: float,
        min_scale: float,
        max_scale: float,
        rho_lo: float,
    ) -> tuple[Any, Audio, dict[str, Any]]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16
        images: list[Any] = []

        positive, negative = self.prompt_encoder(
            [prompt, negative_prompt],
            enhance_first_prompt=False,
            enhance_prompt_image=None,
            enhance_prompt_seed=seed,
        )
        video_positive, audio_positive = positive.video_encoding, positive.audio_encoding
        video_negative, audio_negative = negative.video_encoding, negative.audio_encoding
        stage1_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage1_conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=stage1_shape.height,
                width=stage1_shape.width,
                video_encoder=encoder,
                dtype=dtype,
                device=self.device,
            )
        )
        sigmas = self._scheduler.execute(steps=num_inference_steps).to(
            dtype=torch.float32, device=self.device
        )
        video_guider = TrajectoryFeedbackGuider(
            modality="video",
            params=video_guider_params,
            negative_context=video_negative,
            base_scale=base_scale,
            min_scale=min_scale,
            max_scale=max_scale,
            rho_lo=rho_lo,
        )
        audio_guider = TrajectoryFeedbackGuider(
            modality="audio",
            params=audio_guider_params,
            negative_context=audio_negative,
            base_scale=base_scale,
            min_scale=min_scale,
            max_scale=max_scale,
            rho_lo=rho_lo,
        )
        denoiser = TrajCFGDenoiser(
            video_context=video_positive,
            audio_context=audio_positive,
            video_guider=video_guider,
            audio_guider=audio_guider,
        )
        video_state, audio_state = self.stage_1(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=stage1_shape.width,
            height=stage1_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_positive, conditionings=stage1_conditionings),
            audio=ModalitySpec(context=audio_positive),
            max_batch_size=max_batch_size,
        )

        upscaled_video = self.upsampler(video_state.latent[:1])
        stage2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        stage2_conditionings = self.image_conditioner(
            lambda encoder: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=encoder,
                dtype=dtype,
                device=self.device,
            )
        )
        video_state, _ = self.stage_2(
            denoiser=SimpleDenoiser(v_context=video_positive, a_context=audio_positive),
            sigmas=stage2_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_positive,
                conditionings=stage2_conditionings,
                noise_scale=stage2_sigmas[0].item(),
                initial_latent=upscaled_video,
            ),
            audio=ModalitySpec(
                context=audio_positive,
                noise_scale=stage2_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
        )
        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio, {
            "video": video_guider.stats,
            "audio": audio_guider.stats,
        }


def disable_cudnn_sdpa() -> None:
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    ltx_attention._SDPA_FULL_PRIORITY = tuple(
        backend
        for backend in ltx_attention._SDPA_FULL_PRIORITY
        if backend is not ltx_attention.SDPBackend.CUDNN_ATTENTION
    )
    if not getattr(ltx_attention.PytorchAttention, "_trajcfg_no_cudnn_patch", False):
        original_init = ltx_attention.PytorchAttention.__init__
        original_call = ltx_attention.PytorchAttention.__call__

        def filter_priority(priority):
            filtered = [
                backend
                for backend in priority
                if backend is not ltx_attention.SDPBackend.CUDNN_ATTENTION
            ]
            return filtered or [ltx_attention.SDPBackend.MATH]

        def patched_init(self, priority=None):
            original_init(self, priority)
            self._priority = filter_priority(self._priority)

        def patched_call(self, *args, **kwargs):
            self._priority = filter_priority(self._priority)
            return original_call(self, *args, **kwargs)

        ltx_attention.PytorchAttention.__init__ = patched_init
        ltx_attention.PytorchAttention.__call__ = patched_call
        ltx_attention.PytorchAttention._trajcfg_no_cudnn_patch = True
    ltx_attention.automatic_attention.cache_clear()
    ltx_attention.automatic_masked_attention.cache_clear()


def visible(path: Path, root: Path) -> bool:
    return all(
        part and not part.startswith(".") and "temp" not in part.lower()
        for part in path.relative_to(root).parts
    )


def find_first(root: Path, patterns: list[str]) -> Path | None:
    if not root.exists():
        return None
    for pattern in patterns:
        matches = sorted(path for path in root.rglob(pattern) if path.is_file() and visible(path, root))
        if matches:
            return matches[0]
    return None


def resolve_model_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = args.model_root.expanduser()
    paths = {
        "checkpoint": args.checkpoint_path
        or find_first(root, ["ltx-2.3-22b-dev.safetensors", "*22b-dev*.safetensors"]),
        "spatial_upsampler": args.spatial_upsampler_path
        or find_first(root, ["ltx-2.3-spatial-upscaler-x2-1.1.safetensors", "*spatial-upscaler*x2*.safetensors"]),
        "distilled_lora": args.distilled_lora_path
        or find_first(root, ["ltx-2.3-22b-distilled-lora-384-1.1.safetensors", "*distilled-lora*.safetensors"]),
        "gemma": args.gemma_root,
    }
    missing = [name for name, path in paths.items() if path is None or not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model assets: {', '.join(missing)}. Pass their paths explicitly."
        )
    return {name: Path(path).expanduser().resolve() for name, path in paths.items()}


def build_pipeline(args: argparse.Namespace, paths: dict[str, Path], device: torch.device) -> TrajCFGLTXPipeline:
    quantization = None
    if args.quantization != "none":
        quantization = QuantizationKind(args.quantization).to_policy(
            checkpoint_path=str(paths["checkpoint"])
        )
    distilled_lora = [
        LoraPathStrengthAndSDOps(
            str(paths["distilled_lora"]), 1.0, LTXV_LORA_COMFY_RENAMING_MAP
        )
    ]
    return TrajCFGLTXPipeline(
        checkpoint_path=str(paths["checkpoint"]),
        distilled_lora=distilled_lora,
        spatial_upsampler_path=str(paths["spatial_upsampler"]),
        gemma_root=str(paths["gemma"]),
        loras=[],
        device=device,
        quantization=quantization,
        offload_mode=OffloadMode(args.offload),
        alloc_trim_strategy=AllocatorTrimStrategy(args.allocator_trim_strategy),
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("LTX-2.3 inference requires a CUDA device.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    if not args.enable_cudnn_sdpa:
        disable_cudnn_sdpa()

    paths = resolve_model_paths(args)
    pipeline = build_pipeline(args, paths, device)
    tiling_config = TilingConfig.default()
    video_chunks = get_video_chunks_number(args.num_frames, tiling_config)
    video_params = LTX_2_3_PARAMS.video_guider_params
    audio_params = LTX_2_3_PARAMS.audio_guider_params

    with torch.inference_mode():
        video, audio, diagnostics = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=video_params,
            audio_guider_params=audio_params,
            tiling_config=tiling_config,
            max_batch_size=args.max_batch_size,
            base_scale=args.base_scale,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            rho_lo=args.rho_lo,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=str(args.output),
        video_chunks_number=video_chunks,
    )
    metadata = {
        "backbone": "LTX-2.3-22B-dev",
        "prompt": args.prompt,
        "model_root": str(args.model_root),
        "resolved_model_paths": {name: str(path) for name, path in paths.items()},
        "output": str(args.output),
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "frame_rate": args.frame_rate,
        "seed": args.seed,
        "trajcfg": {
            "base_scale": args.base_scale,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "rho_lo": args.rho_lo,
            "adjacent_weight": SHORT_WEIGHT,
            "stride2_weight": STRIDE2_WEIGHT,
            "stage": "stage_1",
        },
        "per_step": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
