"""Self-contained TrajCFG inference for Ovi 1.1 text-to-audio-video generation."""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Any

import ovi.ovi_fusion_engine as ovi_fusion_engine_module
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from ovi.distributed_comms.parallel_states import initialize_sequence_parallel_state
from ovi.ovi_fusion_engine import OviFusionEngine
from ovi.utils.io_utils import save_video
from ovi.utils.processing_utils import snap_hw_to_multiple_of_32


PROMPT = (
    "A dramatic close-up shot focuses on a young man, his face etched with intense concentration, "
    "seated in a dimly lit room where his features are illuminated only by the dynamic, flickering "
    "glow of a large monitor off-screen. The air is thick with the thunderous sounds of intense video "
    "game combat—explosions, rapid gunfire, and urgent in-game callouts. As the camera performs a slow, "
    "dramatic dolly-in, the chaotic combat audio cuts out abruptly, replaced by the stark, repetitive "
    "beep of a low-battery game controller. The man's focused expression shatters into wide-eyed panic "
    "as he shouts in a frantic, high-pitched voice, \"No, no, no! My controller died! REVIVE ME!\""
)
OUTPUT = "outputs/t2av_ovi.mp4"
OVI_CONFIG_FILE = (
    Path(ovi_fusion_engine_module.__file__).resolve().parent
    / "configs/inference/inference_fusion.yaml"
)

SAMPLE_STEPS = 50
FPS = 24.0
SEED = 103

BASE_SCALE = 3.5
MIN_SCALE = 3.5
MAX_SCALE = 10.0
RHO_LO = 0.95
SHORT_WEIGHT = 0.8
STRIDE2_WEIGHT = 0.2
NORM_FLOOR = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/path/to/Ovi-model"),
        help="Ovi checkpoint directory.",
    )
    parser.add_argument(
        "--wan-path",
        type=Path,
        default=Path("/path/to/Wan2.2-TI2V-5B"),
        help="Wan2.2-TI2V-5B directory used by Ovi.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("/path/to/Ovi/ckpts"),
        help="Ovi auxiliary checkpoint root.",
    )
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--output", type=Path, default=Path(OUTPUT))
    parser.add_argument("--sample-steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--solver-name", default="unipc")
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--slg-layer", type=int, default=11)
    parser.add_argument("--video-negative-prompt", default="jitter, bad hands, blur, distortion")
    parser.add_argument("--audio-negative-prompt", default="robotic, muffled, echo, distorted")
    parser.add_argument("--base-scale", type=float, default=BASE_SCALE)
    parser.add_argument("--min-scale", type=float, default=MIN_SCALE)
    parser.add_argument("--max-scale", type=float, default=MAX_SCALE)
    parser.add_argument("--rho-lo", type=float, default=RHO_LO)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--qint8", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    return parser.parse_args()


class TrajectoryFeedback:
    """Token-wise adjacent/stride-2 trajectory feedback for one modality."""

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
        if velocity.ndim == 4:  # Ovi video: [C, F, H, W]
            return velocity.detach().float().cpu().permute(1, 2, 3, 0).reshape(-1, velocity.shape[0])
        if velocity.ndim == 2:  # Ovi audio: [L, C]
            return velocity.detach().float().cpu()
        raise ValueError(f"Unsupported Ovi prediction shape: {tuple(velocity.shape)}")

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        numerator = (a * b).sum(dim=-1)
        denominator = a.norm(dim=-1).clamp_min(NORM_FLOOR) * b.norm(dim=-1).clamp_min(NORM_FLOOR)
        return (numerator / denominator).clamp(-1.0, 1.0)

    def map_to_scale(self, cosine: torch.Tensor) -> torch.Tensor:
        fraction = ((1.0 - cosine) / (1.0 - self.rho_lo)).clamp(0.0, 1.0)
        return self.min_scale + (self.max_scale - self.min_scale) * fraction

    @staticmethod
    def prediction_shape(token_scale: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == 4:
            return token_scale.reshape(*prediction.shape[1:]).unsqueeze(0)
        return token_scale.reshape(prediction.shape[0], 1)

    @staticmethod
    def scalar_scale(value: float, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.ndim == 4:
            shape = (1, *prediction.shape[1:])
        else:
            shape = (prediction.shape[0], 1)
        return torch.full(shape, value, device=prediction.device, dtype=prediction.dtype)

    def scale_for_step(self, step: int, prediction: torch.Tensor) -> tuple[torch.Tensor, str]:
        if step < 2:
            return self.scalar_scale(self.base_scale, prediction), "base"

        q_adjacent = self.cosine(self.velocities[-2], self.velocities[-1])
        adjacent_scale = self.map_to_scale(q_adjacent)
        token_scale = adjacent_scale
        source = "adjacent"

        if len(self.velocities) == 3:
            q_stride2 = self.cosine(self.velocities[-3], self.velocities[-1])
            delta_adjacent = abs(self.timesteps[-1] - self.timesteps[-2])
            delta_stride2 = abs(self.timesteps[-1] - self.timesteps[-3])
            if delta_adjacent > 0.0 and delta_stride2 > 0.0:
                q_stride2 = torch.cos(
                    torch.acos(q_stride2) * (delta_adjacent / delta_stride2)
                ).clamp(-1.0, 1.0)
            stride2_scale = self.map_to_scale(q_stride2)
            token_scale = SHORT_WEIGHT * adjacent_scale + STRIDE2_WEIGHT * stride2_scale
            source = "adjacent+stride2"

        scale = self.prediction_shape(token_scale, prediction)
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


class TrajCFGOviFusionEngine(OviFusionEngine):
    """Ovi engine with independent token-wise TrajCFG maps for video and audio."""

    @torch.inference_mode()
    def generate(
        self,
        *,
        text_prompt: str,
        video_frame_height_width: list[int],
        seed: int,
        solver_name: str,
        sample_steps: int,
        shift: float,
        slg_layer: int,
        video_negative_prompt: str,
        audio_negative_prompt: str,
        base_scale: float,
        min_scale: float,
        max_scale: float,
        rho_lo: float,
    ) -> tuple[Any, Any, dict[str, Any]]:
        video_feedback = TrajectoryFeedback(
            base_scale=base_scale, min_scale=min_scale, max_scale=max_scale, rho_lo=rho_lo
        )
        audio_feedback = TrajectoryFeedback(
            base_scale=base_scale, min_scale=min_scale, max_scale=max_scale, rho_lo=rho_lo
        )
        scheduler_video, timesteps_video = self.get_scheduler_time_steps(
            sampling_steps=sample_steps, device=self.device, solver_name=solver_name, shift=shift
        )
        scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
            sampling_steps=sample_steps, device=self.device, solver_name=solver_name, shift=shift
        )

        text_prompt = self.text_formatter(text_prompt)
        video_h, video_w = video_frame_height_width
        video_h, video_w = snap_hw_to_multiple_of_32(video_h, video_w, area=self.target_area)
        video_latent_h, video_latent_w = video_h // 16, video_w // 16

        if self.cpu_offload:
            self.text_model.model = self.text_model.model.to(self.device)
        embeddings = self.text_model(
            [text_prompt, video_negative_prompt, audio_negative_prompt], self.text_model.device
        )
        embeddings = [value.to(self.target_dtype).to(self.device) for value in embeddings]
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        video_noise = torch.randn(
            (self.video_latent_channel, self.video_latent_length, video_latent_h, video_latent_w),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        audio_noise = torch.randn(
            (self.audio_latent_length, self.audio_latent_channel),
            device=self.device,
            dtype=self.target_dtype,
            generator=torch.Generator(device=self.device).manual_seed(seed),
        )
        max_seq_len_audio = audio_noise.shape[0]
        patch_h, patch_w = self.model.video_model.patch_size[1:]
        max_seq_len_video = (
            video_noise.shape[1] * video_noise.shape[2] * video_noise.shape[3] // (patch_h * patch_w)
        )

        if self.cpu_offload:
            self.offload_to_cpu(self.vae_model_video.model)
            self.offload_to_cpu(self.vae_model_audio)
            self.model = self.model.to(self.device)

        with torch.amp.autocast("cuda", enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
            for step, (t_video, t_audio) in tqdm(
                enumerate(zip(timesteps_video, timesteps_audio)), total=sample_steps
            ):
                timestep_input = torch.full((1,), t_video, device=self.device)
                pred_video_pos, pred_audio_pos = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    audio_context=[embeddings[0]],
                    vid_context=[embeddings[0]],
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False,
                )
                pred_video_neg, pred_audio_neg = self.model(
                    vid=[video_noise],
                    audio=[audio_noise],
                    t=timestep_input,
                    audio_context=[embeddings[2]],
                    vid_context=[embeddings[1]],
                    vid_seq_len=max_seq_len_video,
                    audio_seq_len=max_seq_len_audio,
                    first_frame_is_clean=False,
                    slg_layer=slg_layer,
                )

                video_scale, video_source = video_feedback.scale_for_step(step, pred_video_pos[0])
                audio_scale, audio_source = audio_feedback.scale_for_step(step, pred_audio_pos[0])
                guided_video = pred_video_neg[0] + video_scale * (pred_video_pos[0] - pred_video_neg[0])
                guided_audio = pred_audio_neg[0] + audio_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                video_feedback.update(guided_video, t_video, video_scale, video_source)
                audio_feedback.update(guided_audio, t_audio, audio_scale, audio_source)

                video_noise = scheduler_video.step(
                    guided_video.unsqueeze(0), t_video, video_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                audio_noise = scheduler_audio.step(
                    guided_audio.unsqueeze(0), t_audio, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)

            if self.cpu_offload:
                self.offload_to_cpu(self.model)
                self.vae_model_video.model = self.vae_model_video.model.to(self.device)
                self.vae_model_audio = self.vae_model_audio.to(self.device)

            generated_audio = self.vae_model_audio.wrapped_decode(audio_noise.unsqueeze(0).transpose(1, 2))
            generated_audio = generated_audio.squeeze().cpu().float().numpy()
            generated_video = self.vae_model_video.wrapped_decode(video_noise.unsqueeze(0))
            generated_video = generated_video.squeeze(0).cpu().float().numpy()

        return generated_video, generated_audio, {
            "video": video_feedback.stats,
            "audio": audio_feedback.stats,
        }


def build_config(args: argparse.Namespace):
    config = OmegaConf.load(OVI_CONFIG_FILE)
    config.ckpt_dir = str(args.ckpt_dir)
    config.model_path = str(args.model_path)
    config.ovi_path = str(args.model_path)
    config.wan_path = str(args.wan_path)
    config.model_name = "960x960_10s"
    config.mode = "t2v"
    config.sp_size = 1
    config.sample_steps = args.sample_steps
    config.solver_name = args.solver_name
    config.shift = args.shift
    config.video_guidance_scale = args.base_scale
    config.audio_guidance_scale = args.base_scale
    config.slg_layer = args.slg_layer
    config.video_negative_prompt = args.video_negative_prompt
    config.audio_negative_prompt = args.audio_negative_prompt
    config.video_frame_height_width = [args.height, args.width]
    config.each_example_n_times = 1
    config.fp8 = args.fp8
    config.qint8 = args.qint8
    config.cpu_offload = args.cpu_offload
    config.text_prompt = ""
    config.image_path = None
    return config


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Ovi inference requires a CUDA device.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.cuda.set_device(args.gpu)
    initialize_sequence_parallel_state(1)
    engine = TrajCFGOviFusionEngine(
        config=build_config(args), device=args.gpu, target_dtype=torch.bfloat16
    )
    video, audio, diagnostics = engine.generate(
        text_prompt=args.prompt,
        video_frame_height_width=[args.height, args.width],
        seed=args.seed,
        solver_name=args.solver_name,
        sample_steps=args.sample_steps,
        shift=args.shift,
        slg_layer=args.slg_layer,
        video_negative_prompt=args.video_negative_prompt,
        audio_negative_prompt=args.audio_negative_prompt,
        base_scale=args.base_scale,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        rho_lo=args.rho_lo,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_video(str(args.output), video, audio, fps=args.fps, sample_rate=args.sample_rate)
    metadata = {
        "backbone": "Ovi 1.1 960x960_10s",
        "prompt": args.prompt,
        "model_path": str(args.model_path),
        "wan_path": str(args.wan_path),
        "output": str(args.output),
        "num_frames": int(video.shape[1]),
        "sample_steps": args.sample_steps,
        "fps": args.fps,
        "seed": args.seed,
        "trajcfg": {
            "base_scale": args.base_scale,
            "min_scale": args.min_scale,
            "max_scale": args.max_scale,
            "rho_lo": args.rho_lo,
            "adjacent_weight": SHORT_WEIGHT,
            "stride2_weight": STRIDE2_WEIGHT,
        },
        "per_step": diagnostics,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
