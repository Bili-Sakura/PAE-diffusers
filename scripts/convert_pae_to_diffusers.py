# Copyright 2026 The HuggingFace Team and PAE authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""Convert original PAE + LightningDiT checkpoints to a Diffusers pipeline directory."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
except ImportError:  # pragma: no cover
    safe_load_file = None
    safe_save_file = None

from diffusers.models.transformers import LIGHTNING_DIT_PRESETS, LightningDiT2DModel


def _load_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    if checkpoint_path.endswith(".safetensors"):
        if safe_load_file is None:
            raise ImportError("Install safetensors to convert .safetensors checkpoints.")
        state_dict = safe_load_file(checkpoint_path, device="cpu")
    else:
        blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(blob, dict):
            for key in ("ema", "model", "state_dict", "module"):
                if key in blob and isinstance(blob[key], dict):
                    blob = blob[key]
                    break
        state_dict = blob
    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        for prefix in ("model.", "module.", "transformer."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    return cleaned


def _save_config(output_dir: Path, config: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _save_weights(output_dir: Path, state_dict: Dict[str, torch.Tensor], safe_serialization: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if safe_serialization:
        if safe_save_file is None:
            raise ImportError("Install safetensors or pass --no-safe-serialization.")
        safe_save_file(state_dict, str(output_dir / "diffusion_pytorch_model.safetensors"), metadata={"format": "pt"})
    else:
        torch.save(state_dict, output_dir / "diffusion_pytorch_model.bin")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert PAE/LightningDiT checkpoints to Diffusers layout.")
    parser.add_argument("--dit-checkpoint", required=True, help="LightningDiT .pt/.safetensors checkpoint.")
    parser.add_argument("--output", required=True, help="Output Diffusers model directory.")
    parser.add_argument("--model-type", choices=sorted(LIGHTNING_DIT_PRESETS), default="LightningDiT-XL/1")
    parser.add_argument("--pae-checkpoint", default=None, help="Optional PAE checkpoint to embed under vae/.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--downsample-ratio", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=32)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    transformer_dir = output_dir / "transformer"
    scheduler_dir = output_dir / "scheduler"

    state_dict = _load_state_dict(args.dit_checkpoint)
    latent_size = args.image_size // args.downsample_ratio
    model = LIGHTNING_DIT_PRESETS[args.model_type](
        input_size=latent_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
    )
    config = dict(getattr(model, "config", {}))
    if not config:
        config = {
            "_class_name": "LightningDiT2DModel",
            "input_size": latent_size,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "hidden_size": model.hidden_size,
            "depth": model.depth,
            "num_heads": model.num_heads,
            "patch_size": model.patch_size,
        }
    else:
        config = dict(config)
        config["_class_name"] = "LightningDiT2DModel"

    _save_config(transformer_dir, config if isinstance(config, dict) else dict(config))
    _save_weights(transformer_dir, state_dict, args.safe_serialization)

    _save_config(
        scheduler_dir,
        {"_class_name": "PAEFlowMatchScheduler", "mode": args.mode, "path_type": "linear", "num_train_timesteps": 1000},
    )

    model_index = {
        "_class_name": "PAEPipeline",
        "_diffusers_version": "0.30.1",
        "scheduler": ["diffusers", "PAEFlowMatchScheduler"],
        "transformer": ["diffusers", "LightningDiT2DModel"],
        "vae": ["diffusers", "PAEAutoencoder"],
    }
    with open(output_dir / "model_index.json", "w", encoding="utf-8") as handle:
        json.dump(model_index, handle, indent=2, sort_keys=True)
        handle.write("\n")

    if args.pae_checkpoint:
        vae_dir = output_dir / "vae"
        vae_state = _load_state_dict(args.pae_checkpoint)
        _save_config(vae_dir, {"_class_name": "PAEAutoencoder"})
        _save_weights(vae_dir, vae_state, args.safe_serialization)

    print(f"Saved Diffusers-style PAE pipeline to {output_dir}")


if __name__ == "__main__":
    main()
