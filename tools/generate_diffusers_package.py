#!/usr/bin/env python3
"""Build src/diffusers native PAE integration."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "diffusers"
LEGACY = ROOT / "pae_with_generator"
NIT = ROOT / "src" / "diffusers"  # may contain nit stubs from initial copy

COPYRIGHT = '''# Copyright 2026 The HuggingFace Team and PAE authors. All rights reserved.
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
'''


def w(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(COPYRIGHT + "\n" + body.lstrip(), encoding="utf-8")


def strip_compile(text: str) -> str:
    return re.sub(r"^\s*@torch\.compile\n", "", text, flags=re.MULTILINE)


def copy_tokenizer_tree() -> None:
    """Port PAE tokenizer implementation into models/autoencoders."""
    mapping = {
        "tokenizer/encoders/__init__.py": "models/autoencoders/encoders/__init__.py",
        "tokenizer/encoders/dinov2.py": "models/autoencoders/encoders/dinov2.py",
        "tokenizer/encoders/dinov3.py": "models/autoencoders/encoders/dinov3.py",
        "tokenizer/encoders/siglip2.py": "models/autoencoders/encoders/siglip2.py",
        "tokenizer/encoders/mae.py": "models/autoencoders/encoders/mae.py",
        "tokenizer/encoders/internvit.py": "models/autoencoders/encoders/internvit.py",
        "tokenizer/encoders/delta.py": "models/autoencoders/encoders/delta.py",
        "tokenizer/latents/__init__.py": "models/autoencoders/latents/__init__.py",
        "tokenizer/latents/distributions.py": "models/autoencoders/latents/distributions.py",
        "tokenizer/decoders/__init__.py": "models/autoencoders/decoders/__init__.py",
        "tokenizer/decoders/utils.py": "models/autoencoders/decoders/utils.py",
        "tokenizer/decoders/decoder.py": "models/autoencoders/decoders/decoder.py",
    }
    for src_rel, dst_rel in mapping.items():
        text = strip_compile((LEGACY / src_rel).read_text(encoding="utf-8"))
        text = text.replace("from .latents import", "from ..latents import")
        w(SRC / dst_rel, text)


def write_layers() -> None:
    for name in ("rmsnorm.py", "pos_embed.py", "swiglu_ffn.py"):
        text = strip_compile((LEGACY / "models" / name).read_text(encoding="utf-8"))
        if name == "rmsnorm.py":
            idx = text.find("\n\n\nclass Attention")
            if idx > 0:
                text = text[:idx]
        w(SRC / "models" / "layers" / name, text)


def write_transformer() -> None:
    text = strip_compile((LEGACY / "models/lightningdit.py").read_text(encoding="utf-8"))
    text = text.replace("from models.swiglu_ffn import SwiGLUFFN", "from ..layers.swiglu_ffn import SwiGLUFFN")
    text = text.replace("from models.pos_embed import VisionRotaryEmbeddingFast", "from ..layers.pos_embed import VisionRotaryEmbeddingFast")
    text = text.replace("from models.rmsnorm import RMSNorm", "from ..layers.rmsnorm import RMSNorm")

    header = '''
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from diffusers.utils import BaseOutput
except Exception:  # pragma: no cover
    class BaseOutput(dict):
        def __post_init__(self):
            self.update(self.__dict__)

    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(torch.nn.Module):
        pass

    def register_to_config(init):
        return init


@dataclass
class LightningDiT2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


'''
    text = text.replace("class LightningDiT(nn.Module):", "class LightningDiT2DModel(ModelMixin, ConfigMixin):")
    text = text.replace(
        "    def __init__(\n        self,\n        input_size=32,",
        "    @register_to_config\n    def __init__(\n        self,\n        input_size: int = 32,",
    )
    text = text.replace(
        "    def forward(self, x, t=None, y=None):",
        "    def forward(\n        self,\n        hidden_states: torch.Tensor,\n        timestep: Optional[torch.LongTensor] = None,\n        class_labels: Optional[torch.LongTensor] = None,\n        return_dict: bool = True,\n    ):\n        x = hidden_states\n        t = timestep\n        y = class_labels",
    )
    text = text.replace(
        "        if self.learn_sigma:\n            x, _ = x.chunk(2, dim=1)\n        return x",
        "        if self.learn_sigma:\n            x, _ = x.chunk(2, dim=1)\n        if not return_dict:\n            return (x,)\n        return LightningDiT2DModelOutput(sample=x)",
    )
    text = text.replace("LightningDiT(", "LightningDiT2DModel(")
    text = text.replace("def LightningDiT_", "def _lightning_dit_")
    text = text.replace("LightningDiT_models = {", "LIGHTNING_DIT_PRESETS = {")
    w(SRC / "models/transformers/transformer_lightning_dit.py", header + text)


def write_autoencoder() -> None:
    text = strip_compile((LEGACY / "tokenizer/pae.py").read_text(encoding="utf-8"))
    text = text.replace("from .decoders import GeneralDecoder", "from .decoders.decoder import GeneralDecoder")
    text = text.replace("from .latents import RMSNorm", "from ..layers.rmsnorm import RMSNorm")
    text = text.replace("class PAE(nn.Module):", "class PAEAutoencoder(ModelMixin, ConfigMixin):")
    header = '''
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoImageProcessor

try:
    from diffusers.configuration_utils import ConfigMixin, register_to_config
    from diffusers.models.modeling_utils import ModelMixin
except Exception:  # pragma: no cover
    class ConfigMixin:
        config_name = "config.json"

    class ModelMixin(nn.Module):
        pass

    def register_to_config(init):
        return init

'''
    w(SRC / "models/autoencoders/autoencoder_pae.py", header + text)


def write_scheduler() -> None:
    nit_path = NIT / "schedulers/scheduling_flow_match_nit.py"
    if not nit_path.exists():
        nit_path = Path("/tmp/NiT-diffusers/src/diffusers/schedulers/scheduling_flow_match_nit.py")
    text = nit_path.read_text(encoding="utf-8")
    text = text.replace("NiTFlowMatchScheduler", "PAEFlowMatchScheduler")
    text = text.replace("NiTFlowMatchSchedulerOutput", "PAEFlowMatchSchedulerOutput")
    text = text.replace(
        "Flow-matching ODE/SDE scheduler used by Native-resolution Image Synthesis (NiT).",
        "Flow-matching ODE/SDE scheduler used by PAE + LightningDiT pipelines.",
    )
    # fixed-grid latents: image_sizes optional for SDE
    text = text.replace(
        '            if image_sizes is None:\n                raise ValueError("image_sizes are required for SDE sampling.")',
        "            if image_sizes is None:\n                batch = sample.shape[0]\n                side = int(sample.shape[-1])\n                image_sizes = torch.tensor([[side, side]] * batch, device=sample.device, dtype=torch.long)",
    )
    w(SRC / "schedulers/scheduling_flow_match_pae.py", text)


def write_pipeline() -> None:
    nit_path = NIT / "pipelines/nit/pipeline_nit.py"
    if not nit_path.exists():
        nit_path = Path("/tmp/NiT-diffusers/src/diffusers/pipelines/nit/pipeline_nit.py")
    text = nit_path.read_text(encoding="utf-8")
    text = text.replace("NiTPipeline", "PAEPipeline")
    text = text.replace("NiTPipelineOutput", "PAEPipelineOutput")
    text = text.replace("Native-resolution Image Synthesis", "Prior-Aligned Autoencoder (PAE)")
    text = text.replace("transformer", "transformer")
    text = text.replace("vae", "vae")
    text = text.replace(
        '    model_cpu_offload_seq = "transformer->vae"',
        '    model_cpu_offload_seq = "transformer->vae"\n    _optional_components = ["vae"]',
    )
    # Replace NiT-specific latent prep with fixed-grid PAE latents
    prep = '''
    def _prepare_latents(
        self,
        batch_size: int,
        num_channels: int,
        height: int,
        width: int,
        downsample_ratio: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        latent_height = height // downsample_ratio
        latent_width = width // downsample_ratio
        shape = (batch_size, num_channels, latent_height, latent_width)
        return torch.randn(shape, generator=generator, device=device, dtype=dtype)
'''
    text = re.sub(
        r"    def _prepare_latents\([\s\S]*?return latents\n",
        prep,
        text,
        count=1,
    )
    w(SRC / "pipelines/pae/pipeline_pae.py", text)


def write_inits() -> None:
    w(
        SRC / "models/layers/__init__.py",
        "from .rmsnorm import RMSNorm\nfrom .pos_embed import VisionRotaryEmbeddingFast\nfrom .swiglu_ffn import SwiGLUFFN\n",
    )
    w(
        SRC / "models/transformers/__init__.py",
        "from .transformer_lightning_dit import LIGHTNING_DIT_PRESETS, LightningDiT2DModel, LightningDiT2DModelOutput\n\n__all__ = [\n    'LIGHTNING_DIT_PRESETS',\n    'LightningDiT2DModel',\n    'LightningDiT2DModelOutput',\n]\n",
    )
    w(
        SRC / "models/autoencoders/__init__.py",
        "from .autoencoder_pae import PAEAutoencoder\n\n__all__ = ['PAEAutoencoder']\n",
    )
    w(
        SRC / "models/__init__.py",
        "from .autoencoders import PAEAutoencoder\nfrom .transformers import LightningDiT2DModel, LightningDiT2DModelOutput\n\n__all__ = ['PAEAutoencoder', 'LightningDiT2DModel', 'LightningDiT2DModelOutput']\n",
    )
    w(
        SRC / "schedulers/__init__.py",
        "from .scheduling_flow_match_pae import PAEFlowMatchScheduler, PAEFlowMatchSchedulerOutput\n\n__all__ = ['PAEFlowMatchScheduler', 'PAEFlowMatchSchedulerOutput']\n",
    )
    w(
        SRC / "pipelines/pae/__init__.py",
        "from .pipeline_pae import PAEPipeline, PAEPipelineOutput\n\n__all__ = ['PAEPipeline', 'PAEPipelineOutput']\n",
    )
    w(
        SRC / "pipelines/__init__.py",
        "from .pae import PAEPipeline, PAEPipelineOutput\n\n__all__ = ['PAEPipeline', 'PAEPipelineOutput']\n",
    )
    w(
        SRC / "__init__.py",
        """from .models import PAEAutoencoder, LightningDiT2DModel, LightningDiT2DModelOutput
from .pipelines import PAEPipeline, PAEPipelineOutput
from .schedulers import PAEFlowMatchScheduler, PAEFlowMatchSchedulerOutput

__all__ = [
    "PAEAutoencoder",
    "LightningDiT2DModel",
    "LightningDiT2DModelOutput",
    "PAEPipeline",
    "PAEPipelineOutput",
    "PAEFlowMatchScheduler",
    "PAEFlowMatchSchedulerOutput",
]
""",
    )


def write_convert_script() -> None:
    body = r'''#!/usr/bin/env python3
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
    preset = LIGHTNING_DIT_PRESETS[args.model_type]
    latent_size = args.image_size // args.downsample_ratio
    config = {
        "_class_name": "LightningDiT2DModel",
        "input_size": latent_size,
        "in_channels": args.in_channels,
        "num_classes": args.num_classes,
        **{k: v for k, v in preset.keywords.items() if k != "kwargs"},
    }
    # factory functions use kwargs only partially; rebuild from signature defaults
    model = preset(input_size=latent_size, in_channels=args.in_channels, num_classes=args.num_classes)
    config = model.config if hasattr(model, "config") else config

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
'''
    w(ROOT / "scripts/convert_pae_to_diffusers.py", body)


def write_sample_script() -> None:
    w(
        ROOT / "scripts/sample_pae.py",
        '''#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from diffusers import PAEPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Sample images with a converted PAE Diffusers pipeline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class-label", type=int, action="append", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-inference-steps", type=int, default=250)
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--guidance-scale", type=float, default=2.5)
    parser.add_argument("--guidance-interval", type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default="samples")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.torch_dtype]
    generator = torch.Generator(device=args.device if args.device != "cpu" else "cpu")
    if args.seed is not None:
        generator.manual_seed(args.seed)

    pipe = PAEPipeline.from_pretrained(args.model, torch_dtype=dtype).to(args.device)
    output = pipe(
        class_labels=args.class_label,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        mode=args.mode,
        guidance_scale=args.guidance_scale,
        guidance_interval=tuple(args.guidance_interval),
        generator=generator,
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(output.images):
        image.save(out_dir / f"{index:06d}.png")


if __name__ == "__main__":
    main()
''',
    )


def write_tests() -> None:
    w(
        ROOT / "tests/test_pae_diffusers.py",
        '''import pytest

torch = pytest.importorskip("torch")

from diffusers.models.transformers import LightningDiT2DModel
from diffusers.schedulers import PAEFlowMatchScheduler


def test_lightning_dit_forward():
    model = LightningDiT2DModel(
        input_size=16,
        patch_size=1,
        in_channels=32,
        hidden_size=64,
        depth=2,
        num_heads=4,
        num_classes=10,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
    )
    latents = torch.randn(2, 32, 16, 16)
    out = model(latents, timestep=torch.tensor([1.0, 0.5]), class_labels=torch.tensor([1, 2]))
    assert out.sample.shape == latents.shape


def test_scheduler_ode_step():
    scheduler = PAEFlowMatchScheduler(mode="ode")
    sample = torch.ones(1, 32, 16, 16)
    velocity = torch.full_like(sample, 2.0)
    output = scheduler.step(velocity, torch.tensor([1.0]), sample, torch.tensor([0.75]))
    assert torch.allclose(output.prev_sample, torch.full_like(sample, 0.5))
''',
    )


def write_pyproject() -> None:
    w(
        ROOT / "pyproject.toml",
        '''[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "diffusers-pae"
version = "0.1.0"
description = "Diffusers-style PAE tokenizer and LightningDiT generation components."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
dependencies = [
    "diffusers>=0.30.1",
    "torch",
    "torchvision",
    "transformers",
    "timm",
    "einops",
    "omegaconf",
    "safetensors",
    "accelerate",
    "Pillow",
]

[project.optional-dependencies]
dev = ["pytest"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
''',
    )


def remove_nit_stubs() -> None:
    for rel in [
        "models/transformers/transformer_nit.py",
        "pipelines/nit",
        "schedulers/scheduling_flow_match_nit.py",
    ]:
        path = SRC / rel
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def main() -> None:
    remove_nit_stubs()
    write_layers()
    copy_tokenizer_tree()
    write_transformer()
    write_autoencoder()
    write_scheduler()
    write_pipeline()
    write_inits()
    write_convert_script()
    write_sample_script()
    write_tests()
    write_pyproject()
    print("Package generated at", SRC)


if __name__ == "__main__":
    main()
