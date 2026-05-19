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
