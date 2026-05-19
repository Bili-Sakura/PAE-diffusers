# PAE Diffusers integration

This repository provides native [Diffusers](https://github.com/huggingface/diffusers) components for **PAE** (Prior-Aligned Autoencoder) tokenizers and **LightningDiT** generators. The layout mirrors [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers): implementations live under `src/diffusers` and can be copied into upstream Diffusers or installed locally.

## Package layout

| Component | Path | Class |
|-----------|------|-------|
| Transformer | `src/diffusers/models/transformers/transformer_lightning_dit.py` | `LightningDiT2DModel` |
| Tokenizer | `src/diffusers/models/autoencoders/autoencoder_pae.py` | `PAEAutoencoder` |
| Scheduler | `src/diffusers/schedulers/scheduling_flow_match_pae.py` | `PAEFlowMatchScheduler` |
| Pipeline | `src/diffusers/pipelines/pae/pipeline_pae.py` | `PAEPipeline` |

## Install

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest tests/
```

## Convert checkpoints

Convert a LightningDiT checkpoint (and optionally a PAE checkpoint) into a Diffusers pipeline directory:

```bash
PYTHONPATH=src python3 scripts/convert_pae_to_diffusers.py \
  --dit-checkpoint /path/to/0100000.pt \
  --pae-checkpoint /path/to/pae/ep-last.pt \
  --output ./pae-dit-xl-diffusers \
  --model-type LightningDiT-XL/1 \
  --image-size 256 \
  --in-channels 32
```

Output layout:

```text
model_index.json
scheduler/scheduler_config.json
transformer/config.json
transformer/diffusion_pytorch_model.safetensors
vae/config.json
vae/diffusion_pytorch_model.safetensors
```

## Sample images

```bash
PYTHONPATH=src python3 scripts/sample_pae.py \
  --model ./pae-dit-xl-diffusers \
  --class-label 207 \
  --height 256 \
  --width 256 \
  --num-inference-steps 250 \
  --mode ode \
  --guidance-scale 2.5 \
  --guidance-interval 0.0 1.0
```

## Python API

```python
import torch
from diffusers import PAEPipeline, LightningDiT2DModel, PAEFlowMatchScheduler, PAEAutoencoder

pipe = PAEPipeline.from_pretrained("path/to/converted-pipeline", torch_dtype=torch.bfloat16)
pipe.to("cuda")
images = pipe(class_labels=[207], num_inference_steps=250, guidance_scale=2.5).images
```

## Upstreaming

To contribute to Hugging Face Diffusers, copy modules from `src/diffusers` into the matching package paths and register classes in Diffusers lazy-import tables.
