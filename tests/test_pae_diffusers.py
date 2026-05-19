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

import pytest

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
