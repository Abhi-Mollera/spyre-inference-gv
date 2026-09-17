# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BLIP-2 / Q-Former workarounds for Spyre."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)


def patch_blip2_qformer_attention() -> None:
    """Patch Blip2QFormerMultiHeadAttention.forward to run on CPU.

    The full forward contains permute/matmul/softmax chains that produce
    non-contiguous layouts Spyre's restickify and bmm_padding passes cannot reconcile.
    Run entirely on CPU. The forward patch only converts activations (not weights)
    on each call — weights are moved to CPU once at apply() time via
    move_blip2_qformer_weights_to_cpu(), eliminating 2 PCIe weight round-trips
    per layer per image (24 transfers saved across 12 Q-Former layers).
    """
    try:
        from vllm.model_executor.models.blip2 import Blip2QFormerMultiHeadAttention
    except ImportError:
        return

    if getattr(Blip2QFormerMultiHeadAttention.forward, "_spyre_patched", False):
        return

    _orig_blip2_attn_forward = Blip2QFormerMultiHeadAttention.forward

    def _blip2_attn_forward_cpu(self, hidden_states, encoder_hidden_states=None):
        target_device = hidden_states.device
        hidden_states = convert(hidden_states, device="cpu")
        if encoder_hidden_states is not None:
            encoder_hidden_states = convert(encoder_hidden_states, device="cpu")
        out = _orig_blip2_attn_forward(self, hidden_states, encoder_hidden_states)
        return convert(out, device=target_device)

    _blip2_attn_forward_cpu._spyre_patched = True  # type: ignore[attr-defined]
    Blip2QFormerMultiHeadAttention.forward = _blip2_attn_forward_cpu  # type: ignore[method-assign]
    logger.info(
        "Spyre: patched Blip2QFormerMultiHeadAttention.forward to run on CPU "
        "(permute/matmul chains not restickifiable on Spyre)."
    )


def move_blip2_qformer_weights_to_cpu(model: torch.nn.Module) -> None:
    """Move all Blip2QFormerMultiHeadAttention weights to CPU permanently.

    Called once at apply() time so the per-forward patch only needs to transfer
    activations, not weights. Eliminates 2 PCIe weight round-trips per layer
    per image (self.to('cpu') + self.to(device) inside the forward patch).
    """
    try:
        from vllm.model_executor.models.blip2 import Blip2QFormerMultiHeadAttention
    except ImportError:
        return

    moved = 0
    for module in model.modules():
        if isinstance(module, Blip2QFormerMultiHeadAttention):
            module.to("cpu")
            moved += 1
    if moved:
        logger.info(
            "Spyre: moved %d Blip2QFormerMultiHeadAttention modules to CPU permanently "
            "(weights stay on CPU; only activations are transferred per call).",
            moved,
        )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply BLIP-2 Q-Former workarounds."""
    patch_blip2_qformer_attention()
    move_blip2_qformer_weights_to_cpu(model)
