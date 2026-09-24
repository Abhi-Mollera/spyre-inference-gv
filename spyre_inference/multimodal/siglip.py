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

"""SigLIP vision-encoder workarounds for Spyre."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)


def patch_siglip_vision_embeddings(model: torch.nn.Module, device: torch.device) -> None:
    """Patch SiglipVisionEmbeddings.forward to run the position embedding on CPU.

    aten.embedding(position_embedding.weight, position_ids) called eagerly on
    Spyre tensors hits torch-spyre's compile_once eager kernel, causing Dynamo
    re-entrancy / RecursionError when tracing. Keeping the position embedding lookup
    on CPU avoids this.
    """
    try:
        from vllm.model_executor.models.siglip import SiglipVisionEmbeddings
    except ImportError:
        return

    if getattr(SiglipVisionEmbeddings.forward, "_spyre_patched", False):
        return

    def _siglip_embeddings_forward(
        self: SiglipVisionEmbeddings,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        _, _, height, width = pixel_values.shape
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        # Download once: position_embedding and position_ids are pinned to CPU
        # (see module setup below), and aten.embedding / aten.add called eagerly
        # on Spyre tensors hit compile_once, causing Dynamo re-entrancy.
        embeddings_cpu = convert(embeddings, device="cpu")
        if interpolate_pos_encoding:
            pos_emb = self.interpolate_pos_encoding(embeddings_cpu, height, width)
        else:
            pos_emb = self.position_embedding(self.position_ids)
        return convert(embeddings_cpu + pos_emb, device=device)

    _siglip_embeddings_forward._spyre_patched = True  # type: ignore[attr-defined]
    SiglipVisionEmbeddings.forward = _siglip_embeddings_forward  # type: ignore[method-assign]

    # Pin weights and buffers to CPU on every existing instance so the forward
    # replacement can perform the embedding lookup there without a device mismatch.
    for module in model.modules():
        if isinstance(module, SiglipVisionEmbeddings):
            module.position_embedding.to("cpu")
            module.register_buffer(
                "position_ids",
                module.position_ids.to("cpu"),  # ty: ignore[invalid-argument-type]
                persistent=False,
            )

    logger.info(
        "Spyre: patched SiglipVisionEmbeddings.forward to run the position "
        "embedding lookup on CPU (aten.embedding not traceable on Spyre)."
    )


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Apply SigLIP vision workarounds."""
    patch_siglip_vision_embeddings(model, device)
