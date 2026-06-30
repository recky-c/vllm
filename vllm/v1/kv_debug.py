# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from collections.abc import Iterable
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def kv_debug_enabled() -> bool:
    return os.environ.get("VLLM_KV_DEBUG", "").lower() in _TRUE_VALUES


def kv_debug_log(logger: Any, msg: str, *args: Any) -> None:
    if kv_debug_enabled():
        logger.info("[KV_DEBUG] " + msg, *args)


def kv_block_ids_summary(blocks: Any, max_items: int = 16) -> Any:
    if blocks is None:
        return None
    if hasattr(blocks, "get_block_ids"):
        try:
            return blocks.get_block_ids(allow_none=True)
        except TypeError:
            return blocks.get_block_ids()
    if isinstance(blocks, Iterable) and not isinstance(blocks, (str, bytes, dict)):
        ids = []
        for idx, block in enumerate(blocks):
            if idx >= max_items:
                ids.append("...")
                break
            ids.append(getattr(block, "block_id", block))
        return ids
    return blocks


def kv_tensor_summary(tensor: Any, max_items: int = 8) -> str:
    if tensor is None:
        return "None"
    shape = tuple(getattr(tensor, "shape", ()))
    dtype = getattr(tensor, "dtype", None)
    device = getattr(tensor, "device", None)
    stride = None
    try:
        stride = tuple(tensor.stride())
    except Exception:
        pass
    summary = f"shape={shape}, dtype={dtype}, device={device}, stride={stride}"
    if os.environ.get("VLLM_KV_DEBUG_VALUES", "").lower() in _TRUE_VALUES:
        try:
            values = tensor.detach().flatten()[:max_items].cpu().tolist()
            summary += f", first_values={values}"
        except Exception as exc:
            summary += f", first_values_error={type(exc).__name__}: {exc}"
    return summary


def kv_spec_summary(spec: Any) -> dict[str, Any]:
    return {
        "type": type(spec).__name__,
        "block_size": getattr(spec, "block_size", None),
        "storage_block_size": getattr(spec, "storage_block_size", None),
        "page_size_bytes": getattr(spec, "page_size_bytes", None),
        "num_kv_heads": getattr(spec, "num_kv_heads", None),
        "head_size": getattr(spec, "head_size", None),
        "dtype": str(getattr(spec, "dtype", None)),
    }
