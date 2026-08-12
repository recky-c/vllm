# SPDX-License-Identifier: Apache-2.0
"""Physical KV-cache placement policies shared by cache planning."""

from collections import defaultdict

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheSpec,
    UniformTypeKVCacheSpecs,
)


def project_kv_cache_groups_to_worker(
    global_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
) -> list[KVCacheGroupSpec]:
    """Project global logical groups onto one worker's physical layers."""
    projected_groups: list[KVCacheGroupSpec] = []
    for group in global_groups:
        worker_layer_names = [
            layer_name for layer_name in group.layer_names if layer_name in worker_spec
        ]
        group_spec = group.kv_cache_spec
        if worker_layer_names and isinstance(group_spec, UniformTypeKVCacheSpecs):
            group_spec = UniformTypeKVCacheSpecs(
                block_size=group_spec.block_size,
                kv_cache_specs={
                    layer_name: group_spec.kv_cache_specs[layer_name]
                    for layer_name in worker_layer_names
                },
            )
        projected_groups.append(
            KVCacheGroupSpec(
                worker_layer_names,
                group_spec,
                is_eagle_group=group.is_eagle_group and bool(worker_layer_names),
            )
        )
    return projected_groups


def get_kvpp_layer_owners(
    kv_cache_specs: dict[str, KVCacheSpec], kvpp_size: int
) -> dict[str, int]:
    """Assign contiguous transformer-layer bundles to KVPP ranks."""
    layers_by_index: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_cache_specs:
        layers_by_index[extract_layer_index(layer_name)].append(layer_name)

    layer_indices = sorted(layers_by_index)
    if len(layer_indices) < kvpp_size:
        raise ValueError(
            f"KVPP size ({kvpp_size}) exceeds the number of KV cache layer "
            f"bundles ({len(layer_indices)})."
        )

    base, remainder = divmod(len(layer_indices), kvpp_size)
    owners: dict[str, int] = {}
    offset = 0
    for owner_rank in range(kvpp_size):
        partition_size = base + int(owner_rank < remainder)
        for layer_index in layer_indices[offset : offset + partition_size]:
            for layer_name in layers_by_index[layer_index]:
                owners[layer_name] = owner_rank
        offset += partition_size
    return owners


def get_kvpp_allocation_groups(
    logical_groups: list[KVCacheGroupSpec],
    worker_spec: dict[str, KVCacheSpec],
    owners: dict[str, int],
    kvpp_rank: int,
) -> tuple[list[KVCacheGroupSpec], dict[str, list[str]]]:
    """Select persistent layers plus two scratch caches per physical layout."""
    allocation_spec: dict[str, KVCacheSpec] = {}
    scratch_aliases: dict[str, list[str]] = {}

    for group in logical_groups:
        local_names = [name for name in group.layer_names if name in worker_spec]
        owned_names = [name for name in local_names if owners[name] == kvpp_rank]
        non_owned_names = [name for name in local_names if owners[name] != kvpp_rank]
        allocation_names = list(owned_names)
        scratch_layout_groups: list[list[str]] = []
        for name in non_owned_names:
            for layout_names in scratch_layout_groups:
                if worker_spec[name] == worker_spec[layout_names[0]]:
                    layout_names.append(name)
                    break
            else:
                scratch_layout_groups.append([name])
        for layout_names in scratch_layout_groups:
            scratch_names = layout_names[:2]
            allocation_names.extend(scratch_names)
            for scratch_index, scratch_name in enumerate(scratch_names):
                scratch_aliases[scratch_name] = layout_names[
                    scratch_index :: len(scratch_names)
                ]
        for layer_name in allocation_names:
            allocation_spec[layer_name] = worker_spec[layer_name]

    return (
        project_kv_cache_groups_to_worker(logical_groups, allocation_spec),
        scratch_aliases,
    )


def expand_kvpp_scratch_aliases(
    kv_cache_config: KVCacheConfig, scratch_aliases: dict[str, list[str]]
) -> None:
    """Bind every non-owned logical layer to its alternating scratch cache."""
    for tensor in kv_cache_config.kv_cache_tensors:
        expanded_names: list[str] = []
        for layer_name in tensor.shared_by:
            expanded_names.extend(scratch_aliases.get(layer_name, [layer_name]))
        tensor.shared_by = list(dict.fromkeys(expanded_names))
