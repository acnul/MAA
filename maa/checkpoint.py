from typing import Dict, Mapping, MutableMapping, Tuple

import torch
import torch.nn as nn


def _strip_module_prefix(key: str) -> str:
    return key[7:] if key.startswith("module.") else key


def remap_legacy_adapter_keys(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        key = _strip_module_prefix(key)
        key = key.replace(".content_adapter.", ".maa_adapter.")
        remapped[key] = value
    return remapped


def _select_state_dict(checkpoint: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, MutableMapping):
        raise TypeError("Adapter checkpoint must be a state dict or a mapping containing a state dict.")

    for key in ("adapter_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, MutableMapping):
            return value
    return checkpoint


def extract_maa_adapter_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if ".maa_adapter." in key}


def save_maa_adapter_state(model: nn.Module, path: str) -> None:
    torch.save(extract_maa_adapter_state(model), path)


def load_maa_adapter_state(
    model: nn.Module,
    adapter_path: str,
    map_location: str = "cpu",
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    checkpoint = torch.load(adapter_path, map_location=map_location)
    state_dict = remap_legacy_adapter_keys(_select_state_dict(checkpoint))
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return tuple(incompatible.missing_keys), tuple(incompatible.unexpected_keys)
