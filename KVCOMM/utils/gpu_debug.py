from __future__ import annotations

import csv
import gc
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
from transformers.cache_utils import DynamicCache

DEFAULT_TARGET_DEVICE = 0


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _is_scoped_dict_like(obj: Any) -> bool:
    return hasattr(obj, "_base") and hasattr(obj, "_local")


def _walk_cuda_tensors(
    obj: Any,
    *,
    path: str,
    target_device: Optional[int],
    seen_objects: Set[int],
    seen_tensors: Set[int],
    out: List[Dict[str, Any]],
) -> None:
    obj_id = id(obj)
    if obj_id in seen_objects:
        return
    seen_objects.add(obj_id)

    if isinstance(obj, torch.Tensor):
        if obj.is_cuda and (target_device is None or obj.device.index == target_device):
            ptr = obj.data_ptr()
            if ptr not in seen_tensors:
                seen_tensors.add(ptr)
                out.append(
                    {
                        "path": path,
                        "shape": tuple(obj.shape),
                        "dtype": str(obj.dtype),
                        "device": str(obj.device),
                        "mb": _tensor_nbytes(obj) / 1024**2,
                    }
                )
        return

    if isinstance(obj, DynamicCache):
        if hasattr(obj, "layers"):
            for idx, layer in enumerate(getattr(obj, "layers", [])):
                _walk_cuda_tensors(
                    getattr(layer, "keys", None),
                    path=f"{path}.layers[{idx}].keys",
                    target_device=target_device,
                    seen_objects=seen_objects,
                    seen_tensors=seen_tensors,
                    out=out,
                )
                _walk_cuda_tensors(
                    getattr(layer, "values", None),
                    path=f"{path}.layers[{idx}].values",
                    target_device=target_device,
                    seen_objects=seen_objects,
                    seen_tensors=seen_tensors,
                    out=out,
                )
        else:
            for idx, key in enumerate(getattr(obj, "key_cache", [])):
                _walk_cuda_tensors(
                    key,
                    path=f"{path}.key_cache[{idx}]",
                    target_device=target_device,
                    seen_objects=seen_objects,
                    seen_tensors=seen_tensors,
                    out=out,
                )
            for idx, value in enumerate(getattr(obj, "value_cache", [])):
                _walk_cuda_tensors(
                    value,
                    path=f"{path}.value_cache[{idx}]",
                    target_device=target_device,
                    seen_objects=seen_objects,
                    seen_tensors=seen_tensors,
                    out=out,
                )
        return

    if isinstance(obj, dict):
        for key, value in obj.items():
            _walk_cuda_tensors(
                value,
                path=f"{path}[{repr(key)}]",
                target_device=target_device,
                seen_objects=seen_objects,
                seen_tensors=seen_tensors,
                out=out,
            )
        return

    if isinstance(obj, (list, tuple, set)):
        for idx, value in enumerate(obj):
            _walk_cuda_tensors(
                value,
                path=f"{path}[{idx}]",
                target_device=target_device,
                seen_objects=seen_objects,
                seen_tensors=seen_tensors,
                out=out,
            )
        return

    if _is_scoped_dict_like(obj):
        _walk_cuda_tensors(
            getattr(obj, "_base", None),
            path=f"{path}._base",
            target_device=target_device,
            seen_objects=seen_objects,
            seen_tensors=seen_tensors,
            out=out,
        )
        _walk_cuda_tensors(
            getattr(obj, "_local", None),
            path=f"{path}._local",
            target_device=target_device,
            seen_objects=seen_objects,
            seen_tensors=seen_tensors,
            out=out,
        )
        return

    if hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            _walk_cuda_tensors(
                value,
                path=f"{path}.{key}",
                target_device=target_device,
                seen_objects=seen_objects,
                seen_tensors=seen_tensors,
                out=out,
            )


def collect_cuda_tensor_info(
    name: str,
    obj: Any,
    *,
    target_device: Optional[int] = DEFAULT_TARGET_DEVICE,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    _walk_cuda_tensors(
        obj,
        path=name,
        target_device=target_device,
        seen_objects=set(),
        seen_tensors=set(),
        out=entries,
    )
    entries.sort(key=lambda item: item["mb"], reverse=True)
    total_mb = sum(item["mb"] for item in entries)
    return {
        "name": name,
        "count": len(entries),
        "total_mb": total_mb,
        "entries": entries,
    }


def current_cuda_memory(device: Optional[int] = DEFAULT_TARGET_DEVICE) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}
    device_arg = torch.device(f"cuda:{device}") if device is not None else None
    return {
        "allocated_mb": torch.cuda.memory_allocated(device_arg) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device_arg) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device_arg) / 1024**2,
    }


def reset_cuda_peak_memory(device: Optional[int] = DEFAULT_TARGET_DEVICE) -> None:
    """Reset CUDA peak stats so later max_allocated reflects a local interval."""
    if not torch.cuda.is_available():
        return
    device_arg = torch.device(f"cuda:{device}") if device is not None else None
    torch.cuda.reset_peak_memory_stats(device_arg)


def summarize_kvcomm_cuda_state(
    *,
    topk: int = 20,
    include_gc_tensors: bool = False,
    target_device: Optional[int] = DEFAULT_TARGET_DEVICE,
) -> Dict[str, Any]:
    from KVCOMM.llm.gpt_chat import LLMChat
    from KVCOMM.llm.kvcomm_engine import KVCOMMEngine

    sections = [
        ("anchors", KVCOMMEngine.anchors),
        ("weight_dict", KVCOMMEngine.weight_dict),
        ("request_states", KVCOMMEngine._request_states),
        ("staged_commits", KVCOMMEngine._staged_commits),
        ("shared_kv_cache_memory", LLMChat._shared_kv_cache_memory),
    ]

    report = {
        "memory": current_cuda_memory(target_device),
        "target_device": target_device,
        "sections": [],
    }
    for name, obj in sections:
        section = collect_cuda_tensor_info(name, obj, target_device=target_device)
        section["entries"] = section["entries"][:topk]
        report["sections"].append(section)

    if include_gc_tensors:
        gc_entries: List[Dict[str, Any]] = []
        seen_ptrs: Set[int] = set()
        for obj in gc.get_objects():
            try:
                if (
                    isinstance(obj, torch.Tensor)
                    and obj.is_cuda
                    and (target_device is None or obj.device.index == target_device)
                ):
                    ptr = obj.data_ptr()
                    if ptr in seen_ptrs:
                        continue
                    seen_ptrs.add(ptr)
                    gc_entries.append(
                        {
                            "shape": tuple(obj.shape),
                            "dtype": str(obj.dtype),
                            "device": str(obj.device),
                            "mb": _tensor_nbytes(obj) / 1024**2,
                        }
                    )
            except Exception:
                continue
        gc_entries.sort(key=lambda item: item["mb"], reverse=True)
        report["gc_cuda_tensors"] = gc_entries[:topk]

    return report


def append_kvcomm_cuda_state_csv(
    output_dir: Optional[str | Path],
    *,
    tag: str,
    batch_index: Optional[int] = None,
    phase: Optional[str] = None,
    topk: int = 0,
    include_gc_tensors: bool = False,
    target_device: Optional[int] = DEFAULT_TARGET_DEVICE,
    filename: str = "CudaMemory.csv",
) -> None:
    """Append a structured CUDA memory snapshot for experiment comparisons."""
    if output_dir is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    report = summarize_kvcomm_cuda_state(
        topk=topk,
        include_gc_tensors=include_gc_tensors,
        target_device=target_device,
    )
    memory = report["memory"]
    row: Dict[str, Any] = {
        "timestamp": time.time(),
        "tag": tag,
        "batch_index": batch_index if batch_index is not None else "",
        "phase": phase or "",
        "target_device": "" if target_device is None else target_device,
        "allocated_mb": memory["allocated_mb"],
        "reserved_mb": memory["reserved_mb"],
        "max_allocated_mb": memory["max_allocated_mb"],
    }
    for section in report["sections"]:
        name = section["name"]
        row[f"{name}_cuda_tensor_count"] = section["count"]
        row[f"{name}_cuda_tensor_mb"] = section["total_mb"]

    csv_path = Path(output_dir).expanduser() / filename
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "tag",
        "batch_index",
        "phase",
        "target_device",
        "allocated_mb",
        "reserved_mb",
        "max_allocated_mb",
        "anchors_cuda_tensor_count",
        "anchors_cuda_tensor_mb",
        "weight_dict_cuda_tensor_count",
        "weight_dict_cuda_tensor_mb",
        "request_states_cuda_tensor_count",
        "request_states_cuda_tensor_mb",
        "staged_commits_cuda_tensor_count",
        "staged_commits_cuda_tensor_mb",
        "shared_kv_cache_memory_cuda_tensor_count",
        "shared_kv_cache_memory_cuda_tensor_mb",
    ]
    file_exists = csv_path.exists()
    if file_exists:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if existing_fieldnames and existing_fieldnames != fieldnames:
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for existing_row in existing_rows:
                    writer.writerow({key: existing_row.get(key, "") for key in fieldnames})
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def print_kvcomm_cuda_state(
    *,
    tag: Optional[str] = None,
    topk: int = 20,
    include_gc_tensors: bool = False,
    target_device: Optional[int] = DEFAULT_TARGET_DEVICE,
) -> None:
    report = summarize_kvcomm_cuda_state(
        topk=topk,
        include_gc_tensors=include_gc_tensors,
        target_device=target_device,
    )
    prefix = f"[{tag}] " if tag else ""
    memory = report["memory"]
    device_label = (
        f"cuda:{report['target_device']}"
        if report["target_device"] is not None
        else "all_cuda_devices"
    )
    print(
        f"{prefix}CUDA memory ({device_label}): "
        f"allocated={memory['allocated_mb']:.1f}MB "
        f"reserved={memory['reserved_mb']:.1f}MB "
        f"max_allocated={memory['max_allocated_mb']:.1f}MB"
    )
    for section in report["sections"]:
        print(
            f"{prefix}{section['name']}: "
            f"cuda_tensors={section['count']} total={section['total_mb']:.1f}MB"
        )
        for item in section["entries"]:
            print(
                f"{prefix}  - {item['path']} "
                f"shape={item['shape']} dtype={item['dtype']} "
                f"device={item['device']} size={item['mb']:.1f}MB"
            )
    if include_gc_tensors:
        gc_entries = report.get("gc_cuda_tensors", [])
        print(f"{prefix}gc_cuda_tensors: {len(gc_entries)} shown")
        for item in gc_entries:
            print(
                f"{prefix}  - shape={item['shape']} dtype={item['dtype']} "
                f"device={item['device']} size={item['mb']:.1f}MB"
            )
