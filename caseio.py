"""离线读取合成采集资料。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_case(path: Path):
    """读出交接记录和原始张量。"""
    envelope = json.loads(path.read_text(encoding="utf-8"))
    arrays = {}
    for key, descriptor in envelope["tensors"].items():
        value = np.asarray(descriptor["values"], dtype=descriptor["dtype"])
        if tuple(descriptor["shape"]) != value.shape:
            raise ValueError(f"张量 {key} 与声明形状不一致")
        arrays[key] = value
    return envelope["records"], arrays


def main():
    parser = argparse.ArgumentParser(description="查看本地采集记录与张量尺寸")
    parser.add_argument("path", type=Path)
    parser.add_argument("--handoff", action="store_true", help="执行带围栏的转置分区交接并输出 JSON")
    parser.add_argument("--values", action="store_true", help="交接结果中包含 uint16 视图数值")
    settings = parser.parse_args()

    if settings.handoff:
        from handoff import execute_handoff
        import json as _json

        records, arrays = load_case(settings.path)
        capacity_path = settings.path.parent / "capacity.json"
        capacity = _json.loads(capacity_path.read_text(encoding="utf-8")) if capacity_path.exists() else None
        result = execute_handoff(records, arrays, capacity)
        print(_json.dumps(result.to_dict(include_values=settings.values), ensure_ascii=False, indent=2))
        raise SystemExit(0 if result.ok else 2)

    records, arrays = load_case(settings.path)
    print("记录字段", "、".join(sorted(records)))
    for key, array in sorted(arrays.items()):
        print(key, "形状", list(array.shape), "类型", array.dtype)


if __name__ == "__main__":
    main()
