"""Streaming, fenced heterogeneous tensor handoff.

The public entry point for an offline capture is :func:`execute_handoff` or the
``handoff`` console entry::

    python handoff.py data/capture.json

The module deliberately works on raw page bytes.  Logical elements are
little-endian uint16, but no global dense NCHW/NHWC tensor is materialised:
NumPy strided views and one aligned link burst at a time are used instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np

AXES = "NCHW"
AXIS_INDEX = {axis: i for i, axis in enumerate(AXES)}
ELEMENT_BYTES = 2
DEFAULT_GENERATION = 0


class HandoffContractError(ValueError):
    """Raised when a caller asks for an impossible synchronous operation."""


# ---------------------------------------------------------------------------
# Interval and descriptor helpers
# ---------------------------------------------------------------------------

Interval = tuple[int, int]
Box = tuple[tuple[int, int], ...]


def iv(value: Sequence[int]) -> Interval:
    start, end = int(value[0]), int(value[1])
    if end < start:
        raise HandoffContractError(f"反向区间: {value!r}")
    return start, end


def iv_overlap(a: Interval, b: Interval) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def iv_intersection(a: Interval, b: Interval) -> Interval | None:
    start, end = max(a[0], b[0]), min(a[1], b[1])
    return (start, end) if start < end else None


def iv_contains(outer: Interval | Sequence[int], inner: Interval | Sequence[int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def union_intervals(intervals: Iterable[Interval]) -> list[Interval]:
    ordered = sorted(iv(x) for x in intervals if x[1] > x[0])
    out: list[Interval] = []
    for start, end in ordered:
        if not out or start > out[-1][1]:
            out.append((start, end))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], end))
    return out


def subtract_intervals(range_: Interval, covered: Iterable[Interval]) -> list[Interval]:
    """Return pieces of ``range_`` not covered by the union."""
    result: list[Interval] = []
    cursor = range_[0]
    for start, end in union_intervals(covered):
        if end <= cursor or start >= range_[1]:
            continue
        start = max(start, range_[0])
        end = min(end, range_[1])
        if cursor < start:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < range_[1]:
        result.append((cursor, range_[1]))
    return result


def box_intersection(a: Box, b: Box) -> Box | None:
    out = []
    for ia, ib in zip(a, b):
        lo, hi = max(ia[0], ib[0]), min(ia[1], ib[1])
        if lo >= hi:
            return None
        out.append((lo, hi))
    return tuple(out)


def box_contains(box: Box, coord: Sequence[int]) -> bool:
    return all(lo <= value < hi for value, (lo, hi) in zip(coord, box))


def box_volume(box: Box | None) -> int:
    if box is None:
        return 0
    return math.prod(max(0, hi - lo) for lo, hi in box)


def iter_box_coords(box: Box) -> Iterator[tuple[int, ...]]:
    ranges = [range(lo, hi) for lo, hi in box]
    if len(ranges) != 4:
        raise HandoffContractError("逻辑盒必须有 N,C,H,W 四个轴")
    for n in ranges[0]:
        for c in ranges[1]:
            for h in ranges[2]:
                for w in ranges[3]:
                    yield n, c, h, w


def iter_coords_in_byte_range(
    layout: "Layout",
    query: Interval,
    clip: Box | None = None,
) -> Iterator[tuple[int, ...]]:
    """Yield logical coordinates whose uint16 span overlaps ``query``.

    Iteration follows the outer physical axes and uses arithmetic bounds on
    the innermost stride, so replaying every burst visits each element only a
    constant number of times.
    """
    q_start, q_end = iv(query)
    intersection = box_intersection(layout.box, clip) if clip is not None else layout.box
    if intersection is None:
        return

    order = layout.layout
    inner_axis = order[-1]
    inner_index = AXIS_INDEX[inner_axis]
    outer_axes = order[:-1]
    outer_ranges = [
        range(intersection[AXIS_INDEX[axis]][0], intersection[AXIS_INDEX[axis]][1])
        for axis in outer_axes
    ]
    inner_clip_lo, inner_clip_hi = intersection[inner_index]
    full_inner_lo = layout.box[inner_index][0]
    inner_size = layout.sizes_global[inner_index]
    stride = layout.global_strides[inner_index]

    for outer_values in _product_ranges(outer_ranges):
        coord = [0, 0, 0, 0]
        for axis, value in zip(outer_axes, outer_values):
            coord[AXIS_INDEX[axis]] = value
        coord[inner_index] = full_inner_lo
        run_start = layout.offset(coord)

        if stride <= 0:
            for global_inner in range(inner_clip_lo, inner_clip_hi):
                coord[inner_index] = global_inner
                if iv_overlap(layout.element_interval(coord), query):
                    yield tuple(coord)
            continue

        clip_k0 = inner_clip_lo - full_inner_lo
        clip_k1_exclusive = inner_clip_hi - full_inner_lo
        # start(k)+ELEMENT_BYTES > q_start
        query_k0 = (q_start - run_start - ELEMENT_BYTES) // stride + 1
        # start(k) < q_end  => k*stride < q_end-run_start
        query_k1 = (q_end - run_start - 1) // stride
        k0 = max(0, clip_k0, query_k0)
        k1 = min(inner_size - 1, clip_k1_exclusive - 1, query_k1)
        for k in range(k0, k1 + 1):
            coord[inner_index] = full_inner_lo + k
            yield tuple(coord)


def boxes_from_w_runs(
    coords_by_reason: dict[str, set[tuple[int, ...]]],
    fallback_reason: str = "incomplete",
) -> list[dict[str, Any]]:
    """Compress coordinates into exact N,C,H,W half-open boxes.

    The returned boxes merge contiguous W coordinates.  This is deliberately
    conservative but never reports a range containing a healthy coordinate.
    """
    grouped: dict[str, dict[tuple[int, int, int], list[int]]] = {}
    for reason, coords in coords_by_reason.items():
        groups: dict[tuple[int, int, int], list[int]] = {}
        for n, c, h, w in coords:
            groups.setdefault((n, c, h), []).append(w)
        grouped[reason] = groups

    reports: list[dict[str, Any]] = []
    for reason, groups in grouped.items():
        boxes: list[list[list[int]]] = []
        for (n, c, h), ws in sorted(groups.items()):
            ws = sorted(set(ws))
            if not ws:
                continue
            start = prev = ws[0]
            for w in ws[1:]:
                if w == prev + 1:
                    prev = w
                    continue
                boxes.append([[n, n + 1], [c, c + 1], [h, h + 1], [start, prev + 1]])
                start = prev = w
            boxes.append([[n, n + 1], [c, c + 1], [h, h + 1], [start, prev + 1]])
        if boxes:
            reports.append({"reason": reason or fallback_reason, "boxes": boxes})
    return reports


# ---------------------------------------------------------------------------
# Descriptor wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    descriptor: dict[str, Any]
    role: str
    index: int

    @property
    def id(self) -> str:
        return str(self.descriptor["id"])

    @property
    def page(self) -> str:
        return str(self.descriptor.get("page", self.descriptor.get("tensor", "")))

    @property
    def tensor(self) -> str:
        return str(self.descriptor["tensor"])

    @property
    def box(self) -> Box:
        return tuple(iv(x) for x in self.descriptor["box"])  # type: ignore[return-value]

    @property
    def layout(self) -> str:
        value = str(self.descriptor["layout"]).upper()
        if sorted(value) != sorted(AXES):
            raise HandoffContractError(f"{self.id}: layout {value!r} 不是 NCHW 的排列")
        return value

    @property
    def base(self) -> int:
        return int(self.descriptor.get("base", 0))

    @property
    def global_strides(self) -> tuple[int, int, int, int]:
        order = self.layout
        raw_strides = list(self.descriptor["strides"])
        if len(raw_strides) != 4 or len(order) != 4:
            raise HandoffContractError(f"{self.id}: layout/strides 长度必须为 4")
        by_axis = {axis: int(stride) for axis, stride in zip(order, raw_strides)}
        return tuple(by_axis[axis] for axis in AXES)  # type: ignore[return-value]

    @property
    def local_strides(self) -> tuple[int, ...]:
        return tuple(int(x) for x in self.descriptor["strides"])

    @property
    def sizes_global(self) -> tuple[int, int, int, int]:
        return tuple(hi - lo for lo, hi in self.box)  # type: ignore[return-value]

    def offset(self, coord: Sequence[int]) -> int:
        strides = self.global_strides
        result = self.base
        for axis, value in enumerate(coord):
            lo = self.box[axis][0]
            result += (int(value) - lo) * strides[axis]
        return result

    def element_interval(self, coord: Sequence[int]) -> Interval:
        start = self.offset(coord)
        return start, start + ELEMENT_BYTES

    def byte_element_view(self, page: np.ndarray) -> np.ndarray:
        """Zero-copy uint8 view with a trailing two-byte element axis."""
        shape = self.sizes_global + (ELEMENT_BYTES,)
        strides = tuple(self.global_strides) + (1,)
        return np.ndarray(
            shape=shape,
            dtype=np.uint8,
            buffer=page,
            offset=self.base,
            strides=strides,
        )

    def uint16_view(self, page: np.ndarray, global_order: bool = False) -> np.ndarray:
        """Return a device-readable uint16 view sharing the page bytes."""
        if global_order:
            shape = self.sizes_global
            strides = self.global_strides
        else:
            order = self.layout
            sizes = self.sizes_global
            shape = tuple(sizes[AXIS_INDEX[axis]] for axis in order)
            strides = self.local_strides
        return np.ndarray(
            shape=shape,
            dtype="<u2",
            buffer=page,
            offset=self.base,
            strides=tuple(int(x) for x in strides),
        )

    def logical_intervals(self, clip: Box | None = None) -> Iterator[Interval]:
        """Yield contiguous byte runs, avoiding per-padding-byte materialisation."""
        active = box_intersection(self.box, clip) if clip is not None else self.box
        if active is None:
            return

        order = self.layout
        inner = order[-1]
        inner_index = AXIS_INDEX[inner]
        inner_lo, inner_hi = active[inner_index]
        inner_count = inner_hi - inner_lo
        inner_stride = int(self.descriptor["strides"][-1])

        outer_axes = order[:-1]
        outer_ranges = [
            range(active[AXIS_INDEX[axis]][0], active[AXIS_INDEX[axis]][1])
            for axis in outer_axes
        ]
        for outer_values in _product_ranges(outer_ranges):
            coord = [0, 0, 0, 0]
            for axis, value in zip(outer_axes, outer_values):
                coord[AXIS_INDEX[axis]] = value
            coord[inner_index] = inner_lo
            start = self.offset(coord)
            if inner_stride == ELEMENT_BYTES:
                yield start, start + inner_count * ELEMENT_BYTES
            else:
                for global_inner in range(inner_lo, inner_hi):
                    coord[inner_index] = global_inner
                    item_start = self.offset(coord)
                    yield item_start, item_start + ELEMENT_BYTES

    def geometry_span(self) -> int:
        """Half-open byte span from the first element to the end of the last."""
        order = self.layout
        sizes_in_layout = [
            self.box[AXIS_INDEX[axis]][1] - self.box[AXIS_INDEX[axis]][0]
            for axis in order
        ]
        strides = self.local_strides
        child_span = ELEMENT_BYTES
        for size, stride in reversed(list(zip(sizes_in_layout, strides))):
            if size > 1 and stride < child_span:
                raise HandoffContractError(
                    f"{self.id}: 物理步幅 {stride} 小于内层跨度 {child_span}，"
                    "不同逻辑元素的字节会重叠"
                )
            child_span = (size - 1) * stride + child_span
        return child_span


def _product_ranges(ranges: Sequence[range]) -> Iterator[tuple[int, ...]]:
    if not ranges:
        yield ()
        return
    if len(ranges) == 1:
        for value in ranges[0]:
            yield (value,)
        return
    head = ranges[0]
    for value in head:
        for tail in _product_ranges(ranges[1:]):
            yield (value,) + tail


# ---------------------------------------------------------------------------
# Static plan
# ---------------------------------------------------------------------------


@dataclass
class Diagnostic:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "error"

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details,
                "severity": self.severity}


@dataclass
class BlockPlan:
    page: str
    start: int
    end: int
    destinations: set[str] = field(default_factory=set)
    source_ids: set[str] = field(default_factory=set)
    fill_ranges: dict[str, list[Interval]] = field(default_factory=dict)
    errors: list[Diagnostic] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int]:
        return self.page, self.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "range": [self.start, self.end],
            "destinations": sorted(self.destinations),
            "source_ids": sorted(self.source_ids),
            "errors": [x.as_dict() for x in self.errors],
            "feasible": not self.errors,
        }


def populate_block_source_ids(plan: "HandoffPlan") -> None:
    """Attach exact source owners to every aligned physical block.

    Normal layouts have a contiguous innermost axis, so this walks physical
    runs rather than every activation element.
    """
    for dest in plan.destinations:
        for source in plan.sources:
            intersection = box_intersection(dest.box, source.box)
            if intersection is None:
                continue
            for interval in dest.logical_intervals(intersection):
                for block_start in _blocks_for_interval(interval, plan.burst_align):
                    block = plan.blocks.get((dest.page, block_start))
                    if block is not None:
                        block.source_ids.add(source.id)


@dataclass
class HandoffPlan:
    record: dict[str, Any]
    generation: int
    sources: list[Layout]
    destinations: list[Layout]
    page_sizes: dict[str, int]
    burst_align: int
    link_max: int
    capacity: dict[str, Any] | None
    blocks: dict[tuple[str, int], BlockPlan]
    ownership: dict[str, dict[str, Any]]
    diagnostics: list[Diagnostic]
    scratch_budget: int | None
    scratch_required: int

    @property
    def feasible(self) -> bool:
        return not any(d.severity == "error" for d in self.diagnostics) and all(
            not b.errors for b in self.blocks.values()
        )

    @property
    def ok(self) -> bool:
        return self.feasible

    @property
    def valid(self) -> bool:
        return self.feasible

    def diagnostic_dicts(self) -> list[dict[str, Any]]:
        return [d.as_dict() for d in self.diagnostics] + [
            e.as_dict()
            for block in self.blocks.values()
            for e in block.errors
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "feasible": self.feasible,
            "burst_align": self.burst_align,
            "link_max": self.link_max,
            "scratch_budget": self.scratch_budget,
            "scratch_required": self.scratch_required,
            "streaming": True,
            "ownership": self.ownership,
            "blocks": [self.blocks[k].as_dict() for k in sorted(self.blocks)],
            "diagnostics": self.diagnostic_dicts(),
        }


def tensor_to_page(descriptor: Any) -> np.ndarray:
    """Convert a capture tensor descriptor to raw little-endian page bytes."""
    if isinstance(descriptor, np.ndarray):
        array = descriptor
        if array.dtype == np.uint8:
            return array.copy()
        return np.ascontiguousarray(array.astype("<u2")).view(np.uint8).copy()
    if isinstance(descriptor, dict):
        dtype = str(descriptor.get("dtype", "uint8")).lower()
        values = descriptor.get("values", [])
    else:
        dtype = "uint8"
        values = descriptor

    if dtype in {"uint16", "<u2", "u2"}:
        array = np.asarray(values, dtype="<u2")
        return np.ascontiguousarray(array).view(np.uint8).copy()
    # uint8 and unknown one-byte captures are treated as raw bytes.
    return np.asarray(values, dtype=np.uint8).copy()


def _page_sizes_from_tensors(tensors: dict[str, Any] | None) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if not tensors:
        return sizes
    for name, descriptor in tensors.items():
        try:
            sizes[name] = int(tensor_to_page(descriptor).size)
        except Exception:
            values = descriptor.get("values", []) if isinstance(descriptor, dict) else descriptor
            sizes[name] = len(values)
    return sizes


def _fill_entries(dest: Layout) -> list[tuple[Interval, Any]]:
    raw = dest.descriptor.get("fill", [])
    out: list[tuple[Interval, Any]] = []
    for item in raw:
        start, end, word = int(item[0]), int(item[1]), item[2]
        out.append(((start, end), word))
    return out


def _writable_intervals(dest: Layout) -> list[Interval]:
    intervals: list[Interval] = [iv(x) for x in dest.descriptor.get("write_grants", [])]
    intervals.extend(dest.logical_intervals())
    intervals.extend(iv(x[:2]) for x in dest.descriptor.get("fill", []))
    return union_intervals(intervals)


def _blocks_for_interval(interval: Interval, align: int) -> list[int]:
    start, end = interval
    first = (start // align) * align
    last = ((end - 1) // align) * align
    return list(range(first, last + align, align))


def _ownership_report(dest: Layout, sources: list[Layout]) -> dict[str, Any]:
    """Box-cell coverage proof: each target coordinate has exactly one owner."""
    relevant = [s for s in sources if box_intersection(dest.box, s.box) is not None]

    points_per_axis: list[list[int]] = []
    for axis in range(4):
        lo, hi = dest.box[axis]
        points = {lo, hi}
        for source in relevant:
            slo, shi = source.box[axis]
            if lo < slo < hi:
                points.add(slo)
            if lo < shi < hi:
                points.add(shi)
        points_per_axis.append(sorted(points))

    gaps: list[list[list[int]]] = []
    overlaps: list[dict[str, Any]] = []
    owner_ids = sorted(s.id for s in relevant)

    # Membership is constant inside every elementary cell.
    for n_i in range(len(points_per_axis[0]) - 1):
        for c_i in range(len(points_per_axis[1]) - 1):
            for h_i in range(len(points_per_axis[2]) - 1):
                for w_i in range(len(points_per_axis[3]) - 1):
                    cell = (
                        (points_per_axis[0][n_i], points_per_axis[0][n_i + 1]),
                        (points_per_axis[1][c_i], points_per_axis[1][c_i + 1]),
                        (points_per_axis[2][h_i], points_per_axis[2][h_i + 1]),
                        (points_per_axis[3][w_i], points_per_axis[3][w_i + 1]),
                    )
                    probe = tuple(x[0] for x in cell)
                    hits = [s.id for s in relevant if box_contains(s.box, probe)]
                    if len(hits) == 0:
                        gaps.append([list(x) for x in cell])
                    elif len(hits) > 1:
                        overlaps.append({"box": [list(x) for x in cell], "sources": hits})

    return {
        "source_ids": owner_ids,
        "unique_owner": not gaps and not overlaps,
        "gaps": gaps,
        "overlaps": overlaps,
        "target_elements": box_volume(dest.box),
    }


def build_plan(
    record: dict[str, Any],
    tensors: dict[str, Any] | None = None,
    capacity: dict[str, Any] | None = None,
    page_sizes: dict[str, int] | None = None,
) -> HandoffPlan:
    """Validate layouts and construct aligned physical burst blocks."""
    diagnostics: list[Diagnostic] = []
    sources = [Layout(x, "source", i) for i, x in enumerate(record.get("sources", []))]
    destinations = [Layout(x, "destination", i) for i, x in enumerate(record.get("destinations", []))]
    generation = int(record.get("generation", DEFAULT_GENERATION))
    align = int(record.get("burst_align", capacity.get("burst_align", 1) if capacity else 1))
    link_max = int(record.get("link_max", capacity.get("link_max", align) if capacity else align))

    sizes = _page_sizes_from_tensors(tensors)
    if page_sizes:
        sizes.update(page_sizes)

    budget = capacity.get("peak_scratch_bytes") if capacity else None
    budget = int(budget) if budget is not None else None
    scratch_required = align

    if align <= 0:
        diagnostics.append(Diagnostic("bad_alignment", "burst_align 必须为正整数", {"burst_align": align}))
    if link_max <= 0:
        diagnostics.append(Diagnostic("bad_link_limit", "link_max 必须为正整数", {"link_max": link_max}))
    if align > link_max:
        diagnostics.append(
            Diagnostic(
                "burst_larger_than_link",
                "一个对齐突发比单次链路最大载荷还大，无法合法交付",
                {"burst_align": align, "link_max": link_max},
            )
        )
    if budget is not None and align > budget:
        diagnostics.append(
            Diagnostic(
                "scratch_budget_exceeded",
                "最小对齐写缓冲超过工作区内存上限",
                {"burst_align": align, "peak_scratch_bytes": budget},
            )
        )
        scratch_required = align

    # Geometry and page bounds.
    for item in sources + destinations:
        try:
            span = item.geometry_span()
            if any(s < 0 for s in item.global_strides):
                diagnostics.append(
                    Diagnostic("negative_stride_unsupported", "当前流式执行器要求非负步幅", {"id": item.id})
                )
            if item.base < 0:
                diagnostics.append(Diagnostic("bad_base", "base 不能为负", {"id": item.id, "base": item.base}))
            page_size = sizes.get(item.page)
            if page_size is not None and item.base + span > page_size:
                diagnostics.append(
                    Diagnostic(
                        "page_out_of_bounds",
                        f"{item.id} 的物理跨度超出页 {item.page}",
                        {"id": item.id, "page": item.page, "required_end": item.base + span,
                         "page_size": page_size},
                    )
                )
        except HandoffContractError as exc:
            diagnostics.append(Diagnostic("aliasing_layout", str(exc), {"id": item.id}))

    # Logical ownership from source boxes to destination boxes.
    ownership: dict[str, dict[str, Any]] = {}
    for dest in destinations:
        report = _ownership_report(dest, sources)
        ownership[dest.id] = report
        if report["gaps"]:
            diagnostics.append(
                Diagnostic(
                    "uncovered_target",
                    f"目的分片 {dest.id} 存在没有任何源拥有者的逻辑盒",
                    {"destination": dest.id, "gaps": report["gaps"][:16],
                     "gap_count": len(report["gaps"])},
                )
            )
        if report["overlaps"]:
            diagnostics.append(
                Diagnostic(
                    "non_unique_owner",
                    f"目的分片 {dest.id} 的逻辑元素存在多个源拥有者",
                    {"destination": dest.id, "overlaps": report["overlaps"][:16],
                     "overlap_count": len(report["overlaps"])},
                )
            )

    # Destination shards are normally disjoint.  A logical overlap is only a
    # warning: a shared grant and a joint generation can make it a replica.
    for i, a in enumerate(destinations):
        for b in destinations[i + 1:]:
            overlap = box_intersection(a.box, b.box)
            if overlap is not None and box_volume(overlap) > 0:
                diagnostics.append(
                    Diagnostic(
                        "destination_logical_overlap",
                        f"目的分片 {a.id} 与 {b.id} 覆盖同一逻辑坐标；必须使用统一代次和联合突发",
                        {"destinations": [a.id, b.id], "box": [list(x) for x in overlap]},
                        severity="warning",
                    )
                )

    blocks: dict[tuple[str, int], BlockPlan] = {}

    def get_block(page: str, start: int) -> BlockPlan:
        key = (page, start)
        if key not in blocks:
            blocks[key] = BlockPlan(page=page, start=start, end=start + align)
        return blocks[key]

    # Build aligned burst blocks from exact logical runs and fill windows.
    for dest in destinations:
        writable = _writable_intervals(dest)
        required_intervals: list[Interval] = list(dest.logical_intervals())
        fill_entries = _fill_entries(dest)
        required_intervals.extend(iv(x[0]) for x in fill_entries)

        dest_blocks: set[int] = set()
        for interval in required_intervals:
            if interval[1] <= interval[0]:
                continue
            if align > 0:
                dest_blocks.update(_blocks_for_interval(interval, align))

        for start in sorted(dest_blocks):
            end = start + align
            block = get_block(dest.page, start)
            block.destinations.add(dest.id)
            for range_, word in fill_entries:
                piece = iv_intersection(range_, (start, end))
                if piece is not None:
                    block.fill_ranges.setdefault(dest.id, []).append(piece)

            # The complete physical burst is written, not just logical bytes.
            unwritable = subtract_intervals((start, end), writable)
            if unwritable:
                block.errors.append(
                    Diagnostic(
                        "write_not_authorized",
                        f"目的 {dest.id} 的对齐突发 [{start},{end}) 含未授权可写字节",
                        {"destination": dest.id, "range": [start, end],
                         "unwritable": [list(x) for x in unwritable]},
                    )
                )

            page_size = sizes.get(dest.page)
            if page_size is not None and end > page_size:
                block.errors.append(
                    Diagnostic(
                        "burst_out_of_page",
                        f"对齐突发 [{start},{end}) 超出页 {dest.page} 长度 {page_size}",
                        {"destination": dest.id, "range": [start, end], "page_size": page_size},
                    )
                )

    # Shared physical burst authorization.
    shared_grants = record.get("shared_grants", [])
    for block in blocks.values():
        if len(block.destinations) <= 1:
            continue
        allowed = False
        matched: dict[str, Any] | None = None
        for grant in shared_grants:
            if str(grant.get("page")) != block.page:
                continue
            grant_range = iv(grant["range"])
            grant_dests = set(grant.get("destinations", []))
            if iv_contains(grant_range, (block.start, block.end)) and block.destinations <= grant_dests:
                allowed = True
                matched = grant
                break
        if not allowed:
            block.errors.append(
                Diagnostic(
                    "shared_grant_missing",
                    "两个目的分片需要写同一个对齐突发，但 shared_grants 未授权完整共用写区",
                    {"page": block.page, "range": [block.start, block.end],
                     "destinations": sorted(block.destinations),
                     "available_shared_grants": shared_grants},
                )
            )
        else:
            block.source_ids  # explicit no-op for readability

    if capacity:
        dense_bytes = None
        if "global" in capacity:
            dense_bytes = math.prod(int(x) for x in capacity["global"]) * ELEMENT_BYTES
        # This planner allocates at most one aligned packet plus small metadata.
        if budget is not None and scratch_required > budget:
            diagnostics.append(
                Diagnostic(
                    "capacity_impossible",
                    "容量资料中的内存上限无法容纳最小流式交接缓冲",
                    {"scratch_required": scratch_required, "peak_scratch_bytes": budget,
                     "dense_bytes": dense_bytes},
                )
            )

    plan = HandoffPlan(
        record=record,
        generation=generation,
        sources=sources,
        destinations=destinations,
        page_sizes=sizes,
        burst_align=align,
        link_max=link_max,
        capacity=capacity,
        blocks=blocks,
        ownership=ownership,
        diagnostics=diagnostics,
        scratch_budget=budget,
        scratch_required=scratch_required,
    )
    populate_block_source_ids(plan)
    return plan


def stream_copy_intersection(
    source: Layout,
    destination: Layout,
    source_page: np.ndarray,
    destination_page: np.ndarray,
    intersection: Box,
    scratch_budget: int | None = None,
) -> int:
    """Copy one box intersection through bounded NumPy strided views.

    No dense NCHW or NHWC tensor is allocated.  Chunking only bounds NumPy's
    internal iterator/buffer scratch; the source and destination views
    themselves are zero-copy.
    """
    source_view = source.byte_element_view(source_page)
    destination_view = destination.byte_element_view(destination_page)
    shape = tuple(hi - lo for lo, hi in intersection)
    element_count = box_volume(intersection)
    if element_count == 0:
        return 0

    # Conservative factor: two chunk views plus an iterator temporary.
    budget = scratch_budget if scratch_budget is not None else 1 << 20
    max_elements = max(1, int(budget) // (ELEMENT_BYTES * 4))

    chunk_lengths: list[int] = []
    for axis, size in enumerate(shape):
        later = math.prod(shape[axis + 1 :]) if axis + 1 < len(shape) else 1
        chunk_lengths.append(max(1, min(size, max_elements // max(1, later))))

    starts = [range(0, size, chunk) for size, chunk in zip(shape, chunk_lengths)]
    copied = 0
    for offsets in _product_ranges(starts):
        source_slices = []
        destination_slices = []
        for axis in range(4):
            length = min(chunk_lengths[axis], shape[axis] - offsets[axis])
            source_start = intersection[axis][0] - source.box[axis][0] + offsets[axis]
            destination_start = intersection[axis][0] - destination.box[axis][0] + offsets[axis]
            source_slices.append(slice(source_start, source_start + length))
            destination_slices.append(slice(destination_start, destination_start + length))
        source_slices.append(slice(None))
        destination_slices.append(slice(None))
        np.copyto(
            destination_view[tuple(destination_slices)],
            source_view[tuple(source_slices)],
        )
        copied += 1
    return copied


# ---------------------------------------------------------------------------
# Fenced event session and read views
# ---------------------------------------------------------------------------


@dataclass
class PacketRecord:
    seq: int
    key: tuple[str, int]
    generation: int
    digest: str
    payload: bytes
    destinations: list[str]
    outcome: str

    def as_dict(self, include_payload: bool = False) -> dict[str, Any]:
        data = {
            "seq": self.seq,
            "page": self.key[0],
            "range": [self.key[1], self.key[1] + 0],  # replaced below by session/end
            "generation": self.generation,
            "digest": self.digest,
            "destinations": self.destinations,
            "outcome": self.outcome,
        }
        if include_payload:
            data["payload_hex"] = self.payload.hex()
        return data


@dataclass
class ReadView:
    destination_id: str
    page: str
    status: str
    generation: int | None
    shape: tuple[int, ...]
    layout: str
    strides: tuple[int, ...]
    requested_generation: int | None = None
    missing: list[dict[str, Any]] = field(default_factory=list)
    diagnostic: dict[str, Any] | None = None
    byte_ranges: list[list[int]] = field(default_factory=list)
    array: np.ndarray | None = field(default=None, repr=False)
    logical_array: np.ndarray | None = field(default=None, repr=False)

    @property
    def readable(self) -> bool:
        return self.status == "readable" and self.array is not None and self.generation is not None

    @property
    def ready(self) -> bool:
        return self.readable

    @property
    def missing_ranges(self) -> list[dict[str, Any]]:
        return self.missing

    @property
    def missing_boxes(self) -> list[list[list[int]]]:
        boxes: list[list[list[int]]] = []
        for item in self.missing:
            boxes.extend(item.get("boxes", []))
        return boxes

    @property
    def diagnostics(self) -> dict[str, Any] | None:
        return self.diagnostic

    def numpy(self, global_order: bool = False) -> np.ndarray:
        if not self.readable or self.array is None:
            raise HandoffContractError(f"{self.destination_id} 当前不可读: {self.status}")
        if global_order:
            if self.logical_array is None:
                raise HandoffContractError("全局逻辑视图不可用")
            return self.logical_array
        return self.array

    def __getitem__(self, key: str) -> Any:
        if key in {"id", "destination_id"}:
            return self.destination_id
        if hasattr(self, key):
            return getattr(self, key)
        return self.as_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def as_dict(self, include_values: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "destination": self.destination_id,
            "page": self.page,
            "status": self.status,
            "generation": self.generation,
            "requested_generation": self.requested_generation,
            "shape": list(self.shape),
            "global_shape": list(self.logical_array.shape) if self.logical_array is not None else None,
            "dtype": "uint16-le",
            "layout": self.layout,
            "strides": list(self.strides),
            "missing": self.missing,
            "diagnostic": self.diagnostic,
            "readable_byte_ranges": self.byte_ranges,
            "readable": self.readable,
            "ready": self.readable,
            "complete": self.readable,
            "ok": self.readable,
        }
        if include_values and self.array is not None:
            out["values_u16"] = self.array.tolist()
        return out


class HandoffSession:
    """Replay deliveries/releases and own the simulated destination pages."""

    def __init__(
        self,
        record: dict[str, Any],
        tensors: dict[str, Any] | None = None,
        capacity: dict[str, Any] | None = None,
        *,
        pages: dict[str, np.ndarray] | None = None,
        strict_plan: bool = False,
        retain_payloads: bool | None = None,
        packet_log_limit: int = 4096,
    ) -> None:
        # Allow passing either records["records"] or the whole capture envelope.
        if isinstance(record, dict) and "sources" not in record and isinstance(record.get("records"), dict):
            envelope = record
            tensors = tensors if tensors is not None else envelope.get("tensors", {})
            record = envelope["records"]
        self.record = record
        self.tensors = tensors or {}
        self.plan = build_plan(record, self.tensors, capacity)
        self.generation = self.plan.generation
        self.strict_plan = strict_plan

        self.pages: dict[str, np.ndarray] = {}
        if pages:
            for name, page in pages.items():
                self.pages[name] = np.asarray(page, dtype=np.uint8).copy()
        else:
            for dest in self.plan.destinations:
                if dest.page in self.pages:
                    continue
                tensor_name = dest.page
                if tensor_name not in self.tensors:
                    raise HandoffContractError(f"缺少目的页张量 {tensor_name!r}")
                self.pages[dest.page] = tensor_to_page(self.tensors[tensor_name])

        # Source pages are caller-owned read-only inputs.  Events may also
        # carry generation-specific payloads for retry/abort simulations.
        self.source_pages: dict[str, np.ndarray] = {}
        for source in self.plan.sources:
            descriptor = self.tensors[source.tensor]
            self.source_pages[source.id] = tensor_to_page(descriptor)
        self.source_versions: dict[tuple[str, int], np.ndarray] = {}

        self.delivered: dict[str, set[int]] = {s.id: set() for s in self.plan.sources}
        self.active_reads: dict[str, dict[str, Any]] = {}
        for read in record.get("live_reads", []):
            self.active_reads[str(read["id"])] = dict(read)
        self.released_reads: set[str] = set()

        self.block_state: dict[tuple[str, int], dict[str, Any]] = {}
        self.packets: list[PacketRecord] = []
        self.packet_log_limit = packet_log_limit
        self.dropped_packet_records = 0
        total_page_bytes = sum(page.nbytes for page in self.pages.values())
        if retain_payloads is None:
            # Keep retry bytes for unit/session-sized captures, but never turn
            # a large transfer log into a second full activation copy.
            retain_payloads = total_page_bytes <= 4096
        self.retain_payloads = retain_payloads
        self.event_log: list[dict[str, Any]] = []
        self.aborted_sources: set[tuple[str, int]] = set()
        self._seq = 0
        self._requirements_built = False

        if strict_plan and not self.plan.feasible:
            raise HandoffContractError("; ".join(d["message"] for d in self.plan.diagnostic_dicts()))

    # ----- event handling ---------------------------------------------------

    def _source_page(self, source_id: str, generation: int) -> np.ndarray:
        version = self.source_versions.get((source_id, generation))
        if version is not None:
            return version
        return self.source_pages[source_id]

    def deliver_source(
        self,
        source_id: str,
        generation: int | None = None,
        data: bytes | bytearray | np.ndarray | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        generation = self.generation if generation is None else int(generation)
        known = {s.id for s in self.plan.sources}
        if source_id not in known:
            outcome = {"event": "source", "source": source_id, "generation": generation,
                       "outcome": "unknown_source"}
            self.event_log.append(outcome)
            return outcome
        if data is not None:
            if isinstance(data, (bytes, bytearray)):
                self.source_versions[(source_id, generation)] = np.frombuffer(bytes(data), dtype=np.uint8)
            else:
                self.source_versions[(source_id, generation)] = tensor_to_page(data)
        key = (source_id, generation)
        if key in self.aborted_sources:
            # Redelivery after abort is a new attempt only if caller clears it.
            self.aborted_sources.remove(key)
        if generation in self.delivered[source_id]:
            outcome = {"event": "source", "source": source_id, "generation": generation,
                       "outcome": "duplicate_ignored"}
        else:
            self.delivered[source_id].add(generation)
            outcome = {"event": "source", "source": source_id, "generation": generation,
                       "outcome": "accepted", "current_generation": self.generation,
                       "stale": generation != self.generation}
        self.event_log.append(outcome)
        self.commit_ready(self.generation)
        return outcome

    def release(self, reader_id: str) -> dict[str, Any]:
        existed = self.active_reads.pop(reader_id, None)
        self.released_reads.add(reader_id)
        outcome = {"event": "release", "id": reader_id,
                   "outcome": "released" if existed is not None else "duplicate_or_unknown"}
        self.event_log.append(outcome)
        self.commit_ready(self.generation)
        return outcome

    def abort(self, source_id: str | None = None, generation: int | None = None,
              reason: str = "aborted") -> dict[str, Any]:
        if source_id is not None:
            gen = self.generation if generation is None else generation
            self.delivered.get(source_id, set()).discard(gen)
            self.aborted_sources.add((source_id, gen))
        outcome = {"event": "abort", "source": source_id, "generation": generation,
                   "outcome": reason}
        self.event_log.append(outcome)
        return outcome

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if "release" in event:
            return self.release(str(event["release"]))
        if "source" in event:
            return self.deliver_source(
                str(event["source"]),
                event.get("generation"),
                event.get("data", event.get("payload")),
            )
        if "abort" in event:
            target = event["abort"]
            if isinstance(target, dict):
                return self.abort(target.get("source"), target.get("generation"), "aborted")
            return self.abort(str(target), event.get("generation"), "aborted")
        if event.get("type") == "fence":
            return {"event": "fence", **self.fence(str(event["destination"]), event.get("generation"))}
        outcome = {"event": "unrecognized", "raw": event, "outcome": "ignored"}
        self.event_log.append(outcome)
        return outcome

    def run_events(self, events: Sequence[dict[str, Any]] | None = None) -> "HandoffSession":
        for event in events if events is not None else self.record.get("delivery_events", []):
            self.handle_event(event)
        self.commit_ready(self.generation)
        return self

    # ----- physical block protocol -----------------------------------------

    def _build_requirements(self) -> None:
        if self._requirements_built:
            return
        populate_block_source_ids(self.plan)
        self._requirements_built = True

    def _active_live_overlap(self, page: str, interval: Interval) -> dict[str, Any] | None:
        for read in self.active_reads.values():
            if str(read.get("page")) != page:
                continue
            read_range = iv(read["range"])
            if iv_overlap(read_range, interval):
                return read
        return None

    def _global_layout_error(self) -> Diagnostic | None:
        return next((d for d in self.plan.diagnostics if d.severity == "error"), None)

    def _first_error_dict(self) -> dict[str, Any] | None:
        return next(
            (d for d in self.plan.diagnostic_dicts() if d.get("severity", "error") == "error"),
            None,
        )

    def _block_ready_reason(self, block: BlockPlan, generation: int) -> str | None:
        if self._global_layout_error() is not None:
            return "layout_unlandable"
        if block.errors:
            return "layout_unlandable"
        live = self._active_live_overlap(block.page, (block.start, block.end))
        if live is not None:
            return "live_read_busy"
        for source_id in block.source_ids:
            if generation not in self.delivered.get(source_id, set()):
                return "missing_source_generation"
        if block.start >= block.end:
            return "bad_block"
        return None

    def _fill_bytes(self, word: Any, length: int, width: int | None = None) -> bytes:
        # Default to the contract-wide little-endian uint16 fill word.  A
        # record/descriptor may explicitly provide fill_width=1 for byte fills.
        if isinstance(word, (list, tuple, bytes)):
            pattern = bytes(word)
        else:
            value = int(word)
            if width is None:
                width = int(self.record.get("fill_width", self.record.get("word_bytes", ELEMENT_BYTES)))
            pattern = value.to_bytes(int(width), "little", signed=False)
        repeats = (length + len(pattern) - 1) // len(pattern)
        return (pattern * repeats)[:length]

    def build_joint_packet(self, key: tuple[str, int], generation: int) -> bytes:
        """Build a deterministic full burst containing every logical owner."""
        self._build_requirements()
        block = self.plan.blocks[key]
        page = self.pages[block.page]
        packet = page[block.start:block.end].copy()

        sources_by_id = {s.id: s for s in self.plan.sources}
        dests_by_id = {d.id: d for d in self.plan.destinations}

        for dest_id in block.destinations:
            dest = dests_by_id[dest_id]
            for source in self.plan.sources:
                inter = box_intersection(dest.box, source.box)
                if inter is None:
                    continue
                for coord in iter_coords_in_byte_range(dest, (block.start, block.end), inter):
                    d_start, _ = dest.element_interval(coord)
                    s_start, _ = source.element_interval(coord)
                    source_page = self._source_page(source.id, generation)
                    for byte_index in range(ELEMENT_BYTES):
                        page_offset = d_start + byte_index
                        if block.start <= page_offset < block.end:
                            packet[page_offset - block.start] = source_page[s_start + byte_index]

            for fill_range in block.fill_ranges.get(dest_id, []):
                local_start = fill_range[0] - block.start
                length = fill_range[1] - fill_range[0]
                # Find the original word from the descriptor.
                word = 0
                for raw_range, raw_word in _fill_entries(dest):
                    if raw_range[0] <= fill_range[0] and fill_range[1] <= raw_range[1]:
                        word = raw_word
                        break
                packet[local_start:local_start + length] = np.frombuffer(
                    self._fill_bytes(word, length, dest.descriptor.get("fill_width")),
                    dtype=np.uint8,
                )

        return bytes(packet)

    def receive_packet(
        self,
        key: tuple[str, int],
        generation: int,
        payload: bytes,
        destinations: list[str] | None = None,
    ) -> dict[str, Any]:
        if key not in self.plan.blocks:
            return {"outcome": "unknown_block", "key": key}
        block = self.plan.blocks[key]
        interval = (block.start, block.end)
        if len(payload) != self.plan.burst_align:
            return {"outcome": "bad_payload_length", "expected": self.plan.burst_align,
                    "actual": len(payload)}
        if len(payload) > self.plan.link_max:
            return {"outcome": "denied_link_limit", "limit": self.plan.link_max,
                    "actual": len(payload)}
        if key[1] % self.plan.burst_align != 0:
            return {"outcome": "misaligned_block_start", "start": key[1],
                    "alignment": self.plan.burst_align}
        if block.errors:
            return {"outcome": "denied_layout", "errors": [e.as_dict() for e in block.errors]}
        live = self._active_live_overlap(block.page, interval)
        if live is not None:
            return {"outcome": "live_read_busy", "reader": live.get("id"), "range": live.get("range")}

        digest = hashlib.sha256(payload).hexdigest()
        previous = self.block_state.get(key)
        outcome = "applied"
        if previous is not None:
            if generation < previous["generation"]:
                outcome = "rejected_stale_generation"
            elif generation == previous["generation"] and digest != previous["digest"]:
                outcome = "rejected_version_conflict"
            elif generation == previous["generation"]:
                outcome = "duplicate_idempotent"
            else:
                outcome = "applied_new_generation"

        self._seq += 1
        if len(self.packets) < self.packet_log_limit:
            record = PacketRecord(
                seq=self._seq,
                key=key,
                generation=generation,
                digest=digest,
                payload=bytes(payload) if self.retain_payloads else b"",
                destinations=destinations or sorted(block.destinations),
                outcome=outcome,
            )
            self.packets.append(record)
        else:
            self.dropped_packet_records += 1

        if outcome.startswith("applied"):
            self.pages[block.page][block.start:block.end] = np.frombuffer(payload, dtype=np.uint8)
            self.block_state[key] = {"generation": generation, "digest": digest, "seq": self._seq}
        return {"outcome": outcome, "seq": self._seq, "digest": digest, "logged": len(self.packets) > 0 and self.packets[-1].seq == self._seq}

    def _vectorized_apply_contract(self, generation: int) -> None:
        """Fast O(activation) path used once every physical burst is legal."""
        for dest in self.plan.destinations:
            page = self.pages[dest.page]
            for source in self.plan.sources:
                if generation not in self.delivered.get(source.id, set()):
                    continue
                intersection = box_intersection(dest.box, source.box)
                if intersection is None:
                    continue
                stream_copy_intersection(
                    source,
                    dest,
                    self._source_page(source.id, generation),
                    page,
                    intersection,
                    self.plan.scratch_budget,
                )

            for fill_range, word in _fill_entries(dest):
                payload = np.frombuffer(
                    self._fill_bytes(word, fill_range[1] - fill_range[0], dest.descriptor.get("fill_width")),
                    dtype=np.uint8,
                )
                page[fill_range[0] : fill_range[1]] = payload

    def commit_ready(self, generation: int | None = None) -> list[dict[str, Any]]:
        """Construct and deliver all currently legal joint bursts."""
        self._build_requirements()
        generation = self.generation if generation is None else int(generation)
        results: list[dict[str, Any]] = []

        pending: list[BlockPlan] = []
        all_blocks_ready = True
        for key in sorted(self.plan.blocks):
            block = self.plan.blocks[key]
            reason = self._block_ready_reason(block, generation)
            if reason is not None:
                all_blocks_ready = False
                continue
            previous = self.block_state.get(key)
            if previous is not None and previous["generation"] >= generation:
                continue
            pending.append(block)

        if not pending:
            return results

        # In the normal complete-capture case, copy all strided intersections
        # with NumPy and then emit each aligned burst as a page-slice payload.
        if self.plan.feasible and all_blocks_ready:
            self._vectorized_apply_contract(generation)
            for block in pending:
                payload = bytes(self.pages[block.page][block.start : block.end])
                result = self.receive_packet((block.page, block.start), generation, payload)
                result.update({"block": {"page": block.page, "range": [block.start, block.end]}})
                results.append(result)
            return results

        # Partial/out-of-order case: build only ready joint bursts.  The range
        # generator visits only coordinates intersecting each burst.
        for block in pending:
            payload = self.build_joint_packet((block.page, block.start), generation)
            result = self.receive_packet((block.page, block.start), generation, payload)
            result.update({"block": {"page": block.page, "range": [block.start, block.end]}})
            results.append(result)
        return results

    def retry_packet(self, seq: int) -> dict[str, Any]:
        packet = next((p for p in self.packets if p.seq == seq), None)
        if packet is None:
            return {"outcome": "unknown_or_truncated_packet", "seq": seq}
        if not packet.payload:
            return {"outcome": "retry_payload_not_retained", "seq": seq}
        return self.receive_packet(packet.key, packet.generation, packet.payload, packet.destinations)

    def fence(self, destination_id: str, generation: int | None = None) -> dict[str, Any]:
        generation = self.generation if generation is None else int(generation)
        view = self.read_view(destination_id)
        if view.status != "readable":
            return {"destination": destination_id, "generation": generation,
                    "outcome": "not_complete", "missing": view.missing,
                    "diagnostic": view.diagnostic}
        if view.generation != generation:
            return {"destination": destination_id, "requested_generation": generation,
                    "effective_generation": view.generation, "outcome": "stale_generation"}
        return {"destination": destination_id, "generation": generation, "outcome": "fenced"}

    # ----- status / read views ---------------------------------------------

    def _coord_reason(self, dest: Layout, coord: tuple[int, ...], generation: int) -> str | None:
        if self._global_layout_error() is not None:
            return "layout_unlandable"
        hits = [s.id for s in self.plan.sources if box_contains(s.box, coord)]
        if len(hits) == 0:
            return "uncovered_target"
        if len(hits) > 1:
            return "non_unique_owner"
        source_id = hits[0]
        if generation not in self.delivered.get(source_id, set()):
            return f"missing_source_generation:{source_id}:{generation}"

        start, end = dest.element_interval(coord)
        for block_start in _blocks_for_interval((start, end), self.plan.burst_align):
            key = (dest.page, block_start)
            block = self.plan.blocks.get(key)
            if block is None:
                return "layout_unlandable"
            ready_reason = self._block_ready_reason(block, generation)
            if ready_reason == "live_read_busy":
                return "live_read_busy"
            if ready_reason == "missing_source_generation":
                return "blocked_shared_burst"
            if ready_reason in {"layout_unlandable", "scratch_budget_exceeded", "burst_larger_than_link"}:
                return "layout_unlandable"
            state = self.block_state.get(key)
            if state is None or state["generation"] != generation:
                return "write_not_effective"
        return None

    def read_view(self, destination_id: str) -> ReadView:
        dest = next((d for d in self.plan.destinations if d.id == destination_id), None)
        if dest is None:
            raise HandoffContractError(f"未知目的分片 {destination_id!r}")
        generation = self.generation
        self._build_requirements()

        fill_problems: list[dict[str, Any]] = []
        for raw_range, _word in _fill_entries(dest):
            for block_start in _blocks_for_interval(raw_range, self.plan.burst_align):
                key = (dest.page, block_start)
                state = self.block_state.get(key)
                block = self.plan.blocks.get(key)
                live = self._active_live_overlap(dest.page, raw_range)
                if block is not None and block.errors:
                    fill_problems.append({"range": list(raw_range), "reason": "layout_unlandable"})
                elif live is not None:
                    fill_problems.append({"range": list(raw_range), "reason": "live_read_busy"})
                elif state is None or state["generation"] != generation:
                    fill_problems.append({"range": list(raw_range), "reason": "fill_not_established"})

        dest_blocks = [block for block in self.plan.blocks.values() if dest.id in block.destinations]
        all_blocks_effective = bool(dest_blocks) and all(
            self.block_state.get(block.key, {}).get("generation") == generation
            for block in dest_blocks
        )
        if self.plan.feasible and not fill_problems and all_blocks_effective:
            array = dest.uint16_view(self.pages[dest.page], global_order=False)
            logical_array = dest.uint16_view(self.pages[dest.page], global_order=True)
            contract_ranges = list(dest.logical_intervals())
            contract_ranges.extend(iv(x[0]) for x in _fill_entries(dest))
            byte_ranges = [list(x) for x in union_intervals(contract_ranges)]
            return ReadView(
                destination_id=dest.id,
                page=dest.page,
                status="readable",
                generation=generation,
                requested_generation=generation,
                shape=tuple(dest.sizes_global[AXIS_INDEX[axis]] for axis in dest.layout),
                layout=dest.layout,
                strides=dest.local_strides,
                byte_ranges=byte_ranges,
                array=array,
                logical_array=logical_array,
            )

        bad: dict[str, set[tuple[int, ...]]] = {}
        for coord in iter_box_coords(dest.box):
            reason = self._coord_reason(dest, coord, generation)
            if reason is not None:
                bad.setdefault(reason, set()).add(coord)

        missing = boxes_from_w_runs(bad)

        diagnostic = None
        status = "readable"
        if any(r["reason"] in {"uncovered_target", "non_unique_owner", "layout_unlandable"} for r in missing):
            status = "unlandable"
            diagnostic = self._first_error_dict() or {
                "code": "layout_unlandable",
                "message": "源/目的版图无法唯一、合法地共同落地",
            }
        elif missing or fill_problems:
            status = "incomplete"
            if any("live_read_busy" in r["reason"] for r in missing) or any(
                p["reason"] == "live_read_busy" for p in fill_problems
            ):
                status = "blocked_by_live_read"
            if not self.plan.feasible:
                status = "unlandable"
                diagnostic = self._first_error_dict()
        elif not self.plan.feasible:
            status = "unlandable"
            diagnostic = self._first_error_dict()

        array = None
        logical_array = None
        byte_ranges: list[list[int]] = []
        effective_generation: int | None = generation if status == "readable" else None
        if status == "readable":
            array = dest.uint16_view(self.pages[dest.page], global_order=False)
            logical_array = dest.uint16_view(self.pages[dest.page], global_order=True)
            contract_ranges = list(dest.logical_intervals())
            contract_ranges.extend(iv(x[0]) for x in _fill_entries(dest))
            byte_ranges = [list(x) for x in union_intervals(contract_ranges)]

        return ReadView(
            destination_id=dest.id,
            page=dest.page,
            status=status,
            generation=effective_generation,
            requested_generation=generation,
            shape=tuple(dest.sizes_global[AXIS_INDEX[axis]] for axis in dest.layout),
            layout=dest.layout,
            strides=dest.local_strides,
            missing=missing + ([{"reason": "fill", "ranges": fill_problems}] if fill_problems else []),
            diagnostic=diagnostic,
            byte_ranges=byte_ranges,
            array=array,
            logical_array=logical_array if status == "readable" else None,
        )

    def read_views(self) -> dict[str, ReadView]:
        return {d.id: self.read_view(d.id) for d in self.plan.destinations}

    def result(self, include_memory: bool = False, include_packets: bool = False) -> "HandoffResult":
        views = self.read_views()
        ok = all(v.readable for v in views.values()) and self.plan.feasible
        # Do not expand a large page into Python ints unless a caller explicitly
        # asks for JSON values.  ``pages`` remains the authoritative page.
        memory: dict[str, list[int] | str] = {}
        if include_memory:
            for name, page in self.pages.items():
                memory[name] = page.astype(np.uint8).tolist()
        return HandoffResult(
            ok=ok,
            generation=self.generation if ok else None,
            plan=self.plan,
            views=views,
            event_log=self.event_log,
            packets=self.packets,
            memory=memory,
            pages=self.pages,
            include_packet_payload=include_packets,
            dropped_packet_records=self.dropped_packet_records,
        )


@dataclass
class HandoffResult:
    ok: bool
    generation: int | None
    plan: HandoffPlan
    views: dict[str, ReadView]
    event_log: list[dict[str, Any]]
    packets: list[PacketRecord]
    memory: dict[str, list[int] | str]
    pages: dict[str, np.ndarray]
    include_packet_payload: bool = False
    dropped_packet_records: int = 0

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        return self.plan.diagnostic_dicts()

    @property
    def requested_generation(self) -> int:
        return self.plan.generation

    def numpy_view(self, destination_id: str) -> np.ndarray:
        return self.views[destination_id].numpy()

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except (KeyError, AttributeError):
            return default

    def to_dict(
        self,
        *,
        include_values: bool = False,
        include_memory_hex: bool | None = None,
        include_memory_values: bool = False,
    ) -> dict[str, Any]:
        total_bytes = sum(page.nbytes for page in self.pages.values())
        if include_memory_hex is None:
            include_memory_hex = total_bytes <= 4096
        pages_hex = (
            {name: bytes(page).hex() for name, page in self.pages.items()}
            if include_memory_hex
            else {}
        )
        if include_memory_values:
            memory_values = {name: page.astype(np.uint8).tolist() for name, page in self.pages.items()}
        else:
            memory_values = {}
        return {
            "ok": self.ok,
            "generation": self.generation,
            "plan": self.plan.as_dict(),
            "read_views": {k: v.as_dict(include_values=include_values) for k, v in self.views.items()},
            "events": self.event_log,
            "dropped_packet_records": self.dropped_packet_records,
            "writes": [
                {
                    "seq": p.seq,
                    "page": p.key[0],
                    "block_start": p.key[1],
                    "block_end": p.key[1] + self.plan.burst_align,
                    "generation": p.generation,
                    "digest": p.digest,
                    "destinations": p.destinations,
                    "outcome": p.outcome,
                    **({"payload_hex": p.payload.hex()} if self.include_packet_payload else {}),
                }
                for p in self.packets
            ],
            "memory_hex": pages_hex,
            "memory_included": include_memory_values,
            "memory_values": memory_values,
            "page_bytes_total": total_bytes,
        }


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------


def execute_handoff(
    record: dict[str, Any],
    tensors: dict[str, Any] | None = None,
    capacity: dict[str, Any] | None = None,
    *,
    pages: dict[str, np.ndarray] | None = None,
    replay_events: bool = True,
    **session_kwargs: Any,
) -> HandoffResult:
    """Run one source-shard to destination-read-view handoff."""
    if isinstance(record, dict) and "sources" not in record and isinstance(record.get("records"), dict):
        envelope = record
        tensors = tensors if tensors is not None else envelope.get("tensors")
        record = envelope["records"]
    session = HandoffSession(record, tensors, capacity, pages=pages, **session_kwargs)
    if replay_events:
        session.run_events()
    else:
        session.commit_ready(record.get("generation"))
    return session.result()


# Common aliases for callers/tests that name the operation differently.
run_handoff = execute_handoff
handoff = execute_handoff
transfer = execute_handoff
solve = execute_handoff
plan_handoff = build_plan
verify_layouts = build_plan
validate_handoff = build_plan
HandoffPlanner = HandoffSession
TensorHandoff = HandoffSession
TensorHandoffPlanner = HandoffSession


def run_capture(path: str | Path, *, packet_payloads: bool = False) -> HandoffResult:
    record, tensors, capacity = load_capture(path)
    result = execute_handoff(record, tensors, capacity)
    result.include_packet_payload = packet_payloads
    return result


def load_capture(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    root = Path(path)
    envelope = json.loads(root.read_text(encoding="utf-8"))
    record = envelope["records"]
    tensors = envelope.get("tensors", {})
    capacity_path = root.parent / "capacity.json"
    capacity = json.loads(capacity_path.read_text(encoding="utf-8")) if capacity_path.exists() else None
    return record, tensors, capacity


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="执行带代次围栏的异构张量转置交接")
    default_capture = Path(__file__).resolve().parent / "data" / "capture.json"
    parser.add_argument("capture", nargs="?", default=str(default_capture), help="capture.json 路径")
    parser.add_argument("--capacity", default=None, help="可选 capacity.json；默认取同目录文件")
    parser.add_argument("--packet-payloads", action="store_true", help="在 JSON 中包含突发载荷十六进制")
    parser.add_argument("--values", action="store_true", help="包含目的 uint16 视图数值")
    parser.add_argument("--memory-values", action="store_true", help="包含完整页字节数组（大版图时很大）")
    parser.add_argument("--memory-hex", action="store_true", help="强制包含页十六进制")
    args = parser.parse_args(argv)

    record, tensors, capacity = load_capture(args.capture)
    if args.capacity:
        capacity = json.loads(Path(args.capacity).read_text(encoding="utf-8"))
    result = execute_handoff(record, tensors, capacity)
    result.include_packet_payload = args.packet_payloads
    payload = result.to_dict(
        include_values=args.values,
        include_memory_hex=True if args.memory_hex else None,
        include_memory_values=args.memory_values,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
