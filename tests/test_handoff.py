"""Tests for fenced heterogeneous layout handoff."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from handoff import (
    HandoffSession,
    build_plan,
    execute_handoff,
    iter_coords_in_byte_range,
    load_capture,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class SampleCaptureHandoffTests(unittest.TestCase):
    def setUp(self):
        self.record, self.tensors, self.capacity = load_capture(DATA / "capture.json")

    def test_plan_proves_unique_owners_and_shared_burst(self):
        plan = build_plan(self.record, self.tensors, self.capacity)
        self.assertTrue(plan.feasible, [d.as_dict() for d in plan.diagnostics])
        for report in plan.ownership.values():
            self.assertTrue(report["unique_owner"])
            self.assertEqual(report["source_ids"], ["s0", "s1"])

        shared = plan.blocks[("memory", 16)]
        self.assertEqual(shared.end, 32)
        self.assertEqual(shared.destinations, {"d0", "d1"})
        self.assertEqual(shared.source_ids, {"s0", "s1"})
        self.assertFalse(shared.errors)

    def test_event_log_reaches_gen7_read_views(self):
        session = HandoffSession(self.record, self.tensors, self.capacity)
        initial = session.pages["memory"].copy()

        session.deliver_source("s1", 7)
        session.deliver_source("s1", 7)
        # The active old reader owns [0, 16); early delivery must not change it.
        self.assertTrue(np.array_equal(session.pages["memory"][:16], initial[:16]))
        self.assertEqual(session.read_view("d0").status, "blocked_by_live_read")

        session.deliver_source("s0", 6)
        session.release("old-d0")
        self.assertTrue(np.array_equal(session.pages["memory"][:16], initial[:16]))
        self.assertNotIn(7, session.delivered["s0"])

        session.deliver_source("s0", 7)
        views = session.read_views()
        self.assertEqual(views["d0"].status, "readable")
        self.assertEqual(views["d1"].status, "readable")
        self.assertEqual(views["d0"].generation, 7)
        self.assertEqual(views["d1"].generation, 7)

        result = session.result()
        self.assertTrue(result.ok)
        self.assertEqual(result.numpy_view("d0").shape, (1, 1, 3, 4))
        self.assertEqual(result.numpy_view("d1").shape, (1, 1, 3, 4))

    def test_final_bytes_are_strided_transpose_not_contiguous_pack(self):
        result = execute_handoff(self.record, self.tensors, self.capacity)
        memory = bytes(result.pages["memory"])

        # Independently calculate every expected byte from logical coordinates.
        sources = {s.id: s for s in result.plan.sources}
        dests = {d.id: d for d in result.plan.destinations}
        expected = bytearray(memory)
        for dest in dests.values():
            for source in sources.values():
                from handoff import box_intersection, iter_box_coords

                inter = box_intersection(dest.box, source.box)
                if inter is None:
                    continue
                source_page = np.asarray(self.tensors[source.tensor]["values"], dtype=np.uint8)
                for coord in iter_box_coords(inter):
                    ds = dest.element_interval(coord)[0]
                    ss = source.element_interval(coord)[0]
                    expected[ds : ds + 2] = bytes(source_page[ss : ss + 2])

        self.assertEqual(memory[:48], bytes(expected[:48]))
        # Contract fill word 255 is little-endian uint16: ff 00 repeated.
        self.assertEqual(memory[48:56], bytes.fromhex("ff00") * 4)
        # Alignment tail has no fill contract and must preserve old page bytes.
        self.assertEqual(memory[56:64], b"\xd2" * 8)

        # The old wrong contiguous-pack interpretation would place source
        # padding/channel bytes at d0 offset 0.  The first correct element is
        # source offset 2, not source offset 0.
        self.assertEqual(memory[0:2], bytes(self.tensors["src0"]["values"][2:4]))

    def test_duplicate_and_stale_retries_cannot_erase_effective_block(self):
        session = HandoffSession(self.record, self.tensors, self.capacity).run_events()
        key = ("memory", 16)
        good_digest = session.block_state[key]["digest"]

        duplicate = session.receive_packet(key, 7, session.packets[-1].payload)
        # Find the actual seq for block 16; retry through the stored packet.
        block_packet = next(p for p in session.packets if p.key == key and p.outcome == "applied")
        retry = session.retry_packet(block_packet.seq)
        self.assertEqual(retry["outcome"], "duplicate_idempotent")
        self.assertEqual(session.block_state[key]["digest"], good_digest)

        session.deliver_source("s1", 6)
        stale_payload = session.build_joint_packet(key, 6)
        stale = session.receive_packet(key, 6, stale_payload)
        self.assertEqual(stale["outcome"], "rejected_stale_generation")

        conflicting = bytearray(session.build_joint_packet(key, 7))
        conflicting[0] ^= 0xFF
        conflict = session.receive_packet(key, 7, bytes(conflicting))
        self.assertEqual(conflict["outcome"], "rejected_version_conflict")
        self.assertTrue(session.result().ok)

    def test_missing_shared_grant_is_exact_unlandable_diagnostic(self):
        record = copy.deepcopy(self.record)
        record["shared_grants"] = []
        result = execute_handoff(record, self.tensors, self.capacity)
        self.assertFalse(result.ok)
        codes = {d["code"] for d in result.plan.diagnostic_dicts()}
        self.assertIn("shared_grant_missing", codes)
        self.assertEqual(result.views["d0"].status, "unlandable")
        self.assertIsNotNone(result.views["d0"].diagnostic)

    def test_old_generation_event_does_not_complete_new_fence(self):
        session = HandoffSession(self.record, self.tensors, self.capacity)
        session.run_events([
            {"source": "s1", "generation": 7},
            {"source": "s0", "generation": 6},
            {"release": "old-d0"},
        ])
        fence = session.fence("d0", 7)
        self.assertEqual(fence["outcome"], "not_complete")
        self.assertEqual(session.read_view("d0").status, "incomplete")


class SyntheticLayoutTests(unittest.TestCase):
    def _case(self):
        n, c, h, w = 1, 2, 2, 3
        values = np.zeros((n, c, h, w), dtype="<u2")
        for cc in range(c):
            for hh in range(h):
                for ww in range(w):
                    values[0, cc, hh, ww] = 1 + cc * 100 + hh * 10 + ww
        raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        page_size = 64
        memory = np.full(page_size, 0xAA, dtype=np.uint8)
        record = {
            "global": [n, c, h, w],
            "generation": 3,
            "burst_align": 8,
            "link_max": 8,
            "sources": [{
                "id": "s0", "tensor": "src",
                "box": [[0, n], [0, c], [0, h], [0, w]],
                "layout": "NCHW",
                "strides": [c * h * w * 2, h * w * 2, w * 2, 2],
                "base": 0,
            }],
            "destinations": [{
                "id": "d0", "page": "memory",
                "box": [[0, n], [0, c], [0, h], [0, w]],
                "layout": "NHWC",
                # H row contains W*C elements; W stride C*2.
                "strides": [h * w * c * 2, w * c * 2, c * 2, 2],
                "base": 0,
                "fill": [],
                "write_grants": [[0, 24]],
            }],
            "shared_grants": [],
            "live_reads": [],
            "delivery_events": [{"source": "s0", "generation": 3}],
        }
        tensors = {
            "src": {"dtype": "uint8", "shape": [raw.size], "values": raw.tolist()},
            "memory": {"dtype": "uint8", "shape": [page_size], "values": memory.tolist()},
        }
        return record, tensors

    def test_nchw_to_nhwc_transpose_values(self):
        record, tensors = self._case()
        result = execute_handoff(record, tensors)
        self.assertTrue(result.ok, result.plan.diagnostic_dicts())
        view = result.numpy_view("d0")
        self.assertEqual(view.shape, (1, 2, 3, 2))  # N,H,W,C
        for hh in range(2):
            for ww in range(3):
                for cc in range(2):
                    self.assertEqual(int(view[0, hh, ww, cc]), 1 + cc * 100 + hh * 10 + ww)
        # Alignment tail [24,32) was authorised and preserves old bytes.
        self.assertEqual(bytes(result.pages["memory"][24:32]), b"\xaa" * 8)

    def test_uncovered_channel_is_reported_as_logical_box(self):
        record, tensors = self._case()
        record["sources"][0]["box"][1] = [0, 1]
        record["sources"][0]["strides"][1] = 12
        result = execute_handoff(record, tensors)
        self.assertFalse(result.ok)
        self.assertIn("uncovered_target", {d.code for d in result.plan.diagnostics})
        self.assertEqual(result.views["d0"].status, "unlandable")

    def test_unauthorized_alignment_byte_fails(self):
        record, tensors = self._case()
        record["burst_align"] = 16
        record["link_max"] = 16
        record["destinations"][0]["write_grants"] = [[0, 24]]
        result = execute_handoff(record, tensors)
        self.assertFalse(result.ok)
        self.assertTrue(any(d["code"] == "write_not_authorized" for d in result.plan.diagnostic_dicts()))

    def test_range_coordinate_generator_visits_each_element(self):
        record, tensors = self._case()
        plan = build_plan(record, tensors)
        dest = plan.destinations[0]
        coords = list(iter_coords_in_byte_range(dest, (8, 16), dest.box))
        intervals = [dest.element_interval(c) for c in coords]
        self.assertTrue(all(start < 16 and end > 8 for start, end in intervals))
        self.assertGreater(len(coords), 0)


if __name__ == "__main__":
    unittest.main()
