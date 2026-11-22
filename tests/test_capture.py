"""核对合成采集的交换记录与原始尺寸。"""
import json
from pathlib import Path
import unittest

import numpy as np

from caseio import load_case


class CaptureTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1] / "data"
        self.records, self.tensors = load_case(data / "capture.json")
        self.capacity = json.loads((data / "capacity.json").read_text(encoding="utf-8"))

    def test_capture_contract(self):
        r, a = self.records, self.tensors
        self.assertEqual(a["memory"].dtype, np.uint8)
        self.assertEqual(len(a["memory"]), r["destinations"][1]["write_grants"][0][1])
        self.assertNotEqual(r["sources"][0]["layout"], r["sources"][1]["layout"])
        self.assertLess(r["destinations"][1]["base"], r["destinations"][0]["write_grants"][0][1])
