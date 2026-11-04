# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import tempfile
import unittest
from pathlib import Path

import numpy as np

from examples.common.light_eval import compare, summarize, write_json


class LightEvalComparisonTest(unittest.TestCase):
    def test_known_angles_and_token_flip(self):
        a = np.array([[1.0, 0.0], [1.0, 0.0]])
        b = np.array([[2.0, 0.0], [0.0, 1.0]])
        result = compare(a, b)
        self.assertEqual(result["cosine_by_step"], [1.0, 0.0])
        self.assertEqual(result["cosine_mean"], 0.5)
        self.assertEqual(result["worst_step"], 1)
        self.assertEqual(result["top1_agreement"], 0.5)

    def test_reject_corrupt_or_incompatible_outputs(self):
        for a, b in [
            (np.ones((2, 3)), np.ones((1, 3))),
            (np.ones((2, 3)), np.zeros((2, 3))),
            (np.ones((2, 3)), np.full((2, 3), np.nan)),
        ]:
            with self.assertRaises(ValueError):
                compare(a, b)

    def test_baseline_pass_warning_and_trajectory_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            base, run = Path(temp) / "base", Path(temp) / "run"
            cfg = {
                "model": "gemma4",
                "capacity": 5120,
                "output_tokens": 2,
                "repeats": 1,
            }
            for root, backends in [(base, ["pte", "eager"]), (run, ["pte"])]:
                write_json(
                    root / "inputs/cases.json", [{"name": "short", "length": 128}]
                )
                for backend in backends:
                    folder = root / backend / "short"
                    write_json(
                        folder / "trial_0.json",
                        {
                            "output_tokens": 2,
                            "vocab_size": 2,
                            "ttft_ms": 1.0,
                            "tpot_ms": 1.0,
                            "total_ms": 2.0,
                            "throughput_tps": 1000.0,
                        },
                    )
                    np.array([[1.0, 0.0], [1.0, 0.0]], dtype="<f4").tofile(
                        folder / "trial_0.f32"
                    )
                    np.array([0, 0], dtype="<i8").tofile(folder / "trajectory.i64")
            summarize(cfg, base, base, True)
            self.assertEqual(summarize(cfg, run, base, False)["status"], "PASS")
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype="<f4").tofile(
                run / "pte/short/trial_0.f32"
            )
            self.assertEqual(summarize(cfg, run, base, False)["status"], "WARN")
            np.array([1, 0], dtype="<i8").tofile(run / "pte/short/trajectory.i64")
            with self.assertRaisesRegex(ValueError, "trajectories"):
                summarize(cfg, run, base, False)

    def test_high_cosine_does_not_hide_argmax_flip(self):
        a = np.array([[100.0, 99.99]])
        b = np.array([[99.99, 100.0]])
        result = compare(a, b)
        self.assertGreater(result["cosine_min"], 0.999999)
        self.assertEqual(result["top1_agreement"], 0.0)


if __name__ == "__main__":
    unittest.main()
