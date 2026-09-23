import importlib.util
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp051_fresh_dev_headline_outcomes.py"
CFG = ROOT / "configs/EXP051-fresh-dev-headline-outcomes-v1.json"


def load_module():
    spec = importlib.util.spec_from_file_location("exp051_tested", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules["exp051_tested"] = module
    spec.loader.exec_module(module)
    return module


class Exp051Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load_module()
        cls.cfg = json.loads(CFG.read_text(encoding="utf-8"))

    def test_neutral_zero_and_full(self):
        self.assertEqual(self.m.neutral_admit_frames(10, 0), [])
        self.assertEqual(
            self.m.neutral_admit_frames(10, 10),
            list(range(1, 11)),
        )

    def test_neutral_mid_budget(self):
        self.assertEqual(
            self.m.neutral_admit_frames(10, 3),
            [2, 6, 9],
        )

    def test_neutral_invalid_budget(self):
        with self.assertRaises(ValueError):
            self.m.neutral_admit_frames(5, 6)

    def test_theft_requires_five_consecutive_frames(self):
        event, first, best = self.m.theft_episode_from_flags(
            [10, 11, 12, 13, 14],
            [True, True, True, True, True],
            5,
        )
        self.assertEqual((event, first, best), (1, 10, 5))

    def test_theft_gap_breaks_streak(self):
        event, first, best = self.m.theft_episode_from_flags(
            [10, 11, 13, 14, 15, 16],
            [True, True, True, True, True, True],
            5,
        )
        self.assertEqual(event, 0)
        self.assertIsNone(first)
        self.assertEqual(best, 4)

    def test_event_interval_truncates_at_next_qualifying_gap(self):
        vis = [
            True, True,
            False, False, False, False, False,
            True, True, True,
            False, False, False, False, False,
            True, True,
        ]
        frames, evaluable, stop = self.m.event_interval(
            vis, 7, 5, 30
        )
        self.assertEqual(frames, [7, 8, 9])
        self.assertEqual(evaluable, 3)
        self.assertEqual(stop, 10)

    def test_frozen_headline_contract(self):
        self.assertEqual(self.cfg["headline"]["r_star"], 0.3)
        self.assertEqual(
            self.cfg["headline"]["taus"],
            {
                "B1": 0.1,
                "B2": 0.1,
                "B3_R": 0.2,
                "B3_S": 0.2,
                "B5": 0.7,
            },
        )
        self.assertFalse(
            self.cfg["boundary"]["test_runtime_inputs_allowed"]
        )
        self.assertFalse(self.cfg["boundary"]["test_touched"])


if __name__ == "__main__":
    unittest.main()
