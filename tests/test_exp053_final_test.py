import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(ROOT / "scripts"),
)

import exp053_final_test as m


class TestExp053(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = m.load_cfg()

    def test_scope_spec_counts(self):
        hard = m.build_specs(
            self.cfg,
            "HARD_TEST80",
        )
        representative = m.build_specs(
            self.cfg,
            "REPRESENTATIVE_TEST40",
        )

        self.assertEqual(len(hard), 39)
        self.assertEqual(
            len(representative),
            11,
        )

    def test_headline_curve_reuse(self):
        self.assertEqual(
            m.curve_run_labels(
                0.3,
                "B3_S",
            ),
            (
                "B3_S",
                "B3_S_NEUTRAL",
            ),
        )

        self.assertEqual(
            m.curve_run_labels(
                0.1,
                "B3_S",
            ),
            (
                "CURVE_R010_B3_S",
                "CURVE_R010_B3_S_NEUTRAL",
            ),
        )

    def test_rate_mismatch_boundary(self):
        tolerance = float(
            self.cfg["rate_mismatch"][
                "absolute_pooled_rate_tolerance"
            ]
        )

        self.assertFalse(
            m.rate_difference_exceeds_tolerance(
                30,
                100,
                319,
                1000,
                tolerance,
            )
        )

        self.assertFalse(
            m.rate_difference_exceeds_tolerance(
                30,
                100,
                32,
                100,
                tolerance,
            )
        )

        self.assertTrue(
            m.rate_difference_exceeds_tolerance(
                30,
                100,
                321,
                1000,
                tolerance,
            )
        )

        def summary(k, n):
            return {
                "sum_admit_count": k,
                "sum_eligible_count": n,
                "pooled_write_rate": k / n,
            }

        fake = {
            "B3_S": summary(30, 100),
            "B2": summary(32, 100),
            "B3_R": summary(30, 100),
            "B1": summary(30, 100),
            "B5": summary(30, 100),
        }

        rows = m.rate_status_rows(
            self.cfg,
            "HARD_TEST80",
            fake,
        )

        primary = next(
            row
            for row in rows
            if row["comparison"]
            == "B3_S_minus_B2"
        )

        self.assertEqual(
            primary["rate_status"],
            "MATCHED_ON_TEST",
        )

        fake["B2"] = summary(
            321,
            1000,
        )

        rows = m.rate_status_rows(
            self.cfg,
            "HARD_TEST80",
            fake,
        )

        primary = next(
            row
            for row in rows
            if row["comparison"]
            == "B3_S_minus_B2"
        )

        self.assertEqual(
            primary["rate_status"],
            "RATE_MISMATCH",
        )

    def test_cache_event_key_validation(self):
        primary = [
            {
                "video": "v",
                "object_id": "1",
                "reappear_frame": "20",
                "disappear_start": "10",
            }
        ]

        expected = {
            "experiment": "EXP053",
            "run_label": "X",
            "test_touched": True,
        }

        row = {
            **expected,
            "eligible_nonconditioning_frames": 10,
            "admit_count": 3,
            "block_count": 7,
            "write_rate": 0.3,
            "event_rows": [
                {
                    "video": "v",
                    "object_id": 1,
                    "reappear_frame": 20,
                    "disappear_start": 10,
                    "run_label": "X",
                    "recovered_w30": 1,
                    "theft_event_w30": 0,
                }
            ],
            "por30": 1.0,
            "itr30": 0.0,
            "peak_vram_gb": 1.0,
            "runtime_sec": 1.0,
        }

        checked = m.validate_cached(
            row,
            expected,
            primary,
        )

        self.assertIs(checked, row)

        bad = copy.deepcopy(row)
        bad["event_rows"][0][
            "reappear_frame"
        ] = 21

        with self.assertRaises(
            RuntimeError
        ):
            m.validate_cached(
                bad,
                expected,
                primary,
            )

    def test_finalization_state_machine(self):
        self.assertEqual(
            m.classify_finalization_state(
                "run",
                False,
                False,
            ),
            "EXECUTE",
        )

        self.assertEqual(
            m.classify_finalization_state(
                "resume",
                True,
                False,
            ),
            "RECOVER_FINAL_OUTPUT",
        )

        self.assertEqual(
            m.classify_finalization_state(
                "resume",
                False,
                True,
            ),
            "REBUILD_STAGING",
        )

        with self.assertRaises(
            RuntimeError
        ):
            m.classify_finalization_state(
                "run",
                True,
                False,
            )

        with self.assertRaises(
            RuntimeError
        ):
            m.classify_finalization_state(
                "resume",
                True,
                True,
            )

    def test_artifact_manifest_hash_guard(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)

            for name in m.OUTPUT_DATA_FILES:
                (
                    out / name
                ).write_text(
                    "fixture\\n",
                    encoding="utf-8",
                )

            m.write_artifact_manifest(
                self.cfg,
                out,
            )

            self.assertTrue(
                m.verify_artifact_manifest(
                    self.cfg,
                    out,
                )
            )

            (
                out / "summary.json"
            ).write_text(
                "tampered\\n",
                encoding="utf-8",
            )

            with self.assertRaises(
                RuntimeError
            ):
                m.verify_artifact_manifest(
                    self.cfg,
                    out,
                )

    def test_campaign_marker_run_resume(self):
        cfg = copy.deepcopy(self.cfg)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            cfg["runtime"][
                "scratch_dir"
            ] = str(root / "scratch")

            cfg["runtime"][
                "touch_marker"
            ] = str(
                root
                / "scratch"
                / "TEST_CAMPAIGN.json"
            )

            first = (
                m.start_or_resume_campaign(
                    cfg,
                    "run",
                )
            )

            self.assertEqual(
                first["status"],
                "RUNNING",
            )

            resumed = (
                m.start_or_resume_campaign(
                    cfg,
                    "resume",
                )
            )

            self.assertEqual(
                resumed["repo_commit"],
                first["repo_commit"],
            )

            with self.assertRaises(
                RuntimeError
            ):
                m.start_or_resume_campaign(
                    cfg,
                    "run",
                )


if __name__ == "__main__":
    unittest.main()
