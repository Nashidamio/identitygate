import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/exp050_fresh_dev_rate_selection.py"
CONFIG = ROOT / "configs/EXP050-fresh-dev-rate-selection-v1.json"


spec = importlib.util.spec_from_file_location(
    "exp050_under_test",
    SCRIPT,
)

if spec is None or spec.loader is None:
    raise RuntimeError("Cannot import EXP050 script")

module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_membership_hash():
    got = module.membership_hash(
        "DEV40",
        ["a", "b"],
    )

    want = hashlib.sha256(
        b"DEV40\na\nb\n"
    ).hexdigest()

    assert got == want


def test_pooled_rate_is_sum_k_over_sum_n():
    rows = [
        {
            "eligible_opportunities": 9,
            "admit_count": 9,
            "block_count": 0,
            "peak_vram_gb": 6.0,
            "runtime_sec": 1.0,
        },
        {
            "eligible_opportunities": 99,
            "admit_count": 0,
            "block_count": 99,
            "peak_vram_gb": 6.2,
            "runtime_sec": 2.0,
        },
    ]

    point = module.aggregate_video_points(
        "B2",
        0.5,
        rows,
        "UNIT",
    )

    assert point["eligible_opportunities"] == 108
    assert point["admit_count"] == 9
    assert point["block_count"] == 99
    assert point["write_rate"] == 9 / 108

    # It must not use the per-video macro mean.
    assert point["write_rate"] != 0.5


def test_action_conservation_rejects_invalid_pool():
    rows = [
        {
            "eligible_opportunities": 10,
            "admit_count": 4,
            "block_count": 5,
            "peak_vram_gb": 1.0,
            "runtime_sec": 1.0,
        }
    ]

    try:
        module.aggregate_video_points(
            "B1",
            0.3,
            rows,
            "UNIT",
        )
    except RuntimeError as exc:
        assert "Pooled action-count mismatch" in str(exc)
    else:
        raise AssertionError(
            "Invalid pooled action counts were accepted"
        )


def test_selection_row_is_rate_only_structure():
    result = {
        "status": "MATCH",
        "selected": {
            "tau": 0.35,
            "write_rate": 0.491,
        },
        "refinements": 2,
        "tested_taus": [
            0.3,
            0.4,
            0.35,
        ],
    }

    row = module.selection_row(
        0.5,
        "B1",
        result,
    )

    assert row["target_rate"] == 0.5
    assert row["variant"] == "B1"
    assert row["status"] == "MATCH"
    assert row["selected_tau"] == 0.35
    assert row["realized_write_rate"] == 0.491
    assert abs(
        row["absolute_rate_error"] - 0.009
    ) < 1e-12

    forbidden = {
        "por30",
        "itr",
        "jf",
        "uar",
        "contamination",
    }

    assert forbidden.isdisjoint(
        set(row)
    )


def test_real_fresh_dev_contract():
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    rows = module.load_dev_rows(
        cfg
    )

    assert len(rows) == 40

    assert sum(
        int(r["primary_event_count"])
        for r in rows
    ) == 233

    videos = [
        r["video"]
        for r in rows
    ]

    assert module.membership_hash(
        "DEV40",
        videos,
    ) == (
        "fde1d5ba4787fa627948301183256a00102a50ab8be2d41a4dd756cd1a859e8d"
    )


def test_a4_selector_matches_frozen_sources():
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    exp038 = json.loads(
        (
            ROOT
            / cfg["source_dependencies"][
                "exp038_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    exp039 = json.loads(
        (
            ROOT
            / cfg["source_dependencies"][
                "exp039_config"
            ]["path"]
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg["a4_selector"]
        == exp038["a4_selector"]
    )

    assert (
        cfg["a4_selector"]
        == exp039["a4_selector"]
    )


def test_outcome_and_test_firewall():
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    firewall = cfg[
        "selection_firewall"
    ]

    assert (
        firewall["pooled_write_rate"]
        == "SUM_K_OVER_SUM_N"
    )

    assert firewall["por_computed"] is False
    assert firewall["itr_computed"] is False
    assert firewall["jf_computed"] is False
    assert firewall["uar_computed"] is False
    assert firewall["contamination_computed"] is False

    assert (
        firewall[
            "tracking_outcome_used_for_threshold_selection"
        ]
        is False
    )

    assert (
        cfg["boundary"]["test_touched"]
        is False
    )

    assert (
        cfg["boundary"][
            "test_runtime_inputs_allowed"
        ]
        is False
    )

    script_text = SCRIPT.read_text(
        encoding="utf-8"
    )

    config_text = CONFIG.read_text(
        encoding="utf-8"
    )

    combined = (
        script_text
        + "\n"
        + config_text
    )

    forbidden_paths = [
        "hard_test.csv",
        "representative_test.csv",
        "EXP049_final_split/hard_test",
        "EXP049_final_split/representative_test",
    ]

    for token in forbidden_paths:
        assert token not in combined


if __name__ == "__main__":
    tests = [
        test_membership_hash,
        test_pooled_rate_is_sum_k_over_sum_n,
        test_action_conservation_rejects_invalid_pool,
        test_selection_row_is_rate_only_structure,
        test_real_fresh_dev_contract,
        test_a4_selector_matches_frozen_sources,
        test_outcome_and_test_firewall,
    ]

    for test in tests:
        test()

    print(
        "EXP050_TESTS=PASS tests={}".format(
            len(tests)
        )
    )
