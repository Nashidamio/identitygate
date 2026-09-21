import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path.home()
    / "thesis"
    / "identitygate"
    / "scripts"
    / "exp046_frame0_identity.py"
)

spec = importlib.util.spec_from_file_location(
    "exp046",
    SCRIPT,
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)


def test_cosine_identity():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    got = module.cosine_matrix_fp32(
        x,
        x,
    )

    assert torch.allclose(
        got,
        torch.eye(2),
        atol=1e-7,
        rtol=1e-7,
    )


def test_single_object_structural_missingness():
    events = [{
        "video":"v",
        "object_id":"7",
        "disappear_start":"10",
        "reappear_frame":"15",
        "frame0_object_count":"1",
        "frame0_competitor_count":"0",
    }]

    rows = module.event_identity_rows(
        events,
        [7],
        None,
    )

    assert len(rows) == 1

    assert (
        rows[0][
            "identity_pressure_status"
        ]
        == "NO_TRACKED_COMPETITOR"
    )

    assert (
        rows[0][
            "max_other_anchor_cos_fp32"
        ]
        is None
    )

    assert (
        rows[0][
            "max_other_object_id"
        ]
        is None
    )


def test_multi_object_max_competitor():
    events = [{
        "video":"v",
        "object_id":"7",
        "disappear_start":"10",
        "reappear_frame":"15",
        "frame0_object_count":"3",
        "frame0_competitor_count":"2",
    }]

    cosine = torch.tensor(
        [
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.3],
            [0.8, 0.3, 1.0],
        ],
        dtype=torch.float32,
    )

    rows = module.event_identity_rows(
        events,
        [7, 8, 9],
        cosine,
    )

    assert (
        rows[0][
            "max_other_object_id"
        ]
        == 9
    )

    assert abs(
        rows[0][
            "max_other_anchor_cos_fp32"
        ]
        - 0.8
    ) < 1e-6


if __name__ == "__main__":
    test_cosine_identity()
    test_single_object_structural_missingness()
    test_multi_object_max_competitor()

    print(
        "EXP046_TESTS=PASS tests=3"
    )
