import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path.home()
    / "thesis"
    / "identitygate"
    / "scripts"
    / "exp047_visual_embeddings.py"
)

spec = importlib.util.spec_from_file_location(
    "exp047",
    SCRIPT,
)

module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    module
)


def test_middle_frame_index():
    expected={
        1:0,
        2:1,
        3:1,
        4:2,
        5:2,
        100:50,
        101:50,
    }

    for n_frames, want in expected.items():
        got=module.middle_frame_index(
            n_frames
        )

        assert got == want, (
            n_frames,
            got,
            want,
        )


def test_embedding_contract():
    value=torch.zeros(
        3,
        384,
        dtype=torch.float32,
    )

    assert module.validate_embedding_tensor(
        value,
        3,
    )


def test_embedding_contract_rejects_wrong_dimension():
    value=torch.zeros(
        1,
        383,
        dtype=torch.float32,
    )

    failed=False

    try:
        module.validate_embedding_tensor(
            value,
            1,
        )
    except AssertionError:
        failed=True

    assert failed


if __name__ == "__main__":
    test_middle_frame_index()
    test_embedding_contract()
    test_embedding_contract_rejects_wrong_dimension()

    print(
        "EXP047_TESTS=PASS tests=3"
    )
