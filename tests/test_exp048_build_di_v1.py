import importlib.util
from pathlib import Path

import numpy as np


SCRIPT=(
    Path.home()
    / "thesis"
    / "identitygate"
    / "scripts"
    / "exp048_build_di_v1.py"
)

spec=importlib.util.spec_from_file_location(
    "exp048",
    SCRIPT,
)

module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_percentile_rank_ties():
    got=module.percentile_rank(
        [1,1,3,4]
    )

    want=np.asarray([
        1.5/4,
        1.5/4,
        3/4,
        4/4,
    ])

    assert np.allclose(got,want)


def test_smaller_size_becomes_harder():
    size=np.asarray(
        [0.10,0.01,0.001],
        dtype=np.float64,
    )

    ranks=module.percentile_rank(-size)

    assert ranks[2] > ranks[1] > ranks[0]


def test_cluster_cap():
    scores=np.asarray(
        [9,8,7,6,5,4],
        dtype=np.float64,
    )

    labels=np.asarray(
        [0,0,0,1,1,2],
        dtype=np.int64,
    )

    videos=[
        "a","b","c","d","e","f"
    ]

    selected,cap,used=module.select_with_cap(
        scores,
        labels,
        videos,
        4,
        0.5,
    )

    assert cap==2
    assert selected==[0,1,3,4]
    assert max(used.values())==2


if __name__=="__main__":
    test_percentile_rank_ties()
    test_smaller_size_becomes_harder()
    test_cluster_cap()

    print("EXP048_TESTS=PASS tests=3")
