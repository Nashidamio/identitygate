import hashlib
import importlib.util
from collections import Counter
from pathlib import Path


SCRIPT=(
    Path.home()
    / "thesis"
    / "identitygate"
    / "scripts"
    / "exp049_build_final_split.py"
)

spec=importlib.util.spec_from_file_location(
    "exp049",
    SCRIPT,
)

module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_membership_hash():
    got=module.membership_hash(
        "X",
        ["a","b"],
    )

    want=hashlib.sha256(
        b"X\na\nb\n"
    ).hexdigest()

    assert got==want


def test_quota_largest_remainder():
    got=module.allocate_dev_quotas(
        Counter({
            0:5,
            1:4,
            2:3,
        }),
        4,
        12,
    )

    assert sum(got.values())==4
    assert got=={
        0:2,
        1:1,
        2:1,
    }


def test_hard_cluster_cap():
    rows=[
        {
            "video":"a",
            "di_v1_score":"0.9",
            "visual_cluster":"0",
        },
        {
            "video":"b",
            "di_v1_score":"0.8",
            "visual_cluster":"0",
        },
        {
            "video":"c",
            "di_v1_score":"0.7",
            "visual_cluster":"0",
        },
        {
            "video":"d",
            "di_v1_score":"0.6",
            "visual_cluster":"1",
        },
        {
            "video":"e",
            "di_v1_score":"0.5",
            "visual_cluster":"1",
        },
        {
            "video":"f",
            "di_v1_score":"0.4",
            "visual_cluster":"2",
        },
    ]

    selected,cap,used=module.select_hard_pool(
        rows,
        4,
        0.5,
    )

    assert cap==2
    assert selected==[0,1,3,4]
    assert max(used.values())==2


def test_representative_excludes_hard():
    rows=[
        {
            "video":chr(ord("a")+i)
        }
        for i in range(10)
    ]

    got=module.select_representative(
        rows,
        hard=[0,1,2],
        representative_n=4,
        seed=42,
    )

    assert len(got)==4
    assert not (
        set(got)
        & {0,1,2}
    )


if __name__=="__main__":
    test_membership_hash()
    test_quota_largest_remainder()
    test_hard_cluster_cap()
    test_representative_excludes_hard()

    print("EXP049_TESTS=PASS tests=4")
