from agentic_review_annotation_distilabel.agents.run import load_instance, swebench_image


def test_loads_first_swebench_image():
    row = load_instance(
        "data/swe-bench-verified/SWE-bench_Verified",
        0,
    )
    assert row["instance_id"] == "astropy__astropy-12907"
    assert swebench_image(row) == (
        "swebench/sweb.eval.x86_64.astropy_1776_astropy-12907:latest"
    )
