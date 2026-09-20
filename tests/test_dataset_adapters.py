from agentic_review_annotation_distilabel.datasets import get_dataset_adapter


def test_dataset_adapters_select_image_and_workspace():
    verified = get_dataset_adapter("swebench_verified")
    assert verified.cwd == "/testbed"
    assert verified.image(
        {
            "instance_id": "astropy__astropy-12907",
            "image": "swebench/verified:latest",
        }
    ) == "swebench/verified:latest"

    multilingual = get_dataset_adapter("swebench_multilingual")
    assert multilingual.cwd == "/testbed"
    assert multilingual.image(
        {
            "instance_id": "apache__druid-13704",
            "image": "swebench/multilingual:latest",
        }
    ) == "swebench/multilingual:latest"

    pro = get_dataset_adapter("swebench_pro")
    assert pro.cwd == "/app"
    assert pro.image(
        {"instance_id": "instance_demo", "dockerhub_tag": "repo.demo-instance"}
    ) == "jefzda/sweap-images:repo.demo-instance"

    assert get_dataset_adapter(
        benchmark_path="data/SWE-bench_Multilingual"
    ).name == "swebench_multilingual"
    assert get_dataset_adapter(
        instance={"dockerhub_tag": "repo.demo-instance"}
    ).name == "swebench_pro"
