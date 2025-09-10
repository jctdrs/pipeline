import subprocess


def test_NGC2454_pipeline():
    result = subprocess.run(
        ["python", "main.py", "-f", "test/data/pipelines/pipeline_test.yml"],
        check=True,
        capture_output=True,
    )
    assert "Integrated flux PACS1 = 62.759 Jy/px" in result.stdout.decode("utf-8")
