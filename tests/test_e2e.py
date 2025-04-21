import subprocess


def test_NGC2454_pipeline():
    result = subprocess.run(
        ["python", "main.py", "-f", "tests/pipeline_test.yml"],
        check=True,
        capture_output=True,
    )
    assert "Integrated flux PACS1 = 62.654 Jy/px" in result.stdout.decode("utf-8")
