import subprocess


def test_NGC2454_pipeline():
    result = subprocess.run(
        ["python", "main.py", "-f", "tests/pipeline_test.yml"],
        check=True,
        capture_output=True,
    )
    try:
        assert "Integrated flux PACS1 = 62.654 Jy/px" in result.stdout.decode("utf-8")
    except AssertionError:
        print(result.stderr.decode("utf-8"))
        assert False
