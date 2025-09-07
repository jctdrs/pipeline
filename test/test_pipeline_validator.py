import pytest

from models import pipeline
from models import config
from models import bands


class TestMeta:
    def test_valid_creation(self):
        meta = pipeline.Meta(name="HIP", description="Description for testing")
        assert meta.name == "HIP"
        assert meta.description == "Description for testing"

    def test_valid_creation_when_empty(self):
        meta = pipeline.Meta()
        assert meta.name == "Default"
        assert meta.description == "Default"


class TestConfig:
    def test_valid_MC_creation(self):
        cfg = config.Config(mode="Monte-Carlo", niter=10000)
        assert cfg.mode == "Monte-Carlo"
        assert cfg.niter == 10000

    def test_MC_creation_with_missing_niter(self):
        cfg = config.Config(mode="Monte-Carlo")
        assert cfg.mode == "Single Pass"
        assert cfg.niter == 1

    def test_SinglePass_creation(self):
        cfg = config.Config(mode="Single Pass")
        assert cfg.mode == "Single Pass"
        assert cfg.niter == 1

    def test_SinglePass_creation_with_niter(self):
        cfg = config.Config(mode="Single Pass", niter=10000)
        assert cfg.mode == "Single Pass"
        assert cfg.niter == 1

    def test_Analytic_creation(self):
        cfg = config.Config(mode="Analytic")
        assert cfg.mode == "Analytic"
        assert cfg.niter == 1

    def test_Analytic_creation_with_niter(self):
        cfg = config.Config(mode="Analytic", niter=10000)
        assert cfg.mode == "Analytic"
        assert cfg.niter == 1

    def test_invalid_mode(self):
        with pytest.raises(ValueError) as e:
            config.Config(mode="Something", niter=1000)
        assert "Input should be 'Single Pass', 'Monte-Carlo' or 'Analytic'" in str(
            e.value
        )

        with pytest.raises(ValueError) as e:
            config.Config(mode="Something")
        assert "Input should be 'Single Pass', 'Monte-Carlo' or 'Analytic'" in str(
            e.value
        )


class TestBands:
    def test_valid_creation_with_error_file(self):
        bds = bands.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            error="test/data/error/NGC4254_PACS1_Error.fits",
            output="test/data/outputs/",
            calError=5.0,
            name="PACS1",
        )
        assert bds.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bds.error == "test/data/error/NGC4254_PACS1_Error.fits"
        assert bds.output == "test/data/outputs/"
        assert bds.calError == 5.0
        assert bds.name == "PACS1"

    def test_valid_creation_without_error_file(self):
        bds = bands.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            calError=5.0,
            name="PACS1",
        )
        assert bds.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bds.error is None
        assert bds.output == "test/data/outputs/"
        assert bds.calError == 5.0
        assert bds.name == "PACS1"

    def test_valid_creation_without_calError(self):
        bds = bands.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            name="PACS1",
        )
        assert bds.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bds.error is None
        assert bds.output == "test/data/outputs/"
        assert bds.calError == 7
        assert bds.name == "PACS1"

    def test_invalid_creation_with_negative_calError(self):
        with pytest.raises(ValueError) as e:
            bands.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                output="test/data/outputs/",
                name="PACS1",
                calError=-1.5,
            )
        assert "Input should be greater than or equal to 0" in str(e.value)

    def test_invalid_creation_files_not_found(self):
        with pytest.raises(OSError) as e:
            bands.Band(
                input="test/data/inputs/NGC4254_PACS2.fits",
                output="test/data/outputs/",
                name="PACS1",
            )
        assert "Path test/data/inputs/NGC4254_PACS2.fits not found." in str(e.value)

        with pytest.raises(OSError) as e:
            bands.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                error="test/data/inputs/NGC4254_PACS2.fits",
                output="test/data/outputs/",
                name="PACS1",
            )
        assert "Path test/data/inputs/NGC4254_PACS2.fits not found." in str(e.value)

        with pytest.raises(OSError) as e:
            bands.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                error="test/data/inputs/NGC4254_PACS1_Error.fits",
                output="test/data/outputss/",
                name="PACS1",
            )
        assert "Path test/data/outputss/ not found." in str(e.value)

    def test_invalid_creation_without_input(self):
        with pytest.raises(ValueError) as e:
            bands.Band(
                output="test/data/outputs/",
                name="PACS1",
            )

        assert "1 validation error" in str(e.value)

    def test_with_custom_bands(self):
        with pytest.raises(ValueError) as e:
            bands.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                output="test/data/outputs/",
                name="radio1",
            )

        assert "Resolution not defined" in str(e.value)

        bds = bands.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            name="radio1",
            resolution="9"
        )
        assert bds.resolution == 9
        assert bds.pixelSize == 2.0000016

        bds = bands.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            name="radio1",
            resolution="9",
            pixelSize="3",
        )
        assert bds.resolution == 9
        assert bds.pixelSize == 3
