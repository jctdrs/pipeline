import pytest

from setup import pipeline_validation
from setup import config_validation
from setup import bands_validation


class TestMeta:
    def test_valid_creation(self):
        meta = pipeline_validation.Meta(
            name="HIP", description="Description for testing"
        )
        assert meta.name == "HIP"
        assert meta.description == "Description for testing"

    def test_valid_creation_when_empty(self):
        meta = pipeline_validation.Meta()
        assert meta.name == "Default"
        assert meta.description == "Default"


class TestConfig:
    def test_valid_MC_creation(self):
        config = config_validation.Config(mode="Monte-Carlo", niter=10000)
        assert config.mode == "Monte-Carlo"
        assert config.niter == 10000

    def test_MC_creation_with_missing_niter(self):
        config = config_validation.Config(mode="Monte-Carlo")
        assert config.mode == "Single Pass"
        assert config.niter == 1

    def test_SinglePass_creation(self):
        config = config_validation.Config(mode="Single Pass")
        assert config.mode == "Single Pass"
        assert config.niter == 1

    def test_SinglePass_creation_with_niter(self):
        config = config_validation.Config(mode="Single Pass", niter=10000)
        assert config.mode == "Single Pass"
        assert config.niter == 1

    def test_Analytic_creation(self):
        config = config_validation.Config(mode="Analytic")
        assert config.mode == "Analytic"
        assert config.niter == 1

    def test_Analytic_creation_with_niter(self):
        config = config_validation.Config(mode="Analytic", niter=10000)
        assert config.mode == "Analytic"
        assert config.niter == 1

    def test_invalid_mode(self):
        with pytest.raises(ValueError) as e:
            config = config_validation.Config(mode="Something", niter=1000)  # noqa
        assert "Input should be 'Single Pass', 'Monte-Carlo' or 'Analytic'" in str(
            e.value
        )

        with pytest.raises(ValueError) as e:
            config = config_validation.Config(mode="Something")  # noqa
        assert "Input should be 'Single Pass', 'Monte-Carlo' or 'Analytic'" in str(
            e.value
        )


class TestBands:
    def test_valid_creation_with_error_file(self):
        bands = bands_validation.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            error="test/data/error/NGC4254_PACS1_Error.fits",
            output="test/data/outputs/",
            calError=5.0,
            name="PACS1",
        )
        assert bands.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bands.error == "test/data/error/NGC4254_PACS1_Error.fits"
        assert bands.output == "test/data/outputs/"
        assert bands.calError == 5.0
        assert bands.name == "PACS1"

    def test_valid_creation_without_error_file(self):
        bands = bands_validation.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            calError=5.0,
            name="PACS1",
        )
        assert bands.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bands.error is None
        assert bands.output == "test/data/outputs/"
        assert bands.calError == 5.0
        assert bands.name == "PACS1"

    def test_valid_creation_without_calError(self):
        bands = bands_validation.Band(
            input="test/data/inputs/NGC4254_PACS1.fits",
            output="test/data/outputs/",
            name="PACS1",
        )
        assert bands.input == "test/data/inputs/NGC4254_PACS1.fits"
        assert bands.error is None
        assert bands.output == "test/data/outputs/"
        assert bands.calError == 5.4
        assert bands.name == "PACS1"

    def test_invalid_creation_with_negative_calError(self):
        with pytest.raises(ValueError) as e:
            bands_validation.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                output="test/data/outputs/",
                name="PACS1",
                calError=-1.5,
            )
        assert "Input should be greater than or equal to 0" in str(e.value)

    def test_invalid_creation_files_not_found(self):
        with pytest.raises(OSError) as e:
            bands_validation.Band(
                input="test/data/inputs/NGC4254_PACS2.fits",
                output="test/data/outputs/",
                name="PACS1",
            )
        assert "Path test/data/inputs/NGC4254_PACS2.fits not found." in str(e.value)

        with pytest.raises(OSError) as e:
            bands_validation.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                error="test/data/inputs/NGC4254_PACS2.fits",
                output="test/data/outputs/",
                name="PACS1",
            )
        assert "Path test/data/inputs/NGC4254_PACS2.fits not found." in str(e.value)

        with pytest.raises(OSError) as e:
            bands_validation.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                error="test/data/inputs/NGC4254_PACS1_Error.fits",
                output="test/data/outputss/",
                name="PACS1",
            )
        assert "Path test/data/outputss/ not found." in str(e.value)

    def test_invalid_creation_without_input(self):
        with pytest.raises(ValueError) as e:
            bands_validation.Band(
                output="test/data/outputs/",
                name="PACS1",
            )

        assert "1 validation error" in str(e.value)

    def test_invalid_creation_with_name_not_HerBie(self):
        with pytest.raises(ValueError) as e:
            bands_validation.Band(
                input="test/data/inputs/NGC4254_PACS1.fits",
                error="test/data/error/NGC4254_PACS1_Error.fits",
                output="test/data/outputs/",
                name="PACS11",
            )
        assert "Band 'PACS11' not valid HerBie naming." in str(e.value)
