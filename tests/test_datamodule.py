import pytest

from src.datamodules.setfit_datamodule import DataModule


@pytest.mark.parametrize("num_samples", [8, 16])
def test_datamodule(num_samples):
    column_mapping = {
        "sentence": "text",
        "label": "label",
    }
    dm = DataModule(num_samples=num_samples, column_mapping=column_mapping)

    assert dm.train_dataset and dm.valid_dataset and dm.test_dataset
    assert len(dm.train_dataset) == num_samples * 2
