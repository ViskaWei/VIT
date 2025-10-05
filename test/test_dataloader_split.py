import torch

from src.basemodule import BaseDataset as BaseDatasetFromBasemodule
from src.dataloader import BaseDataset, ClassSpecDataset, RegSpecDataset, TestDataset


def test_base_dataset_reexport():
    """Ensure BaseDataset is re-exported for backward compatibility."""
    assert BaseDatasetFromBasemodule is BaseDataset


def test_test_dataset_classification_shapes(tmp_path):
    config = {
        "data": {"num_samples": 4, "cov_path": str(tmp_path / "cov.pt")},
        "model": {"task_type": "cls"},
    }
    dataset = TestDataset.from_config(config)
    dataset.load_data()

    assert len(dataset) == 4
    flux, error, label = dataset[0]
    assert flux.shape == (4096,)
    assert torch.allclose(error, torch.full_like(error, 1e-3))
    assert label.dtype == torch.long


def test_test_dataset_regression_labels_are_float(tmp_path):
    config = {
        "data": {"num_samples": 2, "cov_path": str(tmp_path / "cov.pt")},
        "model": {"task_type": "reg"},
    }
    dataset = TestDataset.from_config(config)
    dataset.load_data()

    _, _, label = dataset[0]
    assert label.dtype == torch.float32


def test_class_spec_dataset_getitem_uses_labels():
    dataset = ClassSpecDataset()
    dataset.flux = torch.randn(3, 5)
    dataset.error = torch.ones_like(dataset.flux)
    dataset.labels = torch.tensor([0, 1, 1])

    _, _, label = dataset[2]
    assert label.item() == 1


def test_reg_spec_dataset_label_normalization():
    dataset = RegSpecDataset()
    dataset.labels = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset.label_norm = "standard"
    dataset._maybe_normalize_labels(stage="fit")
    assert torch.isclose(dataset.labels.mean(dim=0), torch.zeros(2), atol=1e-5).all()

    dataset.labels = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    dataset.label_norm = "minmax"
    dataset._maybe_normalize_labels(stage="fit")
    assert torch.isclose(dataset.labels.min(dim=0).values, torch.zeros(2), atol=1e-6).all()
    assert torch.isclose(dataset.labels.max(dim=0).values, torch.ones(2), atol=1e-6).all()
