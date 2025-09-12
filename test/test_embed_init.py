import os
import torch

from src.model import get_model


def make_test_pca_patch(path: str, dim: int = 32):
    """Create a small PCA file with V = I (dim x dim)."""
    V = torch.eye(dim)
    torch.save({"V": V}, path)


def test_embed_init_identity(tmp_path: str | None = None):
    """Verify that when V = I and hidden_size == patch_size,
    the patch projection weights become identity (I^T == I).
    """
    p = 32
    test_path = os.path.abspath("test_pca_patch.pt") if tmp_path is None else os.path.join(tmp_path, "test_pca_patch.pt")
    make_test_pca_patch(test_path, dim=p)

    config = {
        "warmup": {
            "embed": True,
            "embed_pca_path": test_path,
            "global": False,
        },
        "model": {
            "task_type": "reg",
            "image_size": 256,     # small for construction, not used by embed init
            "patch_size": p,
            "hidden_size": p,      # same as patch for identity check
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "stride_size": p,      # non-overlapping patches
            "proj_fn": "SW",      # uses Linear patch projection
            "num_labels": 1,
        },
        "loss": {"name": "l2"},
    }

    model = get_model(config)
    proj = model.vit.embeddings.patch_embeddings.projection
    assert isinstance(proj, torch.nn.Linear)
    W = proj.weight.detach().cpu()
    I = torch.eye(p)
    assert W.shape == (p, p), f"Unexpected weight shape {tuple(W.shape)}"
    assert torch.allclose(W, I, atol=1e-6), "Projection weight is not identity after embed init"
    print("OK: projection weight initialized to identity from test_pca_patch.pt")


if __name__ == "__main__":
    test_embed_init_identity()

