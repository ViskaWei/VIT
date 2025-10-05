# Spectral Preprocessing Toolkit

`src.prepca` groups the data I/O helpers and numerical routines used to build
spectral representations (PCA, KPCA, PCP, ZCA) with a consistent interface.

```python
from src.prepca.pipeline import (
    PreprocessingPipeline,
    compute_pca,
    compute_kernel_pca,
    compute_pcp,
    compute_cka,
    load_spectra,
)

data = load_spectra("./data/dataset.h5")
pca_stats = compute_pca(data["flux"], patch_size=32)
state = compute_kernel_pca(data["flux"], r=64)
```

The façade `PreprocessingPipeline` wraps data loading and exposes the
same routines through a single entry point:

```python
pipeline = PreprocessingPipeline("./data/dataset.h5", num_samples=10000)
pca = pipeline.run("pca", patch_size=64)
zca = pipeline.run("zca", gamma=0.1, eps=1e-5)
```

For command line usage, the legacy scripts now use the unified loader:

```bash
python -m src.prepca.precompute_pca --data ./data/dataset.h5 --patch-size 32
python -m src.prepca.precompute_kpca --data ./data/dataset.h5 --r 128 --landmarks 2048
```

The KPCA state is compatible with `KPCAWarmSelfAttention` for warm-starting
Transformer attention blocks.

