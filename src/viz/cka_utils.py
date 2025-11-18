"""CKA (Centered Kernel Alignment) utilities for layer similarity analysis."""

import numpy as np
import torch
from typing import Dict, Optional, Tuple


def centering(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix K.
    
    Args:
        K: Kernel matrix of shape (n, n)
    
    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n
    return np.dot(np.dot(H, K), H)


def linear_CKA(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute linear CKA between two sets of representations.
    
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    
    Args:
        X: Representations from layer 1, shape (n_samples, dim_x)
        Y: Representations from layer 2, shape (n_samples, dim_y)
    
    Returns:
        CKA similarity score in [0, 1]
    """
    # X and Y should have same number of samples
    assert X.shape[0] == Y.shape[0], f"X and Y must have same number of samples, got {X.shape[0]} vs {Y.shape[0]}"
    
    # Handle edge cases
    if X.shape[0] < 2:
        return 1.0 if np.allclose(X, Y) else 0.0
    
    # Flatten if multidimensional
    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(Y.shape[0], -1)
    
    # Check for zero variance
    if np.std(X) < 1e-10 or np.std(Y) < 1e-10:
        return 1.0 if np.allclose(X, Y) else 0.0
    
    # Compute centered kernel matrices
    X = X - np.mean(X, axis=0, keepdims=True)
    Y = Y - np.mean(Y, axis=0, keepdims=True)
    
    # Linear kernels
    K_X = X @ X.T
    K_Y = Y @ Y.T
    
    # Center the kernels
    K_X = centering(K_X)
    K_Y = centering(K_Y)
    
    # Compute HSIC (Hilbert-Schmidt Independence Criterion)
    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)
    
    # Avoid division by zero
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 1.0 if np.allclose(X, Y) else 0.0
    
    # CKA
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    
    # Clamp to [0, 1] range (numerical errors can cause slight violations)
    cka = np.clip(cka, 0.0, 1.0)
    
    return float(cka)


def rbf_CKA(X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None) -> float:
    """
    Compute RBF (Gaussian) kernel CKA between two sets of representations.
    
    Args:
        X: Representations from layer 1, shape (n_samples, dim_x)
        Y: Representations from layer 2, shape (n_samples, dim_y)
        sigma: RBF kernel bandwidth (if None, use median heuristic)
    
    Returns:
        CKA similarity score in [0, 1]
    """
    assert X.shape[0] == Y.shape[0], "X and Y must have same number of samples"
    
    def rbf_kernel(A, sigma):
        # Compute pairwise squared distances
        sq_dists = np.sum(A**2, axis=1, keepdims=True) + \
                   np.sum(A**2, axis=1, keepdims=True).T - \
                   2 * A @ A.T
        
        # RBF kernel
        return np.exp(-sq_dists / (2 * sigma**2))
    
    # Median heuristic for sigma if not provided
    if sigma is None:
        sigma_x = np.median(np.sqrt(np.sum((X[:, None] - X[None, :])**2, axis=-1)))
        sigma_y = np.median(np.sqrt(np.sum((Y[:, None] - Y[None, :])**2, axis=-1)))
        sigma = (sigma_x + sigma_y) / 2
        sigma = max(sigma, 1e-10)  # Avoid zero
    
    K_X = rbf_kernel(X, sigma)
    K_Y = rbf_kernel(Y, sigma)
    
    # Center the kernels
    K_X = centering(K_X)
    K_Y = centering(K_Y)
    
    # Compute HSIC
    hsic_xy = np.sum(K_X * K_Y)
    hsic_xx = np.sum(K_X * K_X)
    hsic_yy = np.sum(K_Y * K_Y)
    
    if hsic_xx < 1e-10 or hsic_yy < 1e-10:
        return 0.0
    
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    return float(cka)


def extract_layer_representations(model: torch.nn.Module, 
                                  dataloader: torch.utils.data.DataLoader,
                                  layer_names: list,
                                  device: str = 'cuda',
                                  max_samples: int = 500) -> Dict[str, np.ndarray]:
    """
    Extract representations from specified layers.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for input data
        layer_names: List of layer names to extract from
        device: Device to run on
        max_samples: Maximum number of samples to use
    
    Returns:
        Dictionary mapping layer names to representations (n_samples, hidden_dim)
    """
    model.eval()
    representations = {name: [] for name in layer_names}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            # Take CLS token for ViT or mean pool
            if len(output.shape) == 3:  # (batch, seq_len, hidden_dim)
                rep = output[:, 0, :].detach().cpu()  # CLS token
            else:
                rep = output.detach().cpu()
            representations[name].append(rep)
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if name in layer_names:
            handle = module.register_forward_hook(get_hook(name))
            hooks.append(handle)
    
    # Forward pass
    total_samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if total_samples >= max_samples:
                break
            
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                if len(batch) >= 3:
                    inputs = batch[0]
                else:
                    inputs = batch[0]
            elif isinstance(batch, dict):
                inputs = batch.get('pixel_values', batch.get('flux'))
            else:
                inputs = batch
            
            inputs = inputs.to(device)
            model(inputs)
            
            total_samples += inputs.shape[0]
    
    # Remove hooks
    for handle in hooks:
        handle.remove()
    
    # Concatenate representations
    result = {}
    for name in layer_names:
        if representations[name]:
            reps = torch.cat(representations[name], dim=0).numpy()
            # Flatten if needed (for fully connected layers)
            if len(reps.shape) > 2:
                reps = reps.reshape(reps.shape[0], -1)
            result[name] = reps
    
    return result


def compute_layer_cka_matrix(initial_reps: Dict[str, np.ndarray],
                             trained_reps: Dict[str, np.ndarray],
                             use_rbf: bool = False) -> Tuple[np.ndarray, list]:
    """
    Compute CKA similarity matrix between initial and trained representations.
    
    Args:
        initial_reps: Dictionary of layer representations from initial model
        trained_reps: Dictionary of layer representations from trained model
        use_rbf: If True, use RBF kernel; otherwise use linear kernel
    
    Returns:
        Tuple of (CKA matrix, layer names)
    """
    layer_names = sorted(initial_reps.keys())
    n_layers = len(layer_names)
    
    cka_matrix = np.zeros((n_layers, n_layers))
    
    for i, name_i in enumerate(layer_names):
        for j, name_j in enumerate(layer_names):
            X = initial_reps[name_i]
            Y = trained_reps[name_j]
            
            if use_rbf:
                cka_matrix[i, j] = rbf_CKA(X, Y)
            else:
                cka_matrix[i, j] = linear_CKA(X, Y)
    
    return cka_matrix, layer_names


def compute_diagonal_cka(initial_reps: Dict[str, np.ndarray],
                        trained_reps: Dict[str, np.ndarray],
                        use_rbf: bool = False) -> Dict[str, float]:
    """
    Compute CKA between initial and trained representations for each layer.
    
    This measures how much each layer has changed during training.
    High CKA (≥0.95) indicates the layer barely changed (potential training issue).
    
    Args:
        initial_reps: Dictionary of layer representations from initial model
        trained_reps: Dictionary of layer representations from trained model
        use_rbf: If True, use RBF kernel; otherwise use linear kernel
    
    Returns:
        Dictionary mapping layer names to CKA scores
    """
    cka_scores = {}
    
    for name in initial_reps.keys():
        if name not in trained_reps:
            continue
        
        X = initial_reps[name]
        Y = trained_reps[name]
        
        if use_rbf:
            cka_scores[name] = rbf_CKA(X, Y)
        else:
            cka_scores[name] = linear_CKA(X, Y)
    
    return cka_scores
