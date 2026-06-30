"""Shared helpers for backbone feature extractors."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def to_conv_input(X, n_samples: int, n_cha: int) -> torch.Tensor:
    """EEG epochs ``(n_epochs, n_samples, n_channels)`` → 2-D-conv tensor.

    Shared by the EEG backbones' :meth:`prepare_X`: validates the natural
    per-epoch shape the user passes and returns a contiguous ``float32``
    tensor of shape ``(n_epochs, 1, n_samples, n_channels)`` — adding the
    singleton ``feature`` axis the ``nn.Conv2d`` stack expects. Deterministic
    and parameter-free; nothing else about the data is changed.

    Parameters
    ----------
    X : array-like
        Epoched EEG of shape ``(n_epochs, n_samples, n_channels)``.
    n_samples, n_cha : int
        Per-epoch shape the backbone was built for; used to validate ``X``.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(
            f"X must be a 3-D array (n_epochs, n_samples={n_samples}, "
            f"n_channels={n_cha}); got a {X.ndim}-D array of shape {X.shape}.")
    if X.shape[1:] != (n_samples, n_cha):
        raise ValueError(
            f"each epoch must be (n_samples={n_samples}, n_channels={n_cha}); "
            f"got {tuple(X.shape[1:])}. Check the axis order (samples before "
            "channels) and that the data matches this backbone.")
    return torch.from_numpy(np.ascontiguousarray(X))[:, None, :, :]


def infer_feature_dim(module: nn.Module, input_shape: tuple[int, ...]) -> int:
    """Return the feature width of ``module``'s ``forward`` by probing it.

    Runs one dummy forward on ``torch.zeros(1, *input_shape)`` so the value is
    derived from the actual layers and can never desync from a hand-derived
    formula under kernel/pool/stride changes. The module is switched to
    ``eval()`` for the probe (BatchNorm uses running stats and tolerates a
    single sample); its previous train/eval state is restored afterwards.

    Parameters
    ----------
    module : nn.Module
        Backbone whose ``forward(x)`` returns ``[N, features]``.
    input_shape : tuple of int
        Per-sample input shape excluding the batch axis, e.g.
        ``(1, n_samples, n_channels)``.

    Returns
    -------
    int
        Dimensionality of the feature vector returned by ``module.forward``.
    """
    was_training = module.training
    module.eval()
    try:
        with torch.no_grad():
            feats = module(torch.zeros(1, *input_shape))
    finally:
        module.train(was_training)
    return int(feats.shape[1])
