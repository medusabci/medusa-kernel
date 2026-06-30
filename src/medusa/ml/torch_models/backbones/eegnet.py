"""EEGNet architecture (Lawhern et al., 2018).

A pure, headless ``nn.Module`` feature extractor: :meth:`forward` returns the
flattened feature vector with **no classification head and no softmax**.
Classification heads live on the estimators
(:mod:`medusa.ml.torch_models.classification`).

Original source: https://github.com/vlawhern/arl-eegmodels
"""
from __future__ import annotations

import torch.nn as nn

from ._utils import infer_feature_dim, to_conv_input

__all__ = ['EEGNet']


class EEGNet(nn.Module):
    """Compact convolutional backbone (EEGNet-8,2 with defaults).

    Implements the current EEGNet (not the earlier arxiv v1/v2). Wrap it in a
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` to attach
    a head.

    Parameters
    ----------
    n_cha : int, optional
        Number of EEG channels. Default ``64``.
    samples : int, optional
        Number of temporal samples per epoch. Default ``128``.
    dropout_rate : float, optional
        Dropout fraction. Default ``0.5``.
    kern_length : int, optional
        Length of the temporal convolution kernel in block 1. Default ``64``.
    F1 : int, optional
        Number of temporal filters (first conv layer). Default ``8``.
    D : int, optional
        Spatial filters per temporal filter (depth multiplier). Default ``2``.
    F2 : int, optional
        Pointwise filters in the separable conv (typically ``F1 * D``).
        Default ``16``.
    dropout_type : {'Dropout', 'SpatialDropout2D'}, optional
        Dropout layer type. ``'SpatialDropout2D'`` maps to ``nn.Dropout2d``.
        Default ``'Dropout'``.

    Attributes
    ----------
    backbone_features : int
        Dimensionality of the feature vector returned by :meth:`forward`.
    input_layout : tuple of str
        Axis names of the tensor expected by :meth:`forward`.

    References
    ----------
    [1] Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung,
        C. P., & Lance, B. J. (2018). EEGNet: a compact convolutional neural
        network for EEG-based brain–computer interfaces. Journal of Neural
        Engineering, 15(5), 056013.
    """

    #: Axis names of the input tensor expected by ``forward``.
    input_layout = ('batch', 'feature', 'n_samples', 'n_channels')

    def __init__(self, n_cha: int = 64, samples: int = 128,
                 dropout_rate: float = 0.5, kern_length: int = 64,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 dropout_type: str = 'Dropout'):
        super().__init__()

        self.n_cha = n_cha
        self.samples = samples
        self.dropout_rate = dropout_rate
        self.kern_length = kern_length
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.dropout_type = dropout_type

        if dropout_type == 'SpatialDropout2D':
            dropout_layer = nn.Dropout2d
        elif dropout_type == 'Dropout':
            dropout_layer = nn.Dropout
        else:
            raise ValueError(
                "dropout_type must be one of 'SpatialDropout2D' or 'Dropout'.")

        # Block 1: temporal convolution + depthwise spatial convolution.
        # Input layout: (batch, 1, n_samples, n_channels).
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(kern_length, 1),
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(1, n_cha),
                                   bias=False, groups=F1)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool2d(kernel_size=(4, 1))
        self.dropout1 = dropout_layer(dropout_rate)

        # Block 2: separable convolution (depthwise + pointwise).
        self.depthwise2 = nn.Conv2d(F1 * D, F1 * D, kernel_size=(16, 1),
                                    padding='same', groups=F1 * D, bias=False)
        self.pointwise2 = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool2d(kernel_size=(8, 1))
        self.dropout2 = dropout_layer(dropout_rate)

        # Backbone output: flatten (no head).
        self.flatten = nn.Flatten()

        # Feature width is probed from the actual layers (one dummy forward)
        # so it never desyncs under kernel/pooling changes.
        self.backbone_features = infer_feature_dim(self, (1, samples, n_cha))

    def prepare_X(self, X):
        """EEG epochs ``(n_epochs, n_samples, n_channels)`` → forward tensor.

        Input adaptation the estimator calls before :meth:`forward`: adds the
        singleton ``feature`` axis this 2-D-conv backbone needs, yielding
        ``(n_epochs, 1, n_samples, n_channels)``. Deterministic and
        parameter-free (no learnable preprocessing).
        """
        return to_conv_input(X, self.samples, self.n_cha)

    def forward(self, x):
        """Extract features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, 1, n_samples, n_channels)`` (as produced
            by :meth:`prepare_X`).

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(batch, backbone_features)``.
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return self.flatten(x)

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild this module (for ML6)."""
        return {
            'n_cha': self.n_cha,
            'samples': self.samples,
            'dropout_rate': self.dropout_rate,
            'kern_length': self.kern_length,
            'F1': self.F1,
            'D': self.D,
            'F2': self.F2,
            'dropout_type': self.dropout_type,
        }
