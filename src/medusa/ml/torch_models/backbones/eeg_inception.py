"""EEG-Inception architecture (Santamaría-Vázquez et al., 2020).

A pure, headless ``nn.Module`` feature extractor: :meth:`forward` returns the
flattened feature vector with **no classification head and no softmax**.
Classification heads live on the estimators
(:mod:`medusa.ml.torch_models.classification`), which append an
``nn.Linear(backbone_features, n_classes)`` and trains it with
``CrossEntropyLoss`` on class indices — this is what removes the historical
double-``log_softmax`` bug.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ._utils import infer_feature_dim, to_conv_input

__all__ = ['EEGInception']


class EEGInception(nn.Module):
    """EEG-Inception convolutional backbone for EEG classification.

    Wrap it in a
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` to attach
    a head: ``TorchClassifier(backbone=EEGInception(...))``.

    Parameters
    ----------
    input_samples : int
        Number of temporal samples per epoch.
    n_cha : int
        Number of EEG channels.
    scales_samples : tuple of int, optional
        Temporal kernel sizes (in samples) of the multi-scale inception
        branches. Default ``(64, 32, 16)`` (≈ 500/250/125 ms at 128 Hz).
    filters_per_branch : int, optional
        Convolutional filters per inception branch. Default ``8``.
    dropout_rate : float, optional
        Dropout probability. Default ``0.25``.

    Attributes
    ----------
    backbone_features : int
        Dimensionality of the feature vector returned by :meth:`forward`.
    input_layout : tuple of str
        Axis names of the tensor expected by :meth:`forward`.

    References
    ----------
    [1] Santamaría-Vázquez, E., Martínez-Cagigal, V., Vaquerizo-Villar, F., &
        Hornero, R. (2020). EEG-Inception: A Novel Deep Convolutional Neural
        Network for Assistive ERP-based Brain-Computer Interfaces. IEEE
        Transactions on Neural Systems and Rehabilitation Engineering.
    """

    #: Axis names of the input tensor expected by ``forward``.
    input_layout = ('batch', 'feature', 'n_samples', 'n_channels')

    def __init__(self, input_samples: int, n_cha: int,
                 scales_samples=(64, 32, 16), filters_per_branch: int = 8,
                 dropout_rate: float = 0.25):
        super().__init__()

        self.input_samples = input_samples
        self.n_cha = n_cha
        self.scales_samples = tuple(scales_samples)
        self.filters_per_branch = filters_per_branch
        self.dropout_rate = dropout_rate

        n_scales = len(self.scales_samples)

        # Block 1: Single-channel temporal analysis (multi-scale inception).
        self.block1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, filters_per_branch, (scale, 1), padding='same'),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(),
                nn.Dropout(dropout_rate),
            ) for scale in self.scales_samples
        ])
        self.pool1 = nn.AvgPool2d((2, 1))

        # Block 2: Spatial filtering (depthwise across all channels).
        self.block2 = nn.Sequential(
            nn.Conv2d(filters_per_branch * n_scales,
                      filters_per_branch * n_scales * 2,
                      (1, n_cha),
                      groups=filters_per_branch * n_scales,
                      bias=False),
            nn.BatchNorm2d(filters_per_branch * n_scales * 2),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )
        self.pool2 = nn.AvgPool2d((2, 1))

        # Block 3: Multi-channel temporal analysis (multi-scale inception).
        self.block3 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(filters_per_branch * n_scales * 2,
                          filters_per_branch, (scale // 4, 1),
                          padding='same', bias=False),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(),
                nn.Dropout(dropout_rate),
            ) for scale in self.scales_samples
        ])
        self.pool3 = nn.AvgPool2d((2, 1))

        # Block 4: Output block (progressive temporal compression).
        self.block4_1 = nn.Sequential(
            nn.Conv2d(filters_per_branch * n_scales,
                      int(filters_per_branch * n_scales / 2),
                      (8, 1), padding='same', bias=False),
            nn.BatchNorm2d(int(filters_per_branch * n_scales / 2)),
            nn.ELU(),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )
        self.block4_2 = nn.Sequential(
            nn.Conv2d(int(filters_per_branch * n_scales / 2),
                      int(filters_per_branch * n_scales / 4),
                      (4, 1), padding='same', bias=False),
            nn.BatchNorm2d(int(filters_per_branch * n_scales / 4)),
            nn.ELU(),
            nn.AvgPool2d((2, 1)),
            nn.Dropout(dropout_rate),
        )

        # Backbone output: flatten the conv feature map (no head).
        self.flatten = nn.Flatten()

        # Feature width is probed from the actual layers (one dummy forward)
        # so it never desyncs under kernel/pooling changes.
        self.backbone_features = infer_feature_dim(self, (1, input_samples, n_cha))

    def prepare_X(self, X):
        """EEG epochs ``(n_epochs, n_samples, n_channels)`` → forward tensor.

        Input adaptation the estimator calls before :meth:`forward`: adds the
        singleton ``feature`` axis this 2-D-conv backbone needs, yielding
        ``(n_epochs, 1, n_samples, n_channels)``. Deterministic and
        parameter-free (no learnable preprocessing).
        """
        return to_conv_input(X, self.input_samples, self.n_cha)

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
        b1_out = torch.cat([unit(x) for unit in self.block1], dim=1)
        b1_out = self.pool1(b1_out)

        # Block 2
        b2_out = self.block2(b1_out)
        b2_out = self.pool2(b2_out)

        # Block 3
        b3_out = torch.cat([unit(b2_out) for unit in self.block3], dim=1)
        b3_out = self.pool3(b3_out)

        # Block 4
        b4_out = self.block4_1(b3_out)
        b4_out = self.block4_2(b4_out)

        return self.flatten(b4_out)

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild this module (for ML6)."""
        return {
            'input_samples': self.input_samples,
            'n_cha': self.n_cha,
            'scales_samples': self.scales_samples,
            'filters_per_branch': self.filters_per_branch,
            'dropout_rate': self.dropout_rate,
        }
