"""EEG-Inception V2 — hybrid standard + dilated inception backbone.

A pure, headless ``nn.Module`` feature extractor (the v2.0 successor to the
former ``EEGInceptionGen``). :meth:`forward` returns the feature vector with
**no classification head and no softmax**; heads live on the estimators
(:mod:`medusa.ml.torch_models.classification`), so a single pretrained
backbone is reused across tasks (multi-task / foundation-model lifecycle).

Block 1 uses standard large-kernel inception modules to extract multi-scale
temporal features from the raw signal. After spatial filtering (Block 2),
Block 3 switches to dilated inception (small kernels, varying dilation),
capturing diverse temporal phenomena with far fewer parameters. Block 4
progressively compresses the temporal axis before a global average pool.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ._utils import infer_feature_dim, to_conv_input

__all__ = ['EEGInceptionV2']


class EEGInceptionV2(nn.Module):
    """Hybrid (standard + dilated) inception backbone for EEG.

    Wrap it in a
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` (or a
    :class:`~medusa.ml.torch_models.classification.TorchMultiTaskClassifier`
    for several heads) to obtain a usable classifier.

    Parameters
    ----------
    input_samples : int
        Number of temporal samples per epoch.
    n_cha : int
        Number of EEG channels.
    n_temp_inc_blocks : int, optional
        Stacked standard inception modules in Block 1. Default ``1``.
    temp_filt_per_branch : int, optional
        Filters per branch in Block 1. Default ``8``.
    temp_scales_samples : tuple of int, optional
        Temporal kernel sizes (samples) for Block 1. Default ``(100, 75, 50)``.
    dil_filt_per_branch : int, optional
        Filters per branch in Block 3 (dilated). Default ``8``.
    dil_branch_specs : tuple of (int, int), optional
        ``(kernel_size, dilation_rate)`` pairs for Block 3. Kernel sizes must
        be odd. Default ``((5, 1), (5, 5), (5, 10), (5, 15))``.
    n_dil_inc_blocks : int, optional
        Stacked dilated inception modules in Block 3. Default ``1``.
    n_spatial_filt_mult : int, optional
        Spatial filter multiplier in Block 2. Default ``2``.
    output_pooling_factor : int, optional
        Temporal pooling factor per Block 4 layer. Default ``2``.
    dropout_type : {'Dropout', 'SpatialDropout2D'}, optional
        Dropout layer type. Default ``'Dropout'``.
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
    [1] Santamaría-Vázquez, E., et al. (2020). EEG-Inception: A Novel Deep
        Convolutional Neural Network for Assistive ERP-based Brain-Computer
        Interfaces. IEEE TNSRE.
    [2] Yu, F. & Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated
        Convolutions. ICLR.
    """

    #: Axis names of the input tensor expected by ``forward``.
    input_layout = ('batch', 'feature', 'n_samples', 'n_channels')

    def __init__(self, input_samples: int, n_cha: int,
                 n_temp_inc_blocks: int = 1,
                 temp_filt_per_branch: int = 8,
                 temp_scales_samples=(100, 75, 50),
                 dil_filt_per_branch: int = 8,
                 dil_branch_specs=((5, 1), (5, 5), (5, 10), (5, 15)),
                 n_dil_inc_blocks: int = 1,
                 n_spatial_filt_mult: int = 2,
                 output_pooling_factor: int = 2,
                 dropout_type: str = 'Dropout',
                 dropout_rate: float = 0.25):
        super().__init__()

        # ---- validation ---- #
        if dropout_type not in ('Dropout', 'SpatialDropout2D'):
            raise ValueError(
                "dropout_type must be one of 'Dropout', 'SpatialDropout2D'.")
        if not dil_branch_specs:
            raise ValueError(
                "dil_branch_specs must be a non-empty tuple of "
                "(kernel_size, dilation_rate) pairs.")
        dil_branch_specs = tuple(tuple(b) for b in dil_branch_specs)
        for k, _ in dil_branch_specs:
            if k % 2 == 0:
                raise ValueError(
                    f"All kernel sizes must be odd for symmetric padding, "
                    f"got kernel_size={k} in dil_branch_specs.")
        if not temp_scales_samples:
            raise ValueError(
                "temp_scales_samples must be a non-empty tuple of kernel "
                "sizes (in samples).")

        # ---- store config (verbatim, for get_config / rebuild) ---- #
        self.input_samples = input_samples
        self.n_cha = n_cha
        self.n_temp_inc_blocks = n_temp_inc_blocks
        self.temp_filt_per_branch = temp_filt_per_branch
        self.temp_scales_samples = tuple(temp_scales_samples)
        self.dil_filt_per_branch = dil_filt_per_branch
        self.dil_branch_specs = dil_branch_specs
        self.n_dil_inc_blocks = n_dil_inc_blocks
        self.n_spatial_filt_mult = n_spatial_filt_mult
        self.output_pooling_factor = output_pooling_factor
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate

        n_std_branches = len(self.temp_scales_samples)
        n_dil_branches = len(dil_branch_specs)

        # Select activation function.
        act_fn = nn.ELU

        # Select dropout type.
        dropout_cls = (nn.Dropout2d if dropout_type == 'SpatialDropout2D'
                       else nn.Dropout)

        # -------- builder helpers -------- #
        def get_scales(pooling_track):
            pooling_factor = 1
            for p in pooling_track:
                pooling_factor *= p
            return [max(int(sc // pooling_factor), 1)
                    for sc in self.temp_scales_samples]

        def adjust_specs_for_pooling(specs, pooling_track):
            # Rounding (not floor) preserves branch diversity: d=3 → d=2 after
            # pool=2, instead of collapsing to d=1.
            cumulative_pool = 1
            for p in pooling_track:
                cumulative_pool *= p
            return [(k, max(round(d / cumulative_pool), 1)) for k, d in specs]

        def make_conv_block(in_ch, out_ch, kernel_size):
            # Bias disabled: BatchNorm absorbs it.
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size, padding='same',
                          bias=False),
                nn.BatchNorm2d(out_ch),
                act_fn(),
            )

        def make_dilated_conv_block(in_ch, out_ch, kernel_size, dilation):
            pad = ((kernel_size - 1) * dilation) // 2
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(kernel_size, 1),
                          dilation=(dilation, 1), padding=(pad, 0),
                          bias=False),
                nn.BatchNorm2d(out_ch),
                act_fn(),
            )

        def make_inception_module(in_ch, scales):
            return nn.ModuleList([
                make_conv_block(in_ch, temp_filt_per_branch, (scale, 1))
                for scale in scales
            ])

        def make_dilated_inception(in_ch, specs):
            return nn.ModuleList([
                make_dilated_conv_block(in_ch, dil_filt_per_branch, k, d)
                for k, d in specs
            ])

        pooling_track = []

        # ---- Block 1: standard temporal inception ---- #
        scales = get_scales(pooling_track)
        self.block1_inception = nn.ModuleList()
        in_ch = 1
        for _ in range(n_temp_inc_blocks):
            self.block1_inception.append(make_inception_module(in_ch, scales))
            in_ch = temp_filt_per_branch * n_std_branches
        self.pool1 = nn.AvgPool2d((2, 1))
        pooling_track.append(2)
        self.drop1 = dropout_cls(dropout_rate)

        # ---- Block 2: channel selection (depthwise) ---- #
        b1_out_ch = temp_filt_per_branch * n_std_branches
        b2_out_ch = b1_out_ch * n_spatial_filt_mult
        self.block2_dw = nn.Conv2d(b1_out_ch, b2_out_ch,
                                   kernel_size=(1, n_cha), groups=b1_out_ch,
                                   bias=False)
        self.block2_bn = nn.BatchNorm2d(b2_out_ch)
        self.block2_act = act_fn()
        self.drop2 = dropout_cls(dropout_rate)

        # ---- Block 3: dilated spatio-temporal inception ---- #
        specs = adjust_specs_for_pooling(dil_branch_specs, pooling_track)
        self.block3_inception = nn.ModuleList()
        in_ch = b2_out_ch
        for _ in range(n_dil_inc_blocks):
            self.block3_inception.append(make_dilated_inception(in_ch, specs))
            in_ch = dil_filt_per_branch * n_dil_branches
        self.pool3 = nn.AvgPool2d((2, 1))
        pooling_track.append(2)
        self.drop3 = dropout_cls(dropout_rate)

        # ---- Block 4: output block (progressive temporal compression) ---- #
        input_samples_to_out_block = input_samples // math.prod(pooling_track)
        n_out_blocks = math.log(max(input_samples_to_out_block, 2),
                                output_pooling_factor)
        n_out_blocks = max(math.ceil(n_out_blocks) - 1, 1)
        n_out_filt_increase_ratio = 4

        n_out_filt = dil_filt_per_branch * n_dil_branches
        self.block4_layers = nn.ModuleList()
        self.block4_pools = nn.ModuleList()
        prev_ch = dil_filt_per_branch * n_dil_branches

        def get_out_kernel(pooling_track_):
            remaining = input_samples // math.prod(pooling_track_)
            return max(min(remaining // 2, 7), 1)

        filt_increase = int(math.ceil(input_samples_to_out_block /
                                      output_pooling_factor /
                                      n_out_filt_increase_ratio))
        filt_increase = max(filt_increase, n_out_filt_increase_ratio)
        n_out_filt += filt_increase
        out_k = get_out_kernel(pooling_track)
        self.block4_layers.append(make_conv_block(prev_ch, n_out_filt,
                                                  (out_k, 1)))
        self.block4_pools.append(nn.AvgPool2d((output_pooling_factor, 1)))
        pooling_track.append(output_pooling_factor)
        prev_ch = n_out_filt

        for _ in range(n_out_blocks - 1):
            filt_increase = max(int(math.ceil(filt_increase /
                                              output_pooling_factor)),
                                n_out_filt_increase_ratio)
            n_out_filt += filt_increase
            out_k = get_out_kernel(pooling_track)
            self.block4_layers.append(make_conv_block(prev_ch, n_out_filt,
                                                      (out_k, 1)))
            self.block4_pools.append(nn.AvgPool2d((output_pooling_factor, 1)))
            pooling_track.append(output_pooling_factor)
            prev_ch = n_out_filt

        # ---- backbone output ---- #
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_out = nn.Dropout(dropout_rate)

        # Feature width is probed from the actual layers (one dummy forward).
        # The GAP already makes this the final channel count, but probing keeps
        # the contract uniform across architectures.
        self.backbone_features = infer_feature_dim(self, (1, input_samples, n_cha))

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 'relu' is the standard approximation for ELU-family
                # activations (PyTorch has no 'elu' mode).
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def _apply_inception(x, inception_modules):
        out = x
        for inception in inception_modules:
            out = torch.cat([branch(out) for branch in inception], dim=1)
        return out

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
        # Block 1: standard temporal inception
        b1 = self._apply_inception(x, self.block1_inception)
        b1 = self.drop1(self.pool1(b1))

        # Block 2: channel selection (depthwise conv)
        b2 = self.block2_dw(b1)
        b2 = self.block2_act(self.block2_bn(b2))
        b2 = self.drop2(b2)

        # Block 3: dilated spatio-temporal inception
        b3 = self._apply_inception(b2, self.block3_inception)
        b3 = self.drop3(self.pool3(b3))

        # Block 4: output block
        b4 = b3
        for conv_block, pool in zip(self.block4_layers, self.block4_pools):
            b4 = pool(conv_block(b4))

        out = self.gap(b4)
        out = out.view(out.size(0), -1)
        return self.drop_out(out)

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild this module (for ML6)."""
        return {
            'input_samples': self.input_samples,
            'n_cha': self.n_cha,
            'n_temp_inc_blocks': self.n_temp_inc_blocks,
            'temp_filt_per_branch': self.temp_filt_per_branch,
            'temp_scales_samples': self.temp_scales_samples,
            'dil_filt_per_branch': self.dil_filt_per_branch,
            'dil_branch_specs': self.dil_branch_specs,
            'n_dil_inc_blocks': self.n_dil_inc_blocks,
            'n_spatial_filt_mult': self.n_spatial_filt_mult,
            'output_pooling_factor': self.output_pooling_factor,
            'dropout_type': self.dropout_type,
            'dropout_rate': self.dropout_rate,
        }
