"""EEGSym — bilateral inception backbone for EEG (Pérez-Velasco et al., 2022).

A pure, headless ``nn.Module`` feature extractor following the same contract as
the other backbones: :meth:`forward` returns a feature vector with **no
classification head and no softmax** (heads live on the estimators in
:mod:`medusa.ml.torch_models.classification`), and the module exposes
``backbone_features``, ``input_layout``, ``prepare_X`` and ``get_config``.

EEGSym is unique among the backbones in that its very first step is a
*bilateral* rearrangement: channels are split into a left and a right
hemisphere (with the midline channels duplicated into both), stacked along a
new ``hemisphere`` axis, and processed by 3-D convolutions whose inductive bias
is left/right symmetry. That split is intrinsic to the architecture, so it
lives inside :meth:`forward` (using channel-index buffers), and :meth:`forward`
accepts a plain channel-major tensor ``(batch, n_channels, n_samples)``.

The hemisphere layout is supplied **explicitly** at construction
(``left_right_chs`` + ``middle_chs`` over a given ``ch_names`` ordering); the
backbone never tries to infer hemispheres from channel naming.

Odd-length kernels are used throughout so padding stays symmetric.

References
----------
[1] Pérez-Velasco, S., Santamaría-Vázquez, E., Martínez-Cagigal, V.,
    Marcos-Mateo, D., & Hornero, R. (2022). EEGSym: Overcoming Inter-Subject
    Variability in Motor Imagery Based BCIs With Deep Learning. IEEE TNSRE.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from ._utils import infer_feature_dim

# Only the backbone is public; the InceptionBlock/ResidualBlock/Temporal*/
# ChannelMerging*/Output* classes below are internal building blocks of EEGSym.
__all__ = ['EEGSym']


def _activation_cls(name: str) -> type[nn.Module]:
    """Map an activation name to its ``nn.Module`` class."""
    table = {'elu': nn.ELU, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU}
    if name not in table:
        raise ValueError(
            f"Unsupported activation: {name!r}; choose one of "
            f"{sorted(table)}.")
    return table[name]


class EEGSym(nn.Module):
    """Bilateral inception backbone for EEG.

    Wrap it in a
    :class:`~medusa.ml.torch_models.classification.TorchClassifier` (or a
    :class:`~medusa.ml.torch_models.classification.TorchMultiTaskClassifier`)
    to obtain a usable classifier.

    Parameters
    ----------
    input_samples : int
        Number of temporal samples per epoch (``n_times``). Must be at least
        64, since the architecture pools the temporal axis by ``2**6`` before
        the temporal-merging block.
    fs : float
        Sampling rate (Hz). Used to convert ``scales_time`` (ms) into kernel
        sizes in samples.
    ch_names : list of str
        Channel names in the order of the input array's channel axis. Sizes the
        input (``n_cha = len(ch_names)``) and resolves the hemisphere layout
        below.
    left_right_chs : list of (str, str)
        Mirror channel pairs ``(left_name, right_name)`` defining the two
        hemispheres (e.g. ``[('C3', 'C4'), ('FC5', 'FC6')]``). Each name must
        appear in ``ch_names``.
    middle_chs : list of str
        Midline channel names (e.g. ``['Cz', 'Pz']``), duplicated into both
        hemispheres. May be empty.
    filters_per_branch : int, optional
        Filters per inception branch. Default ``24`` (paper, 8-electrode).
    scales_time : tuple of int, optional
        Temporal kernel sizes in **milliseconds** for the first inception
        block. Default ``(500, 250, 125)``.
    drop_prob : float, optional
        Dropout probability. Default ``0.4``.
    activation : {'elu', 'relu', 'leaky_relu'}, optional
        Activation function. Default ``'elu'``.
    spatial_resnet_repetitions : int, optional
        Spatial residual repetitions per inception/residual block. Default
        ``1``.

    Attributes
    ----------
    backbone_features : int
        Dimensionality of the feature vector returned by :meth:`forward`.
    input_layout : tuple of str
        Axis names of the tensor expected by :meth:`forward`.
    """

    #: Axis names of the input tensor expected by ``forward``. The bilateral
    #: split happens inside ``forward``, so the input is plain channel-major.
    input_layout = ('batch', 'n_channels', 'n_samples')

    def __init__(self, input_samples: int, fs: float, ch_names: List[str],
                 left_right_chs: List[Tuple[str, str]],
                 middle_chs: List[str], *,
                 filters_per_branch: int = 24,
                 scales_time: Tuple[int, ...] = (500, 250, 125),
                 drop_prob: float = 0.4, activation: str = 'elu',
                 spatial_resnet_repetitions: int = 1):
        super().__init__()

        # ---- validation ---- #
        if input_samples < 64:
            raise ValueError(
                f"input_samples must be >= 64 (the temporal axis is pooled by "
                f"2**6 before temporal merging); got {input_samples}.")
        if not left_right_chs:
            raise ValueError(
                "left_right_chs must be a non-empty list of (left, right) "
                "channel-name pairs defining the two hemispheres.")
        ch_names = list(ch_names)
        left_right_chs = [tuple(pair) for pair in left_right_chs]
        middle_chs = list(middle_chs)
        for pair in left_right_chs:
            if len(pair) != 2:
                raise ValueError(
                    f"each left_right_chs entry must be a (left, right) pair; "
                    f"got {pair!r}.")
        # Resolve names -> indices into the input channel axis.
        index = {name: i for i, name in enumerate(ch_names)}
        missing = [c for c in (
            [c for pair in left_right_chs for c in pair] + middle_chs)
            if c not in index]
        if missing:
            raise ValueError(
                f"these hemisphere channels are not in ch_names: {missing}.")
        left_idx = [index[left] for left, _ in left_right_chs]
        right_idx = [index[right] for _, right in left_right_chs]
        middle_idx = [index[c] for c in middle_chs]

        # ---- store config (verbatim, for get_config / rebuild) ---- #
        self.input_samples = input_samples
        self.fs = fs
        self.ch_names = ch_names
        self.left_right_chs = left_right_chs
        self.middle_chs = middle_chs
        self.n_cha = len(ch_names)
        self.filters_per_branch = filters_per_branch
        self.scales_time = tuple(scales_time)
        self.drop_prob = drop_prob
        self.activation = activation
        self.spatial_resnet_repetitions = spatial_resnet_repetitions

        # Scales in samples, forced ODD for symmetric padding.
        self.scales_samples = []
        for s in self.scales_time:
            samples = int(s * fs / 1000)
            if samples % 2 == 0:
                samples += 1
            self.scales_samples.append(samples)

        total_filters = filters_per_branch * len(self.scales_samples)

        # Hemisphere channel indices as buffers: they are model state, so they
        # move with .to(device) and serialize in state_dict.
        self.register_buffer('left_idx', torch.tensor(left_idx,
                                                      dtype=torch.long))
        self.register_buffer('right_idx', torch.tensor(right_idx,
                                                       dtype=torch.long))
        self.register_buffer('middle_idx', torch.tensor(middle_idx,
                                                        dtype=torch.long))
        # Both hemispheres get the lateral channels of their side + the midline.
        self.n_channels_per_hemi = len(left_idx) + len(middle_idx)

        act = _activation_cls(activation)()

        # ---- build the network ---- #
        self.inception_block1 = InceptionBlock(
            in_channels=1, scales_samples=self.scales_samples,
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi, activation=act,
            drop_prob=self.drop_prob, average_pool=2,
            spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            init=True)

        self.inception_block2 = InceptionBlock(
            in_channels=total_filters,
            # Scales reduced by 4 and forced odd (no tensor cropping needed).
            scales_samples=[max(1, s // 4 if (s // 4) % 2 != 0 else s // 4 + 1)
                            for s in self.scales_samples],
            filters_per_branch=self.filters_per_branch,
            ncha=self.n_channels_per_hemi, activation=act,
            drop_prob=self.drop_prob, average_pool=2,
            spatial_resnet_repetitions=self.spatial_resnet_repetitions,
            init=False)

        self.residual_blocks = nn.Sequential(
            ResidualBlock(
                in_channels=total_filters, filters=total_filters // 2,
                kernel_size=17, ncha=self.n_channels_per_hemi, activation=act,
                drop_prob=self.drop_prob, average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions),
            ResidualBlock(
                in_channels=total_filters // 2, filters=total_filters // 2,
                kernel_size=9, ncha=self.n_channels_per_hemi, activation=act,
                drop_prob=self.drop_prob, average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions),
            ResidualBlock(
                in_channels=total_filters // 2, filters=total_filters // 4,
                kernel_size=5, ncha=self.n_channels_per_hemi, activation=act,
                drop_prob=self.drop_prob, average_pool=2,
                spatial_resnet_repetitions=self.spatial_resnet_repetitions))

        self.temporal_reduction = nn.Sequential(
            TemporalBlock(
                in_channels=total_filters // 4, filters=total_filters // 4,
                kernel_size=5, activation=act, drop_prob=self.drop_prob),
            nn.AvgPool3d(kernel_size=(1, 2, 1)))

        cm_filters = total_filters // 4
        self.channel_merging = ChannelMergingBlock(
            in_channels=cm_filters, filters=cm_filters, groups=cm_filters // 2,
            ncha=self.n_channels_per_hemi, division=2, activation=act,
            drop_prob=drop_prob)

        self.temporal_merging = TemporalMergingBlock(
            in_channels=cm_filters, filters=total_filters // 2,
            groups=total_filters // 4, n_times=self.input_samples,
            activation=act, drop_prob=self.drop_prob)

        self.output_blocks = nn.Sequential(
            OutputBlock(in_channels=total_filters // 2, activation=act,
                        drop_prob=self.drop_prob),
            nn.Flatten())

        # Feature width probed from the actual layers (one dummy forward).
        self.backbone_features = infer_feature_dim(
            self, (self.n_cha, self.input_samples))

    def prepare_X(self, X):
        """EEG epochs ``(n_epochs, n_samples, n_channels)`` → forward tensor.

        Input adaptation the estimator calls before :meth:`forward`: validates
        the natural per-epoch shape and transposes to the channel-major layout
        ``(n_epochs, n_channels, n_samples)`` this backbone's ``forward``
        expects. Deterministic and parameter-free (the bilateral split itself
        happens inside ``forward``).
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError(
                f"X must be a 3-D array (n_epochs, n_samples="
                f"{self.input_samples}, n_channels={self.n_cha}); got a "
                f"{X.ndim}-D array of shape {X.shape}.")
        if X.shape[1:] != (self.input_samples, self.n_cha):
            raise ValueError(
                f"each epoch must be (n_samples={self.input_samples}, "
                f"n_channels={self.n_cha}); got {tuple(X.shape[1:])}. Check "
                "the axis order (samples before channels) and that the data "
                "matches this backbone.")
        # (n_epochs, n_samples, n_cha) -> (n_epochs, n_cha, n_samples)
        X = np.ascontiguousarray(X.transpose(0, 2, 1))
        return torch.from_numpy(X)

    def forward(self, x):
        """Extract features.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, n_channels, n_samples)`` (as produced by
            :meth:`prepare_X`).

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(batch, backbone_features)``.
        """
        # Bilateral rearrangement: split into hemispheres (midline duplicated)
        # and stack them along a new Z axis, then order axes as
        # (batch, feature, Z, time, space) for the 3-D convolutions.
        x = x.unsqueeze(1)  # (batch, 1, n_channels, n_samples)
        left = torch.index_select(x, 2, self.left_idx)
        right = torch.index_select(x, 2, self.right_idx)
        mid = torch.index_select(x, 2, self.middle_idx)
        left_hemi = torch.cat([left, mid], dim=2)
        right_hemi = torch.cat([right, mid], dim=2)
        x = torch.stack([left_hemi, right_hemi], dim=2)
        # (batch, 1, Z, space, time) -> (batch, 1, Z, time, space)
        x = x.transpose(3, 4).contiguous()

        x = self.inception_block1(x)
        x = self.inception_block2(x)
        x = self.residual_blocks(x)
        x = self.temporal_reduction(x)
        x = self.channel_merging(x)
        x = self.temporal_merging(x)
        x = self.output_blocks(x)
        return x

    def get_config(self) -> dict:
        """Constructor kwargs needed to rebuild this module (for persistence)."""
        return {
            'input_samples': self.input_samples,
            'fs': self.fs,
            'ch_names': list(self.ch_names),
            'left_right_chs': [tuple(p) for p in self.left_right_chs],
            'middle_chs': list(self.middle_chs),
            'filters_per_branch': self.filters_per_branch,
            'scales_time': self.scales_time,
            'drop_prob': self.drop_prob,
            'activation': self.activation,
            'spatial_resnet_repetitions': self.spatial_resnet_repetitions,
        }


# --- BLOCKS IMPLEMENTATION ---

class InceptionBlock(nn.Module):
    """Inception block for EEGSym.

    Parallel temporal convolutions at different scales, concatenated, followed
    by depthwise spatial convolutions and residual connections.
    """

    def __init__(
            self,
            in_channels: int,
            scales_samples: List[int],
            filters_per_branch: int,
            ncha: int,
            activation: nn.Module,
            drop_prob: float,
            average_pool: int,
            spatial_resnet_repetitions: int,
            init: bool = False,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob
        self.average_pool = average_pool
        self.init = init

        # Temporal convolutions
        self.temporal_convs = nn.ModuleList()
        for scale in scales_samples:
            # odd kernel
            padding = (0, scale // 2, 0)
            self.temporal_convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters_per_branch,
                        kernel_size=(1, scale, 1),
                        padding=padding,
                        bias=False
                    ),
                    nn.BatchNorm3d(filters_per_branch),
                    activation,
                    nn.Dropout(drop_prob)
                )
            )

        total_filters = filters_per_branch * len(scales_samples)

        # Residual projection
        self.residual_proj = None
        if in_channels != total_filters:
            self.residual_proj = nn.Conv3d(
                in_channels=in_channels,
                out_channels=total_filters,
                kernel_size=(1, 1, 1),
                bias=False
            )

        # Spatial convolutions (Depthwise - Groups=Filters)
        self.spatial_convs = nn.ModuleList()
        if ncha > 1:
            for _ in range(spatial_resnet_repetitions):
                self.spatial_convs.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=total_filters,
                            out_channels=total_filters,
                            kernel_size=(1, 1, ncha),
                            padding=(0, 0, 0),
                            groups=total_filters,  # Depthwise
                            bias=False
                        ),
                        nn.BatchNorm3d(total_filters),
                        activation,
                        nn.Dropout(drop_prob)
                    )
                )

        self.pool = nn.AvgPool3d((1, average_pool, 1))

    def forward(self, x):
        # Concatenate branches
        branch_outs = [b(x) for b in self.temporal_convs]
        x_cat = torch.cat(branch_outs, dim=1)

        # 1x1 residual projection: when channel counts differ (e.g. the first
        # block maps 1 raw channel to total_filters), project the input so the
        # residual sum is well defined.
        res = x
        if self.residual_proj:
            res = self.residual_proj(res)
        x_out = x_cat + res

        # Pooling
        x_out = self.pool(x_out)

        # Spatial (always residual)
        for op in self.spatial_convs:
            x_out = x_out + op(x_out)

        return x_out


class ResidualBlock(nn.Module):
    """Residual block.

    A single temporal convolution followed by standard (non-depthwise) spatial
    convolutions, downsampling via average pooling.
    """

    def __init__(
            self,
            in_channels: int,
            filters: int,
            kernel_size: int,
            ncha: int,
            activation: nn.Module,
            drop_prob: float,
            average_pool: int,
            spatial_resnet_repetitions: int = 5
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # Make sure that kernel_size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
                bias=False
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob)
        )

        # Residual projection
        self.residual_proj = None
        if in_channels != filters:
            self.residual_proj = nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, 1, 1),
                bias=False
            )

        # Average pooling
        self.pool = nn.AvgPool3d(
            kernel_size=(1, average_pool, 1)
        )

        # Spatial convolutions (standard, not depthwise)
        self.spatial_convs = nn.ModuleList()
        if ncha > 1:
            for _ in range(spatial_resnet_repetitions):
                self.spatial_convs.append(
                    nn.Sequential(
                        nn.Conv3d(
                            in_channels=filters,
                            out_channels=filters,
                            kernel_size=(1, 1, ncha),  # Spatial convolution
                            padding=(0, 0, 0),
                            groups=1,
                            bias=False
                        ),
                        nn.BatchNorm3d(filters),
                        activation,
                        nn.Dropout(drop_prob),
                    )
                )

    def forward(self, x):
        x_res = self.temporal_conv(x)

        if self.residual_proj:
            x = self.residual_proj(x)

        x_out = x_res + x

        # Pooling
        x_out = self.pool(x_out)

        # Spatial (always residual)
        for op in self.spatial_convs:
            x_out = x_out + op(x_out)

        return x_out


class TemporalBlock(nn.Module):
    """Temporal reduction block: a temporal convolution with a residual."""

    def __init__(
            self,
            in_channels: int,
            filters: int,
            kernel_size: int,
            activation: nn.Module,
            drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # odd kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, kernel_size, 1),
                padding=(0, kernel_size // 2, 0),
                bias=False
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return x + self.conv(x)


class ChannelMergingBlock(nn.Module):
    """Merge the two hemisphere representations.

    Reduces the spatial dimension to 1, then merges the Z dimension (the two
    hemispheres) into one.
    """

    def __init__(
            self,
            in_channels: int,
            filters: int,
            groups: int,
            ncha: int,
            division: int,
            activation: nn.Module,
            drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # Two residual convolution iterations; each reduces spatial dim ncha->1
        self.residual_convs = nn.ModuleList()
        for _ in range(2):
            self.residual_convs.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=filters,
                        kernel_size=(division, 1, ncha),  # (Z, Time, Space)
                        padding=(0, 0, 0),  # Valid padding
                        bias=False,
                    ),
                    nn.BatchNorm3d(filters),
                    activation,
                    nn.Dropout(drop_prob),
                )
            )

        # Final grouped convolution; merges Z dimension: 2 -> 1
        self.grouped_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(division, 1, ncha),  # (Z, Time, Space)
                groups=groups,
                padding=(0, 0, 0),
                bias=False),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        for conv in self.residual_convs:
            x_res = conv(x)
            x = x + x_res

        x = self.grouped_conv(x)

        return x


class TemporalMergingBlock(nn.Module):
    """Collapse the temporal dimension into a single feature representation."""

    def __init__(
            self,
            in_channels: int,
            filters: int,
            groups: int,
            n_times: int,
            activation: nn.Module,
            drop_prob: float,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        # Temporal dim has been pooled by 2**6 = 64 up to this point.
        self.temporal_kernel = n_times // 64

        # Residual convolution (collapses the time dimension)
        self.residual_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=in_channels,  # Same channels for residual
                kernel_size=(1, self.temporal_kernel, 1),  # (Z, Time, Space)
                padding=(0, 0, 0),  # Valid padding - reduces time to 1
                bias=False
            ),
            nn.BatchNorm3d(in_channels),
            activation,
            nn.Dropout(drop_prob),
        )

        # Grouped convolution (collapses time, changes filter count)
        self.grouped_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=filters,
                kernel_size=(1, self.temporal_kernel, 1),  # (Z, Time, Space)
                groups=groups,
                padding=(0, 0, 0),
                bias=False
            ),
            nn.BatchNorm3d(filters),
            activation,
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        # Residual convolution
        x_res = self.residual_conv(x)
        x = x + x_res

        # Grouped convolution (reduces time to 1)
        x = self.grouped_conv(x)

        return x


class OutputBlock(nn.Module):
    """Final 1x1x1 convolutions for feature refinement before the head."""

    def __init__(
            self,
            in_channels: int,
            activation: nn.Module,
            drop_prob: float,
            n_residual: int = 4,
    ):
        super().__init__()
        self.activation = activation
        self.drop_prob = drop_prob

        self.conv_blocks = nn.ModuleList()
        for _ in range(n_residual):
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=(1, 1, 1),
                        padding=(0, 0, 0),
                        bias=False,
                    ),
                    nn.BatchNorm3d(in_channels),
                    activation,
                    nn.Dropout(drop_prob),
                )
            )

    def forward(self, x):
        for conv_block in self.conv_blocks:
            x_res = conv_block(x)
            x = x + x_res
        return x
