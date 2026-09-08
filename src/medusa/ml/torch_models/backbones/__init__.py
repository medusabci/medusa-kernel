"""EEG backbone architectures — headless ``nn.Module`` feature extractors.

Each backbone's ``forward(x)`` returns a feature vector (no classification head,
no softmax) and exposes ``backbone_features: int``. Wrap one in
``medusa.ml.torch_models.classification.TorchClassifier`` to attach a head.

Requires PyTorch (the whole ``medusa.ml.torch`` subpackage is torch-gated).
"""
from .eeg_inception import EEGInception
from .eeg_inception_v2 import EEGInceptionV2
from .eegsym import EEGSym

__all__ = ['EEGInception', 'EEGInceptionV2', 'EEGSym']
