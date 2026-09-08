"""Machine-learning layer for medusa-kernel.

The kernel *applies* models inside processing pipelines (model development —
hyper-parameter search, self-supervised pretraining — happens outside the
kernel). The deep-learning estimators and backbones live under the
:mod:`medusa.ml.torch_models` subpackage, which is **torch-gated**: it needs a
user-installed PyTorch (and, for the estimator layer, PyTorch-Lightning), whose
build depends on the local CUDA/ROCm/MPS/CPU stack. medusa-kernel does not ship
those as a dependency.

To keep ``import medusa.ml`` (and a headless DSP / server install) free of a
PyTorch requirement, the torch backends are **not** imported here — pull them in
on demand::

    from medusa.ml.torch_models.backbones import EEGInceptionV2
    from medusa.ml.torch_models.classification import TorchClassifier

There is nothing torch-free to re-export at this top level yet, so the package
exposes no public names of its own.
"""

# Intentionally empty: do NOT import ``medusa.ml.torch_models`` here — it pulls
# torch, which would break the headless / no-torch import contract for
# ``medusa.ml``. Torch backends are imported on demand from their submodules.
__all__: list[str] = []
