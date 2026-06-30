"""Minimal train/test of EEGInceptionV2 via the medusa.ml estimator API.

Run:  python examples/train_eeg_inception_v2.py
Needs a user-installed PyTorch + Lightning (see medusa.ml.torch_models).
"""
import numpy as np
from sklearn.model_selection import train_test_split

from medusa.signal.segmentation import segment_signal
from medusa.signal.generators import EEGSignalGenerator
from medusa.ml.torch_models.backbones import EEGInceptionV2
from medusa.ml.torch_models.classification import (TorchClassifier,
                                                   TorchMultiTaskClassifier)


gen = EEGSignalGenerator(fs=250)
eeg = gen.get_chunk(duration=10, n_channels=8)
# Natural EEG epochs: (n_epochs, n_samples, n_channels). The backbone's
# prepare_X formats them into its forward tensor — you pass your data as-is.
X = segment_signal(eeg, segment_length=250, stride=25)
y = np.random.randint(0, 2, size=X.shape[0])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0)

# Headless backbone + a classifier head, trained with PyTorch Lightning. The
# head is sized from np.unique(y) at fit time — no n_classes to declare.
backbone = EEGInceptionV2(input_samples=250, n_cha=8)
clf = TorchClassifier(
    backbone,
    lr=1e-3,
    max_epochs=50,
    batch_size=32,
    val_split=0.2,     # early-stops on a held-out split, restores best weights
    device='auto')     # GPU if available, else CPU

# Train.
clf.fit(X_tr, y_tr)
print(f"train history : {clf.history_}")

# Test.
print(f"test accuracy : {clf.score(X_te, y_te):.3f}")


# ---------------------------------------------------------------------------- #
# Multi-task classification (same X, several labellings)
# ---------------------------------------------------------------------------- #
# Share one backbone across several tasks (e.g. predict the stimulus class AND
# the subject's attentional state from the same EEG segments). Each task gets
# an independent linear head on the shared features; the backbone is updated
# by the sum of the per-task cross-entropy losses.

# Synthetic second task: a 3-class label aligned with the same X segments.
y_task_b = np.random.randint(0, 3, size=X.shape[0])

X_tr, X_te, y_a_tr, y_a_te, y_b_tr, y_b_te = train_test_split(
    X, y, y_task_b, test_size=0.25, stratify=y, random_state=0)

# Fresh backbone shared between tasks; one head per entry in the y dict.
backbone_mt = EEGInceptionV2(input_samples=250, n_cha=8)
mt_clf = TorchMultiTaskClassifier(
    backbone_mt,
    lr=1e-3,
    max_epochs=50,
    batch_size=32,
    val_split=0.2,
    device='auto')

# Train both heads jointly on the same X.
mt_clf.fit(X_tr, {'task_a': y_a_tr, 'task_b': y_b_tr})
print(f"multitask history : {mt_clf.history_}")

# Per-task inference: name the task explicitly.
scores = mt_clf.score(X_te, {'task_a': y_a_te, 'task_b': y_b_te})
print(f"task_a test accuracy : {scores['task_a']:.3f}")
print(f"task_b test accuracy : {scores['task_b']:.3f}")


# ---------------------------------------------------------------------------- #
# Multi-task classification (one experiment per task, different data)
# ---------------------------------------------------------------------------- #
# Train one shared backbone across several experiments at once — e.g. an ERP
# backbone trained jointly on exp1 (target / non-target) and exp2 (target /
# non-target / distractor). Each experiment is its own dataset with its own
# head and class count; they only need to share the per-epoch shape (channels
# x samples). Uneven epoch counts are handled by cycling the smaller dataset.

eeg2 = EEGSignalGenerator(fs=250).get_chunk(duration=8, n_channels=8)
X_exp2 = segment_signal(eeg2, segment_length=250, stride=25)
y_exp1 = np.random.randint(0, 2, size=X.shape[0])        # 2 classes
y_exp2 = np.random.randint(0, 3, size=X_exp2.shape[0])   # 3 classes

backbone_erp = EEGInceptionV2(input_samples=250, n_cha=8)
erp_clf = TorchMultiTaskClassifier(
    backbone_erp,
    lr=1e-3,
    max_epochs=50,
    batch_size=32,
    val_split=0.2,
    device='auto')

# X is a dict here — one array per experiment, aligned with the y dict keys.
erp_clf.fit({'exp1': X, 'exp2': X_exp2},
            {'exp1': y_exp1, 'exp2': y_exp2})
print(f"cross-experiment history : {erp_clf.history_}")
print(f"exp1 classes : {erp_clf.classes_['exp1']}  "
      f"exp2 classes : {erp_clf.classes_['exp2']}")
