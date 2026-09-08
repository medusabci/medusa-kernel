"""How ``random_state`` makes a deep training run repeatable.

Training draws random numbers in five places -- the backbone's initial weights, the
head's, the validation split, the batch order and the dropout masks -- and a run only
repeats if every one of them is fixed. In a pipeline that is **one setting**, because the
pipeline builds the backbone itself and so can seed that too. Going around the pipeline,
you seed the construction yourself; the last two sections show why.

Run:  python examples/reproducible_training.py
Needs a user-installed PyTorch + Lightning (see medusa.ml.torch_models).
"""
import numpy as np
import torch

from medusa.core.data import BidsInfo, ChannelSet, Events, Recording, Signal
from medusa.ml.torch_models._engine import seeded_rng
from medusa.ml.torch_models.backbones import EEGInceptionV2
from medusa.ml.torch_models.classification import TorchClassifier
from medusa.pipelines.bci.motor_decoding import MIEEGInceptionPipeline
from medusa.signal.generators import EEGSignalGenerator

FS, CHANNELS = 250.0, ["C3", "CZ", "C4"]
#: 'cpu' because CUDA kernels are not deterministic by default: on a GPU two seeded runs
#: come out very close, but not bit-identical, unless you also enable
#: torch.use_deterministic_algorithms (which costs speed).
TRAINING = {"max_epochs": 3, "batch_size": 8, "val_split": 0.2,
            "device": "cpu", "verbose": "silent"}


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def synthetic_recording(n_trials=16, trial_s=3.5, seed=0):
    """A small motor-imagery recording: continuous EEG plus one event per trial."""
    duration = (n_trials + 1) * trial_s
    eeg = EEGSignalGenerator(fs=FS, seed=seed).get_chunk(
        duration=duration, n_channels=len(CHANNELS))

    events = Events(optional_columns={"trial_idx": "Int64", "label": "Int64"})
    events.append([{"onset": (i + 0.5) * trial_s, "duration": trial_s,
                    "trial_idx": i, "label": i % 2} for i in range(n_trials)])

    channel_set = ChannelSet()
    channel_set.add_unipolar_eeg_channels(CHANNELS)
    rec = Recording(BidsInfo(subject="synthetic", task="mi"))
    rec.add_signal("eeg", Signal(eeg, fs=FS, channel_set=channel_set))
    rec.set_events(events)
    return rec


recording = synthetic_recording()
print(f"synthetic recording: {len(CHANNELS)} channels, "
      f"{len(recording.events.df)} trials @ {FS:g} Hz")


# --------------------------------------------------------------------------- #
# 1) In a pipeline: one setting, and it covers the whole run.
#    A fresh pipeline per run -- `fit` continues from whatever a pipeline already
#    holds, so reusing one would train the same model twice instead of twice
#    training the same model.
# --------------------------------------------------------------------------- #
def pipeline_scores(random_state):
    pipe = MIEEGInceptionPipeline(
        channels=CHANNELS,
        classifier={"training": dict(TRAINING, random_state=random_state)})
    return pipe.fit([recording]).predict(recording)


section("1) A pipeline, seeded: classifier.training.random_state = 0")
a, b = pipeline_scores(0), pipeline_scores(0)
print(f"  max |run A - run B| = {np.abs(a - b).max():.4f}   (identical)")

section("2) The same pipeline with the seed switched off (the default)")
a, b = pipeline_scores(None), pipeline_scores(None)
print(f"  max |run A - run B| = {np.abs(a - b).max():.4f}   (every run its own)")


# --------------------------------------------------------------------------- #
# 3) Going around the pipeline, the backbone is yours -- and its initial weights
#    are drawn *before* the estimator exists, so `random_state` cannot reach them.
#    Seed the construction as well; `seeded_rng` does it for any nn.Module,
#    without leaving the seed set behind. This is exactly what the pipeline runs
#    for you when it builds the backbone.
# --------------------------------------------------------------------------- #
def estimator_proba(random_state, backbone_seed):
    with seeded_rng(backbone_seed):
        backbone = EEGInceptionV2(input_samples=256, n_cha=len(CHANNELS))
    clf = TorchClassifier(backbone, max_epochs=3, batch_size=8, val_split=0.2,
                          device="cpu", verbose=0, random_state=random_state)
    X = np.random.RandomState(0).randn(40, 256, len(CHANNELS)).astype("float32")
    y = np.array([0, 1] * 20)
    return clf.fit(X, y).predict_proba(X)


section("3) Estimator only: random_state cannot reach the backbone's init")
a = estimator_proba(random_state=0, backbone_seed=0)
b = estimator_proba(random_state=0, backbone_seed=1)
c = estimator_proba(random_state=0, backbone_seed=0)
print(f"  different backbone seed = {np.abs(a - b).max():.4f}   (the init matters too)")
print(f"  same backbone seed      = {np.abs(a - c).max():.4f}   (both fixed -> repeats)")


# --------------------------------------------------------------------------- #
# 4) The seeding is forked, not global: a repeatable fit does not reset the
#    random numbers your script draws afterwards.
# --------------------------------------------------------------------------- #
section("4) Your own RNG stream is left alone")
torch.manual_seed(123)
expected = torch.randn(3)
torch.manual_seed(123)
pipeline_scores(0)
print(f"  draws after a seeded fit unchanged: {torch.equal(torch.randn(3), expected)}")
