from medusa import components
from medusa import meeg
import os


def _load_eeg_recording():
    """Helper to load the test recording. Used by both the test and __main__."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(test_dir, 'data', 'eeg.rec.json')
    return components.Recording.load(data_path)


def test_load_eeg_recording():
    rec = _load_eeg_recording()
    assert isinstance(rec, components.Recording)


if __name__ == '__main__':
    rec = _load_eeg_recording()
