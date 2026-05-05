from medusa import components
from medusa import meeg
import os


def test_load_eeg_recording():
    # Get the directory of this test file
    test_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(test_dir, 'data', 'eeg.rec.json')
    rec = components.Recording.load(data_path)
    assert isinstance(rec, components.Recording)
    return rec


if __name__ == '__main__':
    rec = test_load_eeg_recording()

