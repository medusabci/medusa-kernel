from medusa import components
from medusa import meeg


def test_load_eeg_recording():
    rec = components.Recording.load('data/eeg.rec.json')
    assert isinstance(rec, components.Recording)
    return rec


if __name__ == '__main__':
    rec = test_load_eeg_recording()

