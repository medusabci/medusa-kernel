from medusa import transforms
from medusa import signal_generators


def test_power_spectral_density():
    generator = signal_generators.SinusoidalSignalGenerator(
        fs=200, freqs=[1, 5], noise_type=None)
    chunk = generator.get_chunk(10, 5)
    f, psd = transforms.power_spectral_density(
        signal=chunk, fs=200, segment_pct=80, overlap_pct=50, window='boxcar')
    assert psd.shape == (1, 801, 5)
    assert psd[0, 8, 1] == 4


if __name__ == '__main__':
    test_power_spectral_density()

