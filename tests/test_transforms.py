from medusa import transforms
from medusa import signal_generators
import numpy as np


def test_power_spectral_density():
    generator = signal_generators.SinusoidalSignalGenerator(
        fs=200, freqs=[1, 5], noise_type=None)
    chunk = generator.get_chunk(10, 5)
    f, psd = transforms.power_spectral_density(
        signal=chunk, fs=200, segment_pct=80, overlap_pct=50, window='boxcar')
    assert psd.shape == (1, 801, 5)
    # Use approximate comparison to account for floating-point precision
    # (expected analytic value is 4 for a unit-amplitude sinusoid at 5 Hz)
    np.testing.assert_allclose(psd[0, 8, 1], 4, rtol=1e-6)


if __name__ == '__main__':
    test_power_spectral_density()
