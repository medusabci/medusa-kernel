from medusa import signal_generators


def test_sinusoidal_signal_generator():
    generator = signal_generators.SinusoidalSignalGenerator(
        fs=200, freqs=[1, 5], noise_type=None)
    chunk = generator.get_chunk(10, 5)
    assert chunk.shape[0] == 2000
    assert chunk.shape[1] == 5
    assert chunk[250, 0] == 2


if __name__ == '__main__':
    test_sinusoidal_signal_generator()