import numpy as np
from abc import ABC, abstractmethod


class SignalGenerator(ABC):

    def __init__(self, fs):
        self.fs = fs

    @abstractmethod
    def get_chunk(self, duration, n_channels):
        pass


class SinusoidalSignalGenerator(SignalGenerator):

    """Experimental class for sinusoidal signals generation
    """

    def __init__(self, fs, freqs, noise_type='white',
                 noise_params={'mean': 0, 'sigma': 1}):
        # Check errors
        assert noise_type in (None, 'white'), 'Unknown noise type'
        if noise_type == 'white':
            assert 'mean' in noise_params and 'sigma' in noise_params, \
                'For white noise, you must define mean and sigma noise params'
        # Attributes
        super().__init__(fs)
        self.freqs = freqs
        self.noise_type = noise_type
        self.noise_params = noise_params

    def get_chunk(self, duration, n_channels):
        t = np.linspace(0, duration-(1/self.fs), self.fs*duration)
        chunk = np.zeros((t.shape[0], n_channels))
        # Add sinusoids
        for f in self.freqs:
            s = np.sin(2*np.pi*f*t).reshape(-1, 1)
            chunk += s
        # Add noise
        if self.noise_type == 'white':
            chunk += np.random.random(chunk.shape)
        return chunk
