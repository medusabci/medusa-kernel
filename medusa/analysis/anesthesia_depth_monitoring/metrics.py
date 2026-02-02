import numpy as np

from .epoching import get_epochs
from .utils import check_dimensions
from .signal_generators import *


def compute_suppression_ratio(
        signal, fs,
        sup_pattern_t=0.5,
        sup_pattern_amp_th=3e-6,
):
    """
    Compute the suppression ratio (SR) for each segment/epoch of a signal.

    The SR is defined as the fraction of sliding windows (of duration
    `sup_pattern_t`) whose RMS amplitude is below `sup_pattern_amp_th`.
    For each segment, windows are generated with maximum overlap
    (stride = 1 sample), and:

        SR = (number of windows with RMS < threshold) / (total number of windows)

    Parameters
    ----------
    signal : np.ndarray
        Input signal. Expected shape is `(n_segments, n_samples, n_channels)`.
        If a different shape is provided, it is adapted by
        `medusa.utils.check_dimensions`.
    fs : int
        Sampling frequency in Hz.
    sup_pattern_t : float, optional
        Sliding window duration in seconds to detect the suppression pattern.
        Default: `0.5`.
    sup_pattern_amp_th : float, optional
        RMS amplitude threshold (same units as `signal`; e.g., volts if the
        signal is in V). Windows whose RMS is below this threshold are marked
        as suppressed. Default: `3e-6`.

    Returns
    -------
    sup_ratios : np.ndarray
        Suppression ratio per segment, shape `(n_segments, n_channels)`, with
        values in `[0, 1]`. Squeezing is applied before returning, so
        singleton dimensions may be removed.
    rms_amps : np.ndarray
        RMS amplitude per window and channel, for each segment. Typical shape:
        `(n_segments, n_windows, n_channels)`. Squeezing is applied before
        returning, so singleton dimensions may be removed.
    sup_patterns : np.ndarray
        Boolean suppression mask aligned with `rms_amps`, where `True` indicates
        a suppressed window (RMS below threshold). Typical shape: `(n_segments,
        n_windows, n_channels)`. Squeezing is applied before returning, so
        singleton dimensions may be removed.
    """
    signal = check_dimensions(signal)
    n_segments = signal.shape[0]
    n_samples = signal.shape[1]
    n_cha = signal.shape[2]
    sup_pattern_s = int(sup_pattern_t * fs)
    rms_amps = list()
    sup_patterns = list()
    sup_ratios = list()
    for seg_idx in range(signal.shape[0]):
        signal_segment = signal[seg_idx, :, :]
        # For each epoch, get windows of sup_pattern_s length
        signal_windows = mds.get_epochs(
            signal_segment,
            epochs_length=sup_pattern_s,
            stride=1
        )
        n_windows = signal_windows.shape[0]
        # Compute rms amplitude for each window
        rms_amp = np.sqrt(np.mean(signal_windows**2, axis=1))
        rms_amps.append(rms_amp)
        # Identify suppressed windows
        sup_pattern = rms_amp < sup_pattern_amp_th
        sup_patterns.append(sup_pattern)
        # Suppression ratio
        suppression_ratio = np.sum(sup_pattern) / n_windows
        sup_ratios.append(suppression_ratio)
    # Convert to numpy arrays
    rms_amps = np.array(rms_amps).squeeze()
    sup_patterns = np.array(sup_patterns).squeeze()
    sup_ratios = np.array(sup_ratios).squeeze()
    return sup_ratios, rms_amps, sup_patterns


if __module__ == "__main__":
    # ==============================================================
    # PARAMETERS
    fs = 100
    sig_duration = 60
    suppression_intervals = [(10, 16)]

    # ==============================================================
    # SIGNAL GENERATION
    sig_gen = SinusoidalSignalGenerator(fs=fs, freqs=[10])
    sig = sig_gen.get_chunk(duration=sig_duration, n_channels=1)
    # Simulated suppression patterns
    for sup_int in suppression_intervals:
        sig[int(sup_int[0] * fs):int(sup_int[1] * fs), :] = 0

    # ==============================================================
    # TESTING SUPPRESSION RATIO COMPUTATION
    sup_ratios, rms_amps, sup_patterns = compute_suppression_ratio(
        signal=s,
        fs=fs
    )
    print(f"Suppression Ratio (%): {np.mean(sup_ratios) * 100:.4f}")