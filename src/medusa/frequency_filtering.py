import numpy as np
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
from medusa import components


class FIRFilter(components.ProcessingMethod):

    def __init__(self, order, cutoff, btype, width=None, window='hamming',
                 scale=True, filt_method='filtfilt', axis=0):
        """FIR filter designed using the implementation of scipy.signal.firwin.
        See the documentation of this function to find useful information about
        this class

        Parameters
        ----------
        order: int
            Length of the filter (number of coefficients, i.e. the filter order
            + 1). numtaps must be odd if a passband includes the Nyquist
            frequency.
        cutoff: float or 1-D array_like
            Cutoff frequency of filter (expressed in the same units as fs) OR an
            array of cutoff frequencies (that is, band edges). In the latter
            case, the frequencies in cutoff should be positive and monotonically
            increasing between 0 and fs/2. The values 0 and fs/2 must not be
            included in cutoff.
        btype: str {‘bandpass’|‘lowpass’|‘highpass’|‘bandstop’}
            Band type of the filter. It also controls the parameter pass_zero of
            them scipy.signal.firwin function
        width: float or None, optional
            If width is not None, then assume it is the approximate width of the
            transition region (expressed in the same units as fs) for use in
            Kaiser FIR filter design. In this case, the window argument is
            ignored.
        window: string or tuple of string and parameter values, optional
            Desired window to use. See scipy.signal.get_window for a list of
            windows and required parameters.
        scale: bool, optional
            Set to True to scale the coefficients so that the frequency response
            is exactly unity at a certain frequency. That frequency is either:

                - 0 (DC) if the first passband starts at 0 (i.e. pass_zero is
                    True)
                - fs/2 (the Nyquist frequency) if the first passband ends at
                    fs/2 (i.e the filter is a single band highpass filter);
                    center of first passband otherwise
        filt_method: str {'lfilter', 'filtfilt'}
            Filtering method. See scipy.signal.lfilter or scipy.signal.filtfilt
            for more information.
        axis: int
            The axis to which the filter is applied. By convention, signals
            in medusa are defined by [samples x channels], so axis is set to
            0 by default.
        """
        # Super call to specify the outputs of fit and apply functions
        super().__init__(fit=[], transform=['s'], fit_transform=['s'])

        # Variables
        self.btype = btype
        self.order = order
        self.cutoff = cutoff
        self.width = width
        self.window = window
        self.scale = scale
        self.filt_method = filt_method
        self.axis = axis

        # Parameters to fit
        self.fs = None
        self.a = None
        self.b = None

    def display(self):
        display_filter(self.b, self.a, self.fs)

    def fit(self, fs):
        self.fs = fs
        self.b = scipy_signal.firwin(numtaps=self.order,
                                     cutoff=self.cutoff,
                                     width=self.width,
                                     window=self.window,
                                     pass_zero=self.btype,
                                     scale=self.scale,
                                     fs=self.fs)
        self.a = [1.0]

    def transform(self, signal):
        if self.filt_method == 'filtfilt':
            s = scipy_signal.filtfilt(self.b, self.a, signal,
                                      axis=self.axis)
        elif self.filt_method == 'lfilter':
            s = scipy_signal.lfilter(self.b, self.a, signal,
                                     axis=self.axis)
        else:
            raise ValueError("Unsupported filtering method method!")
        return s

    def fit_transform(self, signal, fs):
        """Fits and applies the filter

        Parameters
        ----------
        signal: np.ndarray
            Signal to filter. By default, the expected shape is [samples x
            channels], but this order can be changed using axis parameter in
            constructor.
        fs: float
            The sampling frequency of the signal in Hz. Each frequency in
            cutoff must be between 0 and fs/2. Default is 2.
        """
        self.fit(fs)
        return self.transform(signal)


class IIRFilter(components.ProcessingMethod):

    def __init__(self, order, cutoff, btype, filt_method='sosfiltfilt', axis=0):
        """IIR Butterworth filter wrapper designed using implementation of
        scipy.signal.butter. See the documentation of this function to find
        useful information about this class.

        Parameters
        ----------
        order: int
            Length of the filter (number of coefficients, i.e. the filter order
            + 1). This parameter must be odd if a passband includes the
            Nyquist frequency.
        cutoff: float or 1-D array_like
            Cutoff frequency of filter (expressed in the same units as fs) OR an
            array of cutoff frequencies (that is, band edges). In the latter
            case, the frequencies in cutoff should be positive and monotonically
            increasing between 0 and fs/2. The values 0 and fs/2 must not be
            included in cutoff.
        btype: str {‘bandpass’|‘lowpass’|‘highpass’|‘bandstop’}
            Band type of the filter. It also controls the parameter pass_zero of
            them scipy.signal.firwin function
        filt_method: str {'sosfilt', 'sosfiltfilt'}
            Filtering method. See scipy.signal.sosfilt or
            scipy.signal.sosfiltfilt for more information. For real time
            fitlering, use sosfilt. For offline filtering, sosfiltfilt is the
            recommended filtering method.
        axis: int
            The axis to which the filter is applied. By convention, signals
            in medusa are defined by [samples x channels], so axis is set to
            0 by default.
        """
        # Super call to specify the outputs of fit and apply functions
        super().__init__(fit=[], transform=['s'], fit_transform=['s'])

        # Variables
        self.btype = btype
        self.order = order
        self.cutoff = cutoff
        self.filt_method = filt_method
        self.axis = axis

        # Parameters to fit
        self.fs = None
        self.sos = None
        self.zi = None

    def display(self):
        """Displays the filter. Function fit must be called first. This uses
        the function medusa.frequency_filtering.display_filter()
        """
        b, a = scipy_signal.sos2tf(self.sos)
        display_filter(b, a, self.fs)

    def fit(self, fs, n_cha=None):
        """Fits the filter

        Parameters
        ----------
        fs: float
            The sampling frequency of the signal in Hz. Each frequency in
            cutoff must be between 0 and fs/2. Default is 2.
        n_cha: int
            Number of channels. Used to compute the initial conditions of the
            filter. Only required with sosfilt filtering method (online
            filtering)
        """
        self.fs = fs
        self.sos = scipy_signal.butter(N=self.order,
                                       Wn=self.cutoff,
                                       btype=self.btype,
                                       analog=False,
                                       output='sos',
                                       fs=self.fs)
        if self.filt_method == 'sosfilt':
            if n_cha is None:
                raise ValueError('Specify the number of channels to compute '
                                 'the initial conditions of the filter')
            self.zi = scipy_signal.sosfilt_zi(self.sos)
            self.zi = np.repeat(self.zi[:, :, np.newaxis], n_cha, axis=2)

    def transform(self, signal):
        """Applies the filter to the signal

        Parameters
        ----------
        signal: np.ndarray
            Signal to filter. By default, the expected shape is [samples x
            channels], but this order can be changed using axis parameter in
            constructor.
        """
        if self.filt_method == 'sosfiltfilt':
            s = scipy_signal.sosfiltfilt(self.sos, signal, axis=self.axis)
        elif self.filt_method == 'sosfilt':
            s, zo = scipy_signal.sosfilt(self.sos, signal, axis=self.axis,
                                         zi=self.zi)
            self.zi = zo
        else:
            raise ValueError("Unsupported filtering method method!")
        return s

    def fit_transform(self, signal, fs):
        """Fits and applies the filter

        Parameters
        ----------
        signal: np.ndarray
            Signal to filter. By default, the expected shape is [samples x
            channels], but this order can be changed using axis parameter in
            constructor.
        fs: float
            The sampling frequency of the signal in Hz. Each frequency in
            cutoff must be between 0 and fs/2. Default is 2.
        """
        if self.axis == 0:
            n_cha = signal.shape[1]
        else:
            n_cha = signal.shape[0]
        self.fit(fs, n_cha)
        return self.transform(signal)


def display_filter(b, a, fs):
    """Displays the frequency response of a given filter.

    Parameters
    ----------
    b: np.ndarray
        Numerator of the filter
    a: np.ndarray
        Denominator of the filter
    fs: float
        Sampling frequency of the signal (in Hz)
    """
    # Frequency response
    w, h = scipy_signal.freqz(b, a)
    freq = w*fs/(2*np.pi)

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # Frequency response
    ax[0].plot(freq, 20 * np.log10(abs(h)), color='blue')
    ax[0].set_title("Frequency Response")
    ax[0].set_ylabel("Amplitude (dB)", color='blue')
    ax[0].set_xlim([0, fs/2])
    ax[0].set_ylim([-50, 1])
    ax[0].grid()

    # Phase response
    ax[1].plot(freq, np.unwrap(np.angle(h)) * 180 / np.pi, color='green')
    ax[1].set_ylabel("Angle (degrees)", color='green')
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_xlim([0, fs/2])
    ax[1].set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax[1].set_ylim([-90, 90])
    ax[1].grid()

    plt.show()





