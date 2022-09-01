"""Created on Monday March 15 19:27:14 2021

In this module, you will find all functions and data structures related to
EEG and MEG signals. Enjoy!

References:
    https://stefanappelhoff.com/eeg_positions
    https://github.com/sappelhoff/eeg_positions
    https://mne.tools/dev/auto_tutorials/intro/40_sensor_locations.html
@author: Eduardo Santamaría-Vázquez
"""

# Built-in imports
import math

# External imports
import warnings

import scipy.io
import numpy as np

# Medusa imports
from medusa import components
from medusa import meeg


class MEEGChannel(components.SerializableComponent):
    """This class implements a M/EEG channel.
    """

    # TODO: This class is till not being used for compatibility reasons, but it
    #  should be introduced in the next major update of medusa kernel to remove
    #  channel dictionaries and simplify the management of MEEG signals

    def __init__(self, label, coordinates=None, reference=None):
        """Constructor for class MEEGChannel.

        Parameters
        ----------
        label: str
            Label of the channel.
        coordinates: dict, optional
            Dict with the coordinates of the electrode. It is strongly
            recommended to set this optional parameter in order to use advanced
            features of MEDUSA (e.g., topographic plots).
        reference: MEEGChannel, optional
            Use only for bipolar montages to define the reference electrode.
        """
        # Check errors
        if reference is not None:
            if not isinstance(reference, MEEGChannel):
                raise ValueError(
                    'Parameter reference must be of type MEEGChannel or None')
        if coordinates is None:
            warnings.warn('Channel coordinates are used by some advanced '
                          'features of MEDUSA (e.g., topographic plots). '
                          'Please consider setting the coordinates or use '
                          'function MEEGChannel.from_standard_set to load '
                          'standard coordinates.')
        # Set attributes
        self.label = label
        self.coordinates = coordinates
        self.reference = reference

    def to_serializable_obj(self):
        pass

    @classmethod
    def from_standard_set(cls, label, standard='10-5'):
        pass

    @classmethod
    def from_serializable_obj(cls, data):
        pass


class EEGChannelSet(components.SerializableComponent):
    """Class to represent an EEG montage with ordered channels in specific
    coordinates. It also provides functionality to load channels from EEG
    standards 10-20, 10-10 and 10-5 directly from the labels.
    """

    def __init__(self, reference_method='common', dim='2D',
                 coord_system='spherical'):
        """Constructor of class EEGChannelSet

        Parameters
        ----------
        reference_method: str {'common'|'average'|'bipolar'}
            Reference method. Recordings with common reference are referenced
            to the same channel (e.g., ear lobe, mastoid). Recordings with
            average reference are referenced to the average of all or several
            channels. Finally, bipolar reference is the subtraction of 2
            channels.
        dim: str {'2D'|'3D'}
            Dimensions of the coordinates plane.
        coord_system: str {'cartesian'|'spherical'}
            Coordinates system. Take into account that, if dim = '2D' spherical
            refers to polar coordinates
        """

        # Check errors
        if reference_method not in ('common', 'average', 'bipolar'):
            raise ValueError('Unknown reference method %s' % reference_method)
        if dim not in ('2D', '3D'):
            raise ValueError('Unknown number of dimensions %s' % dim)
        if coord_system not in ('cartesian', 'spherical'):
            raise ValueError('Unknown coordinates system %s' % coord_system)

        # Set attributes
        self.reference_method = reference_method
        self.dim = dim
        self.coord_system = coord_system
        self.channels = None
        self.n_cha = None
        self.l_cha = None
        self.montage = None
        self.ground = None
        self.allow_unlocated_channels = False

    def set_ground(self, ground):
        """Sets the ground of the montage

        Parameters
        ----------
        ground: dict
            Dict with the ground data. The easiest way to calculate it is using
            function get_standard_channel_data_from_label. Keys:
                - label: channel label
                - coordinates: depends on the coordinate system and dimensions:
                        - Dim = 2D:
                            - Cartesian coordinates. Ex: {'x': 0.5, 'y': 0}
                            - Spherical coordinates Ex: {'r': 0.5,
                                'theta': np.pi/2}
                        - Dim = 3D:
                            - Cartesian coordinates. Ex: {'x': 0.5,
                                'y': 0, 'z': 0.8}
                            - Spherical coordinates. Ex: {'r': 0.5,
                                'theta': np.pi/2, 'phi': np.pi/4}
        """
        self.ground = ground

    def add_channel(self, channel, reference):
        """Function to add a channel to the end of the current montage.
        Take into account that the order of the channels is important!

        Parameters
        ----------
        channel: dict
            Dict with the channel data. The easiest way to calculate it is using
            function get_standard_channel_data_from_label. Keys:
                - label: channel label
                - coordinates: depends on the coordinate system and dimensions:
                        - Dim = 2D:
                            - Cartesian coordinates. Ex: {'x': 0.5, 'y': 0}
                            - Spherical coordinates Ex: {'r': 0.5,
                                'theta': np.pi/2}
                        - Dim = 3D:
                            - Cartesian coordinates. Ex: {'x': 0.5,
                                'y': 0, 'z': 0.8}
                            - Spherical coordinates. Ex: {'r': 0.5,
                                'theta': np.pi/2, 'phi': np.pi/4}
        reference: dict, list of dicts or str, optional
            For common and bipolar reference modes, reference must be a dict
            with the label and coordinates of the reference. For average
            reference, it can be 'all' to indicate a common average
            reference, or a list of dicts (with label and coordinates) of the
            averaged reference for each channel

        See Also
        --------
        get_standard_channel_data_from_label: returns channel data given the
            channel label and the standard. It can be used to get the reference
        """
        # TODO: check input
        channel['reference'] = reference
        channels = list() if self.channels is None else self.channels
        channels.append(channel)
        # Check channels
        self.__check_channels(channels)
        # Store attributes
        self.channels = channels
        self.n_cha = len(self.channels)
        self.l_cha = [cha['label'] for cha in self.channels]

    def set_montage(self, channels, ground=None,
                    allow_unlocated_channels=False):
        """Sets a custom montage, overwriting the previous one. Add single
        channels more easily using function add_custom_channel and
        add_standard_channel.

        Parameters
        ----------
        channels : list
            List of dicts, each of them representing a channel. The dict must
            contain the label, coordinates according to parameters dim an
            coord_system, and reference. For common reference mode, the
            reference must be a single channel with label and coordinates. For
            average reference mode, reference can be a list of dicts with the
            channels (label and coordinates) or "all", to specify common average
            reference. For bipolar reference mode, the reference must be a
            single channel with label and coordinates. In all cases you can set
            the reference to None, but this is not recommended. The definition
            of the coordinates may be skipped depending on the value of
            allow_unlocated_channels.
        ground : dict
            Dict containing the label and coordinates of the ground electrode
        allow_unlocated_channels: bool
            If False, the coordinates of all channels must be defined within
            channels dict. If True, the channels may not have known coordinates,
            This may be convenient if the localization of a channel is not known
            or it has a non-standard label, but the behaviour in functions that
            need coordinates (e.g., topographic_plots) is unpredictable.

        See Also
        --------
        set_standard_montage: preferred choice in most cases
        """
        # Check errors
        self.allow_unlocated_channels = allow_unlocated_channels
        self.__check_channels(channels, ground)
        # Set attributes
        self.channels = channels
        self.ground = ground
        self.n_cha = len(self.channels)
        self.l_cha = [cha['label'] for cha in self.channels]

    def set_standard_montage(self, l_cha=None, l_reference=None, l_ground=None,
                             montage='10-05', drop_landmarks=True,
                             allow_unlocated_channels=False, standard=None):
        """Set standard EEG channels with common reference. In 3 dimensions,
        the equator is taken a Nz-T10-Iz-T9.

        Parameters
        ----------
        l_cha : list, optional
            List of channels labels. The data will be returned keeping the
            same order. If None, the channels will be returned in the same order
            as they appear in the corresponding standard in medusa.meeg
        l_reference : str, optional
            Label of the reference. Usual choices are left preauricular point
            (LPA) and right preauricular point (RPA). Leave to None if you do
            not want to specify the reference (not recommended). Use only if
            reference_method is 'common'.
        l_ground : str, optional
            Label of the ground. Usual choices are AFz or FPz.
        montage : str {'10-20'|'10-10'|'10-05'} or dict
            EEG standard. If its a string, the corresponding labels and
            locations of the standard channels will be loaded using the files in
            medusa.meeg. To load a different montage, use function
            as meeg.read_montage_file and pass the returned dict here.
        drop_landmarks : bool
            Drop landmarks: nasion (NAS), left preauricular point (LPA) and
            right preauricular point (RPA) or mastoids (M1, M2). These are
            usually used as reference or ground points, so they are usually
            removed from the channel set for data analysis. Only used if l_cha
            is None.
        allow_unlocated_channels: bool
            If False, all the labels in parameter l_cha must be contained in the
            montage, which contains the corresponding coordinates. If True, the
            channels may not be in the standard, and they will be saved with no
            coordinates. This allows to save labels that are not defined in the
            montage, but the behaviour in functions that need locations
            (e.g., topographic_plots) is unpredictable.
        standard: str
            DEPRECATED. Only left for compatibility reasons.
        """
        # Check errors
        if self.reference_method != 'common':
            raise ValueError('Function set_standard_channels is available '
                             'only for recordings with common reference. For '
                             'custom montages use set_custom_channels')
        assert self.dim == '2D' or self.dim == '3D', \
            'Incorrect input on dim parameter'
        assert self.coord_system == 'cartesian' or \
               self.coord_system == 'spherical', \
            'Incorrect input on coord_system parameter'
        # Compatibility
        montage = standard if standard is not None else montage
        # Get montage
        if isinstance(montage, str):
            # Load standard montage
            self.montage = montage
            montage = meeg.get_standard_montage(standard=montage,
                                                dim=self.dim,
                                                coord_system=self.coord_system)
        else:
            # Set custom montage
            montage = montage.copy()
            self.montage = montage
        # Reference
        if l_reference is not None:
            if l_reference in montage:
                reference = montage[l_reference]
            else:
                if allow_unlocated_channels:
                    reference = dict()
                    warnings.warn('Reference not defined in montage')
                else:
                    raise meeg.ChannelNotFound(l_reference)
            reference['label'] = l_reference
        else:
            reference = None
        # Get list of labels to get
        labels = montage.keys() if l_cha is None \
            else [l.upper().strip() for l in l_cha]
        # Get channels
        channels = list()
        for label in labels:
            # Drop landmarks
            if l_cha is None:
                if drop_landmarks:
                    if label in ('NAS', 'LPA', 'RPA', 'M1', 'M2'):
                        continue
            # Append info
            if label in montage:
                channel_data = montage[label]
            else:
                if allow_unlocated_channels:
                    channel_data = dict()
                    warnings.warn('Channel %s not defined in montage' % label)
                else:
                    raise meeg.ChannelNotFound(label)
            channel_data['label'] = label
            channel_data['reference'] = reference
            channels.append(channel_data)
        # Ground
        if l_ground is not None:
            if l_ground in montage:
                ground = montage[l_ground]
            else:
                if allow_unlocated_channels:
                    ground = dict()
                    warnings.warn('Ground not defined in montage')
                else:
                    raise meeg.ChannelNotFound(l_ground)
            ground['label'] = ground
        else:
            ground = None
        # Check channels
        self.set_montage(channels, ground=ground,
                         allow_unlocated_channels=allow_unlocated_channels)

    def __check_channels(self, channels, ground=None):
        # Get mandatory and coordinates keys for each dim and coord_system
        cha_keys = ['label', 'reference']
        ref_keys = ['label']
        gnd_keys = ['label']
        if self.dim == '2D':
            if self.coord_system == 'cartesian':
                coord_keys = ['x', 'y']
            else:
                coord_keys = ['r', 'theta']
        else:
            if self.coord_system == 'cartesian':
                coord_keys = ['x', 'y', 'z']
            else:
                coord_keys = ['r', 'theta', 'phi']
        if not self.allow_unlocated_channels:
            cha_keys += coord_keys
            ref_keys += ref_keys
            gnd_keys += gnd_keys
        # Check keys
        references = list()
        for cha in channels:
            if not all(k in cha for k in cha_keys):
                raise ValueError('Malformed channel %s. Dict keys must be %s' %
                                 (str(cha), str(cha_keys)))
            # Check reference
            reference = cha['reference']
            references.append(reference)
            if reference is not None:
                if self.reference_method == 'common':
                    # Single reference
                    assert isinstance(reference, dict), \
                        'Reference must be None or dict'
                    if not all(k in reference for k in ref_keys):
                        raise ValueError('Malformed reference in channel %s. '
                                         'Reference keys must be %s' %
                                         (str(cha), str(ref_keys)))
                    # All references must be identical
                    if not all(ref == references[0] for ref in references):
                        raise ValueError('All references must be identical '
                                         'in common reference mode')
                elif self.reference_method == 'average':
                    # Average reference
                    assert isinstance(reference, list), \
                        'Reference must be None or list of dicts'
                    for r in reference:
                        if not all(k in r for k in ref_keys):
                            raise ValueError('Malformed reference in '
                                             'channel %s. Reference keys '
                                             'must be %s' %
                                             (str(cha), str(ref_keys)))
                elif self.reference_method == 'bipolar':
                    # Single reference
                    assert isinstance(reference, dict), \
                        'Reference must be of type dict'
                    if not all(k in reference for k in ref_keys):
                        raise ValueError('Malformed reference in channel %s. '
                                         'Reference keys must be %s' %
                                         (str(cha), str(ref_keys)))
        # Check ground
        if ground is not None:
            if not all(k in ground for k in gnd_keys):
                raise ValueError(
                    'Malformed ground. Dict keys must be %s' % (str(gnd_keys)))

    def get_cha_idx_from_labels(self, labels):
        """Returns the position of the channels given the labels

        Parameters
        ----------
         labels : list
            Labels to check. The order matters

        Returns
        -------
        indexes : np.ndarray
            Indexes of the channels in the set
        """
        return [self.l_cha.index(l) for l in labels]

    def check_channels_labels(self, labels, strict=False):
        """Checks the order and labels of the channels

        Parameters
        ----------
        labels : list
            Labels to check. The order matters
        strict : bool
            If True, comparison is strict. The function will check that the
            channel set contains the channels given by parameter labels and
            in the same order. If false, the function checks that the
            channels are contained in the channel set, but they could be in
            different order and the set could contain more channels

        Returns
        -------
        check : bool
            True if the labels and order are the same. False otherwise
        """
        if strict:
            check = True
            for i in range(len(labels)):
                if self.l_cha[i] != labels[i]:
                    check = False
        else:
            check = True
            for l in labels:
                if l not in self.l_cha:
                    check = False
        return check

    def subset(self, cha_idx):
        """Selects the channels given the indexes, creating a subset. The
        order of the channels will be updated

        Parameters
        ----------
        cha_idx : np.ndarray
            Indexes of the channels to select. The order matters
        """
        self.channels = [self.channels[idx] for idx in cha_idx]
        self.n_cha = len(self.channels)
        self.l_cha = [cha['label'] for cha in self.channels]

    def sort_nearest_channels(self):
        """ Sorts the nearest channels for each possible channel
        Return
        ----------
        sorted_dist_ch: dict()
            Dictionary that includes, for each channel (as a key), the rest
            of channels sorted by its closeness to that channel.
        """
        # TODO: Añadir la opción de hacer un sort de una coordenada específica,
        #  o en otra función
        if not self.channels:
            raise Exception(
                'Cannot compute the nearest channels if channel set '
                'is not initialized!')

        # Compute distance matrix
        dist_matrix = self.compute_dist_matrix()

        # Create the dictionary and format it
        sorted_dist_ch = dict()
        for i, label in enumerate(self.l_cha):
            d_sorted = np.sort(dist_matrix[i, :])
            idx_sorted = np.argsort(dist_matrix[i, :])
            if d_sorted[0] != 0:
                raise Exception('[meeg/EEGChannelSet/sort_nearest_channels] '
                                'Something ocurred, distance between the same '
                                'channel is not zero!')
            sorted_dist_ch[label] = []
            for j in range(1, len(d_sorted)):
                sorted_dist_ch[label].append({
                    "dist": d_sorted[j],
                    "channel": self.channels[idx_sorted[j]]
                })
        return sorted_dist_ch

    def compute_dist_matrix(self):
        """This function computes the distances between all channels in the
        channel set and stores them into a matrix.

        Returns
        -------------
        dist_matrix: ndarray of dimensions [ncha x ncha]
            Distances between all the channels.
        """
        if not self.channels:
            raise Exception(
                'Cannot compute the distance matrix if channel set '
                'is not initialized!')
        # Instantiate matrix of distances
        dist_matrix = np.empty((self.n_cha, self.n_cha))

        # For 2D coordinates
        if self.dim == '2D':
            if self.coord_system == 'cartesian':
                for i, cha in enumerate(self.channels):
                    # Find location of channel
                    cha_pos = np.array([cha['x'], cha['y']])
                    # Find the location of the rest of the channels
                    for j, temp_cha in enumerate(self.channels):
                        temp_cha_pos = np.array(
                            [temp_cha['x'], temp_cha['y']]
                        )
                        d = np.sqrt(np.sum(np.power(temp_cha_pos - cha_pos, 2)))
                        dist_matrix[i, j] = d
            elif self.coord_system == 'spherical':
                for i, cha in enumerate(self.channels):
                    # Find location of channel
                    r_cha, theta_cha = cha['r'], cha['theta']
                    # Find the location of the rest of the channels
                    for j, temp_cha in enumerate(self.channels):
                        r_temp_cha, theta_temp_cha = \
                            temp_cha['r'], temp_cha['theta']
                        d = np.abs(np.sqrt(r_cha ** 2 + r_temp_cha ** 2 -
                                           2 * r_cha * r_temp_cha *
                                           np.cos(theta_temp_cha -
                                                  theta_cha)))
                        dist_matrix[i, j] = d
        # For 3D coordinates
        elif self.dim == '3D':
            if self.coord_system == 'cartesian':
                for i, cha in enumerate(self.channels):
                    # Find location of channel
                    cha_pos = np.array([cha['x'], cha['y'], cha['z']])
                    # Find the location of the rest of the channels
                    for j, temp_cha in enumerate(self.channels):
                        temp_cha_pos = np.array(
                            [temp_cha['x'], temp_cha['y'], temp_cha['z']]
                        )
                        d = np.sqrt(np.sum(np.power(temp_cha_pos - cha_pos, 2)))
                        dist_matrix[i, j] = d
            elif self.coord_system == 'spherical':
                for i, cha in enumerate(self.channels):
                    # Find location of channel
                    r_cha, theta_cha, phi_cha = \
                        cha['r'], cha['theta'], cha['phi']
                    # Find the location of the rest of the channels
                    for j, temp_cha in enumerate(self.channels):
                        r_temp_cha, theta_temp_cha, phi_temp_cha = \
                            temp_cha['r'], temp_cha['theta'], temp_cha['phi']
                        d = np.abs(
                            np.sqrt(
                                r_cha ** 2 +
                                r_temp_cha ** 2 -
                                2 * r_cha * r_temp_cha *
                                (
                                        np.sin(theta_cha) *
                                        np.sin(theta_temp_cha) *
                                        np.cos(phi_cha - phi_temp_cha) +
                                        np.cos(theta_temp_cha) *
                                        np.cos(theta_cha))
                            )
                        )
                        dist_matrix[i, j] = d
        return dist_matrix

    def to_serializable_obj(self):
        return self.__dict__

    @classmethod
    def from_serializable_obj(cls, dict_data):
        inst = cls()
        inst.__dict__.update(dict_data)
        return inst


class MEGChannelSet(components.SerializableComponent):

    # TODO
    def __init__(self):
        self.channels = None
        self.n_cha = None
        self.l_cha = None

    def to_serializable_obj(self):
        return self.__dict__

    @classmethod
    def from_serializable_obj(cls, dict_data):
        inst = cls()
        inst.__dict__.update(dict_data)
        return inst


class EEG(components.BiosignalData):
    """Electroencephalography (EEG) biosignal
    """

    def __init__(self, times, signal, fs, channel_set, **kwargs):
        """EEG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. EEG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : meeg_standards.EEGChannelSet
            EEG channel set
        kwargs: kwargs
            Any other parameter provided will be saved in the class (e.g.,
            equipment description)
        """
        # To numpy arrays
        times = np.array(times)
        signal = np.array(signal)
        # Check errors
        if signal.shape[1] != channel_set.n_cha:
            raise Exception("Signal with shape [samples x channels] does not "
                            "match with the number of channels")
        if times.shape[0] != signal.shape[0]:
            raise Exception("Parameters times (shape: %s) and signal (shape: "
                            "%s) must have the same length" % (
                                str(times.shape), str(signal.shape))
                            )

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def change_channel_set(self, channel_set):
        """Smart change of channel set, updating the signal and all related
        attributes

        Parameters
        ----------
        channel_set : meeg_standards.EEGChannelSet
            EEG channel set
        """
        # Get the index of the channels
        cha_idx = self.channel_set.get_cha_idx_from_labels(channel_set.l_cha)
        # Select and reorganize channels channels
        self.channel_set.subset(cha_idx)
        # Reorganize signal
        self.signal = self.signal[:, cha_idx]

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
            if type(rec_dict[key]) == EEGChannelSet:
                rec_dict[key] = rec_dict[key].to_serializable_obj()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        # Load channel set dict
        dict_data['channel_set'] = EEGChannelSet.from_serializable_obj(
            dict_data['channel_set']
        )
        return cls(**dict_data)


class MEG(components.BiosignalData):
    # TODO check everything

    """Magnetoencephalography (MEG) biosignal
    """

    def __init__(self, times, signal, fs, channel_set, **kwargs):
        """MEG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. MEG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : list or meeg_standards.MEGChannelSet
            MEG channel set.
        kwargs: kwargs
            Any other parameter provided will be saved in the class (e.g.,
            equipment description)
        """
        # To numpy arrays
        times = np.array(times)
        signal = np.array(signal)
        # Check errors
        if signal.shape[1] != channel_set.n_cha:
            raise Exception("Signal with shape [samples x channels] does not "
                            "match with the number of channels")
        if times.shape[0] != signal.shape[0]:
            raise Exception("Parameters times and signal must have the same "
                            "length")

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def change_channel_set(self, channel_set):
        """Smart change of channel set, updating the signal and all related
        attributes

        Parameters
        ----------
        channel_set : meeg_standards.MEGChannelSet
            MEG channel set
        """
        raise NotImplementedError

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)

    @staticmethod
    def load_meg_signal_from_spm_file(path, fs):
        # TODO check everything

        """Function to load a MEG recording from a spm file

        Parameters
        ----------
        path : str
            Path of the file
        fs : int or float
            Sample rate of the recording
        """
        data = scipy.io.loadmat(path)
        info = data['D'][0, 0]

        datatype = info['datatype'][0]
        num_chan = np.size(info['channels'])
        num_samples = info['Nsamples'][0, 0]

        raw = np.fromfile(path[0:len(path) - 3] + 'dat',
                          datatype[0:len(datatype) - 3])
        signal = np.reshape(raw, [num_chan, num_samples], order='F')
        times = np.linspace(0, signal[0] / fs, signal[0])
        channels = None
        return MEG(times, signal, fs, channels)


class Connecitivity:
    # TODO: check everything

    """Customizable class with connectivity info from EEG/MEG recordings
    """

    def __init__(self, data, trial_len, parameter, filt_mode, **kwargs):
        """Class constructor

        Parameters
        ----------
        data : bla bla
            Bla bla
        trial_len : bla bla
            Bla bla
        parameter : bla bla
            Bla bla
        filt_mode : bla bla
            Bla bla
        kwargs
            Optional information of the EEG recording (e.g. subject, amplifier,
            etc)
        """
        # Params
        data = np.array(data)

        if not (filt_mode == 'all' or
                filt_mode == 'bands' or
                filt_mode == 'win'):
            raise ValueError("Unknown filtering mode")

        self.data = data
        self.trial_len = trial_len
        self.parameter = parameter
        self.filt_mode = filt_mode

        # Optional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class UnknownStandardChannel(Exception):

    def __init__(self, msg=None):
        """Class constructor

        Parameters
        ----------
        msg: string or None
            Custom message
        """
        if msg is None:
            msg = 'Unknown standard channel'
        super().__init__(msg)