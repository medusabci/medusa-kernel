import numpy as np
from medusa import components

ECG_12LEADS = {

    'electrodes': {

        # Extremity electrodes
        "RA": "Right Arm — Anywhere between the right shoulder and the wrist",
        "LA": "Left Arm — Anywhere between the left shoulder and the wrist",
        "RL": "Right Leg — Anywhere above the right ankle and below the torso (ground/reference)",
        "LL": "Left Leg — Anywhere above the left ankle and below the torso",

        # Pre-cordial electrodes
        "V1": "4th Intercostal space to the right of the sternum",
        "V2": "4th Intercostal space to the left of the sternum",
        "V3": "Midway between V2 and V4",
        "V4": "5th Intercostal space at the midclavicular line",
        "V5": "Anterior axillary line at the same level as V4",
        "V6": "Midaxillary line at the same level as V4 and V5"
    },

    'leads': {

        # Extremity leads
        "I": "LA - RA",  # Left Arm minus Right Arm
        "II": "LL - RA",  # Left Leg minus Right Arm
        "III": "LL - LA",  # Left Leg minus Left Arm

        # Augmented vectors
        "aVR": "(I + II) / 2",  # Augmented Vector Right
        "aVL": "(I - III) / 2",  # Augmented Vector Left
        "aVF": "(II + III) / 2",  # Augmented Vector Foot

        # Pre-cordial leads
        "W": "(RA + LA + LL) / 3)",
        "V1": "V1 - W",
        "V2": "V2 - W",
        "V3": "V3 - W",
        "V4": "V4 - W",
        "V5": "V5 - W",
        "V6": "V6 - W"
    }
}


def get_standard_montage(standard, channel_mode):
    """Retrieves the electrode placements or lead configurations for a given
    ECG standard.

    Parameters
    ----------
    standard: str {'12leads'}
        The ECG standard to retrieve.
    channel_mode: str {'electrodes', 'leads'}
        Specifies whether to return electrode placements or lead computations.

    Returns
    -------
    dict:
        A dictionary containing either the electrode placements or lead computations.

    Raises
    ------
    ValueError: If an unsupported standard is requested.
    """
    # Check standard
    if standard not in ('12leads'):
        raise ValueError('Unknown standard %s' % standard)
    if channel_mode not in ('leads', 'electrodes'):
        raise ValueError('Unknown mode %s' % channel_mode)
    # Get montage
    if standard == '12leads':
        standard_channels = ECG_12LEADS[channel_mode]
    else:
        standard_channels = None
    return standard_channels


class ECGChannelSet(components.SerializableComponent):
    """Class to represent an ECG montage with ordered channels in specific
    coordinates. It also provides functionality to load channels from ECG
    standards directly from the labels.
    """

    def __init__(self, channel_mode='leads'):
        """Constructor of class ECGChannelSet

        Parameters
        ----------
        channel_mode: str {'leads', 'electrodes'}
            If 'leads', it is assumed that the channels are leads. If 'electrodes', it is assumed that
            channels are voltage difference between each electrode and the ground.
        """

        # Check errors
        if channel_mode not in ('leads', 'electrodes'):
            raise ValueError('Unknown mode %s' % channel_mode)

        # Set attributes
        self.channel_mode = channel_mode
        self.channels = None
        self.n_cha = None
        self.l_cha = None
        self.montage = None
        self.ground = None

    def set_ground(self, ground):
        """Sets the ground of the montage

        Parameters
        ----------
        ground: str
            Label of the ground. Standard ECG uses channel RL
        """
        self.ground = ground

    def add_channel(self, label, descr=None):
        """Function to add a channel to the end of the current montage.
        Take into account that the order of the channels is important!

        Parameters
        ----------
        label: str
            Label of the channel. If mode is 'leads' this label must represent a lead.
            If mode is 'leads' this label must represent an electrode.
        descr: str (optional)
            Description of the channel. If mode is 'leads' this description should say how
            it has been computed. If mode is 'electrodes', this description should include
            the location of the electrode.

        See Also
        --------
        get_standard_channel_data_from_label: returns channel data given the
            channel label and the standard. It can be used to get the reference
        """
        channel = {'label': label,
                   'descr': descr}
        channels = list() if self.channels is None else self.channels
        channels.append(channel)
        # Check channels
        self.__check_channels(channels)
        # Store attributes
        self.channels = channels
        self.n_cha = len(self.channels)
        self.l_cha = [cha['label'] for cha in self.channels]

    def set_montage(self, channels, ground=None):
        """Sets a custom montage, overwriting the previous one. Add single
        channels more easily using function add_channel and
        add_standard_channel.

        Parameters
        ----------
        channels : list
            List of dicts, each of them representing a channel. The dict must
            contain the label, and the description of the channel. If mode is 'leads'
            this label must represent a lead. If mode is 'leads' this label must represent
            an electrode.
        ground : dict
            Dict containing the label and description of the ground electrode

        See Also
        --------
        set_standard_montage: preferred choice in most cases
        """
        # Check errors
        self.__check_channels(channels, ground)
        # Set attributes
        self.channels = channels
        self.ground = ground
        self.n_cha = len(self.channels)
        self.l_cha = [cha['label'] for cha in self.channels]

    def set_standard_montage(self, l_cha=None, l_ground=None, montage='12leads'):
        """Set standard ECG channels with common reference. In 3 dimensions,
        the equator is taken a Nz-T10-Iz-T9.

        Parameters
        ----------
        l_cha : list, optional
            List of channels labels. The data will be returned keeping the
            same order. If None, the channels will be returned in the same order
            as they appear in the corresponding standard in medusa.meeg
        l_ground : str, optional
            Label of the ground. Usual choices are AFz or FPz.
        montage : str {'12leads'} or dict
            ECG standard. If it's a string, the corresponding labels and
            locations of the standard channels will be loaded using the
            standards defined in this module. To load a different montage,
            pass a dict the same structure here.
        """
        # Get montage
        if isinstance(montage, str):
            # Load standard montage
            self.montage = montage
            montage = get_standard_montage(
                standard=montage, channel_mode=self.channel_mode)
        else:
            # Set custom montage
            montage = montage.copy()
            self.montage = montage
        # Get list of labels to get
        labels = montage.keys() if l_cha is None \
            else [l.upper().strip() for l in l_cha]
        # Get channels
        channels = list()
        for label in labels:
            # Append info
            if label in montage:
                channel = {'label': label,
                           'descr': montage[label]}
            else:
                raise ChannelNotFound(label)
            channels.append(channel)
        # Ground
        if l_ground is not None:
            if l_ground in montage:
                ground = {'label': l_ground,
                          'descr': montage[l_ground]}
            else:
                raise ChannelNotFound(l_ground)
        else:
            ground = None
        # Check channels
        self.set_montage(channels, ground=ground)

    def __check_channels(self, channels, ground=None):
        # Get mandatory and coordinates keys for each dim and coord_system
        cha_keys = ['label', 'descr']
        gnd_keys = ['label', 'descr']
        # Check keys
        for cha in channels:
            if not all(k in cha for k in cha_keys):
                raise ValueError('Malformed channel %s. Dict keys must be %s' %
                                 (str(cha), str(cha_keys)))
        # Check ground
        if ground is not None:
            if not all(k in ground for k in gnd_keys):
                raise ValueError('Malformed ground. Dict keys must be %s' %
                                 (str(gnd_keys)))

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

    def to_serializable_obj(self):
        return self.__dict__

    @classmethod
    def from_serializable_obj(cls, dict_data):
        inst = cls()
        inst.__dict__.update(dict_data)
        return inst


class ECG(components.BiosignalData):
    """Electrocardiography (ECG) biosignal data class.
    """

    def __init__(self, times, signal, fs, channel_set, **kwargs):
        """ECG constructor

        Parameters
        ----------
        times : list or numpy.ndarray
            1D numpy array [n_samples]. Timestamps of each sample. If they are
            not available, generate them
            artificially. Nevertheless, all signals and events must have the
            same temporal origin
        signal : list or numpy.ndarray
            2D numpy array [n_samples x n_channels]. EMG samples (the units
            should be defined using kwargs)
        fs : int or float
            Sample rate of the recording.
        channel_set : ECGChannelSet
            Channel information
        kwargs: kwargs
            Key-value arguments to be saved in the class. This general class
            does not check anything
        """

        # Standard attributes
        self.times = times
        self.signal = signal
        self.fs = fs
        self.channel_set = channel_set

        # Set the specified arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def change_channel_set(self, channel_set):
        """Smart change of channel set, updating the signal and all related
        attributes

        Parameters
        ----------
        channel_set : ECGChannelSet
            ECG channel set
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
            if type(rec_dict[key]) == ECGChannelSet:
                rec_dict[key] = rec_dict[key].to_serializable_obj()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        return cls(**dict_data)


class ChannelNotFound(Exception):

    def __init__(self, l_cha):
        super().__init__(
            'Channel %s is not defined in the current montage' % l_cha)


class UnlocatedChannel(Exception):

    def __init__(self, l_cha):
        super().__init__(
            'Channel %s does not contain proper description' % l_cha)