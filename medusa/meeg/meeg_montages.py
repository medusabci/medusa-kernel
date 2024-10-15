import csv, math, os

EEG_10_20 = [
    'NZ', 'FP1', 'FPZ', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'OZ', 'O2',
    'A1', 'A2', 'M1', 'M2'
]

EEG_10_10 = [
    'NZ', 'FP1', 'FPZ', 'FP2', 'AF7', 'AFZ', 'AF8',
    'F9', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'F10',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P9', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO9', 'PO7', 'POZ', 'PO8', 'PO10', 'O1', 'OZ', 'O2',
    'I1', 'IZ', 'I2', 'A1', 'A2', 'M1', 'M2'
]

EEG_10_05 = [
    'T10', 'FT10', 'F10', 'NZ', 'F9', 'FT9', 'T9', 'TP9', 'P9', 'PO9',
    'I1', 'IZ', 'I2', 'PO10', 'P10', 'TP10', 'FTT10H', 'FFT10H',
    'FTT9H', 'FFT9H', 'TPP10H', 'PPO10H', 'POO10H', 'O2H', 'O1H',
    'POO9H', 'PPO9H', 'TPP9H', 'T8', 'FT8', 'F8', 'AF8', 'FP2', 'FPZ',
    'FP1', 'AF7', 'F7', 'FT7', 'T7', 'TP7', 'P7', 'PO7', 'O1', 'OZ',
    'O2', 'PO8', 'P8', 'TP8', 'AFZ', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ',
    'POZ', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'AFP4', 'AFP3', 'AF4H',
    'AF6H', 'AF3H', 'AF5H', 'AFF2H', 'AFF4H', 'AFF6H', 'AFF8H', 'AFF1H',
    'AFF3H', 'AFF5H', 'AFF7H', 'F4', 'F2', 'F6', 'F3', 'F1', 'F5',
    'FFC2H', 'FFC4H', 'FFC6H', 'FFT8H', 'FFC1H', 'FFC3H', 'FFC5H',
    'FFT7H', 'FC4', 'FC2', 'FC6', 'FC3', 'FC1', 'FC5', 'FCC2H', 'FCC4H',
    'FCC6H', 'FTT8H', 'FCC1H', 'FCC3H', 'FCC5H', 'FTT7H', 'CCP2H',
    'CCP4H', 'CCP6H', 'TTP8H', 'CCPH', 'CCP3H', 'CCP5H', 'TTP7H', 'CP4',
    'CP2', 'CP6', 'CP3', 'CP1', 'CP5', 'CPP2H', 'CPP4H', 'CPP6H', 'TPP8H',
    'CPPH', 'CPP3H', 'CPP5H', 'TPP7H', 'P4', 'P2', 'P6', 'P3', 'P1', 'P5',
    'PPO2H', 'PPO4H', 'PPO6H', 'PPO8H', 'PPOH', 'PPO3H', 'PPO5H', 'PPO7H',
    'PO4H', 'PO6H', 'PO3H', 'PO5H', 'POO4', 'POO3', 'A1', 'A2', 'M1', 'M2'
]


def get_standard_montage(standard, dim, coord_system):
    """Loads a standard from the ones included in MEDUSA
    """
    # Check errors
    assert standard == '10-20' or standard == '10-10' or standard == '10-05', \
        'Incorrect input on standard parameter. Possible values {"10-20", ' \
        '"10-10", "10-05"}'
    assert dim == '2D' or dim == '3D', \
        'Incorrect input on dim parameter. Possible values {"2D", "3D"}'
    assert coord_system == 'cartesian' or coord_system == 'spherical', \
        'Incorrect input on coord_system parameter. Possible values ' \
        '{"cartesian", "spherical"}'
    folder = os.path.dirname(__file__)
    eeg_channels = read_montage_file(
        path='%s/eeg_standard_%s.tsv' % (folder, dim),
        file_format='tsv', dim=dim, coord_system='cartesian')
    if standard == '10-20':
        standard_channels = EEG_10_20
    elif standard == '10-10':
        standard_channels = EEG_10_10
    elif standard == '10-05':
        standard_channels = EEG_10_05
    montage = {key: eeg_channels[key]
               for key in standard_channels if key in eeg_channels}
    if coord_system == 'spherical':
        montage = switch_coordinates_system(
            montage, dim, 'cartesian')
    return montage


def switch_coordinates_system(montage, dim, coord_system):
    """Switch the coordinates system of the montage. If the coordinates are
     cartesian, it switches to spherical and viceversa

    Parameters
    ----------
    montage: dict
        Montage data
    dim: str {'2D', '3D'}
        Current dimensions of the montage
    coord_system: str {'cartesian', 'spherical'}
        Current coordinates system
    """
    # Make copy to not modify the original dict
    new_montage = dict()
    # Make conversion
    for label, coordinates in montage.items():
        if coord_system == 'cartesian':
            cha_data = dict()
            if dim == '2D':
                cha_data['r'] = math.sqrt(
                    math.pow(montage[label]['x'], 2) +
                    math.pow(montage[label]['y'], 2))
                cha_data['theta'] = math.atan2(montage[label]['y'],
                                               montage[label]['x'])
            elif dim == '3D':
                cha_data['r'] = math.sqrt(
                    math.pow(montage[label]['x'], 2) +
                    math.pow(montage[label]['y'], 2) +
                    math.pow(montage[label]['z'], 2))
                cha_data['theta'] = math.atan2(
                    math.sqrt(math.pow(montage[label]['x'], 2) +
                              math.pow(montage[label]['y'], 2)),
                    montage[label]['z']
                )
                cha_data['phi'] = math.atan2(montage[label]['y'],
                                             montage[label]['x'])
            else:
                raise ValueError('Unknown number of dimensions')
        else:
            raise NotImplemented
        new_montage[label] = cha_data

    return new_montage


def read_montage_file(path, file_format, dim, coord_system):
    """This function reads a montage file.

    Parameters
    ----------
    path: str
        Path of the file
    file_format: str
        File format. Current accepted file formats:
            - csv (.csv): Comma separated values. Example:
                Format csv (cartesian, 2D):
                    label, x, y
                    C3, -0.3249, 0.0000
                    C4, 0.3249, 0.0000
                    Cz, 0.0000, 0.0000
                    ..., ..., ...
            - tsv (.tsv):  Tab separated values. Example:
                Format tsv (cartesian, 2D):
                    label	    x	        y
                    C3	        -0.3249	    0.0000
                    C4	        0.3249	    0.0000
                    Cz	        0.0000	    0.0000
                    ...         ...         ...
            - eeglab (.xyz): EEGLAB toolbox format. Example:
                Format eeglab (cartesian, 3D):
                    1	-3.277285	14.159082	-0.441064	AG001
                    2	-5.717206	13.304894	0.167594	AG002
                    3	-7.723112	12.035844	0.923781	AG003
                    ...         ...         ...
            - brainstorm (.txt): Brainstorm toolbox format. Example:
                Format brainstorm (cartesian, 3D):
                    C3	-4.025389	68.267399	112.962637
                    C4	-3.409963	-70.131718	111.811990
                    O1	-86.474171	34.971167	48.738844
                    ...         ...         ...
    dim: str
        Dimensions. Must be '2D' or '3D'
    coord_system: str
        For 2D, "cartesian" or "polar". For 3D "cartesian" or "spherical"
    """
    # ACCEPTED PARAMS
    file_formats = ['csv', 'tsv', 'eeglab', 'brainstorm']
    dims = ['2D', '3D']
    coord_systems = ['cartesian', 'spherical']

    # CHECK ERRORS
    if file_format not in file_formats:
        raise ValueError("Parameter file_format must be one of %s"
                         % file_formats)
    if dim not in dims:
        raise ValueError("Parameter dim must be one of %s" % dims)
    if coord_system not in coord_systems:
        raise ValueError("Parameter coord_system must be one of %s"
                         % coord_systems)

    # LOAD COORDINATES FILES
    if file_format == 'csv':
        montage = __read_txt_file(path, coord_system, dim, delimiter=',')
    elif file_format == 'tsv':
        montage = __read_txt_file(path, coord_system, dim, delimiter='\t')
    elif file_format == 'eeglab':
        montage = __read_eeglab_file(path, dim)
    elif file_format == 'brainstorm':
        montage = __read_brainstorm_file(path, dim)
    else:
        raise ValueError("Parameter file_format must be one of %s"
                         % file_formats)
    return montage


def __read_txt_file(path, coord_system, dim, delimiter='\t'):
    data = dict()
    # 2D coordinates
    with open(path) as f:
        tsvreader = csv.reader(f, delimiter=delimiter)
        for l, line in enumerate(tsvreader):
            if dim == '2D':
                label, coord1, coord2 = line
                if l == 0:
                    # Check format errors
                    if coord_system == 'cartesian':
                        if coord1 != 'x' or coord2 != 'y':
                            raise ValueError(
                                'Invalid header. '
                                'For dim=%s, '
                                'coord_system=%s, '
                                'the header must be [label, x, y]' %
                                (dim, coord_system)
                            )
                    elif coord_system == 'spherical':
                        if coord1 != 'r' or coord2 != 'theta':
                            raise ValueError(
                                'Invalid header. '
                                'For dim=%s, '
                                'coord_system=%s, '
                                'the header must be [label, r, theta]' %
                                (dim, coord_system)
                            )
                    else:
                        raise ValueError('Invalid coordinate system')
                    continue
                else:
                    lcha = line[0].upper()
                    data[lcha] = dict()
                    if coord_system == 'cartesian':
                        data[lcha]['x'] = float(coord1)
                        data[lcha]['y'] = float(coord2)
                    elif coord_system == 'spherical':
                        data[lcha]['r'] = float(coord1)
                        data[lcha]['theta'] = float(coord2)
                    else:
                        raise ValueError('Invalid coordinate system')
            elif dim == '3D':
                label, coord1, coord2, coord3 = line
                if l == 0:
                    # Check format errors
                    if coord_system == 'cartesian':
                        if coord1 != 'x' or coord2 != 'y' or coord3 != 'z':
                            raise ValueError(
                                'Invalid header. '
                                'For dim=%s, '
                                'coord_system=%s, '
                                'the header must be [label, x, y, z]' %
                                (dim, coord_system)
                            )
                    elif coord_system == 'spherical':
                        if coord1 != 'r' or coord2 != 'theta' \
                                or coord3 != 'phi':
                            raise ValueError(
                                'Invalid header. '
                                'For dim=%s, '
                                'coord_system=%s, '
                                'the header must be [label, r, theta, phi]' %
                                (dim, coord_system)
                            )
                    else:
                        raise ValueError('Invalid coordinate system')
                    continue
                else:
                    lcha = label.upper()
                    data[lcha] = dict()
                    if coord_system == 'cartesian':
                        data[lcha]['x'] = float(coord1)
                        data[lcha]['y'] = float(coord2)
                        data[lcha]['z'] = float(coord3)
                    elif coord_system == 'spherical':
                        data[lcha]['r'] = float(coord1)
                        data[lcha]['theta'] = float(coord2)
                        data[lcha]['phi'] = float(coord3)
                    else:
                        raise ValueError('Invalid coordinate system')
    return data


def __read_eeglab_file(path, dim):
    data = dict()
    # 2D coordinates
    with open(path) as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for l, line in enumerate(tsvreader):
            if dim == '2D':
                index, coord1, coord2, label = line
                # Save data
                lcha = label.upper()
                data[lcha] = dict()
                data[lcha]['x'] = float(coord1)
                data[lcha]['y'] = float(coord2)
            elif dim == '3D':
                index, coord1, coord2, coord3, label = line
                lcha = line[0].upper()
                data[lcha] = dict()
                data[lcha]['x'] = float(coord1)
                data[lcha]['y'] = float(coord2)
                data[lcha]['z'] = float(coord3)
    return data


def __read_brainstorm_file(path, dim):
    data = dict()
    # 2D coordinates
    with open(path) as f:
        tsvreader = csv.reader(f, delimiter='\t')
        for l, line in enumerate(tsvreader):
            if dim == '2D':
                label, coord1, coord2 = line
                # Save data
                lcha = label.upper()
                data[lcha] = dict()
                data[lcha]['x'] = float(coord1)
                data[lcha]['y'] = float(coord2)
            elif dim == '3D':
                label, coord1, coord2, coord3 = line
                lcha = line[0].upper()
                data[lcha] = dict()
                data[lcha]['x'] = float(coord1)
                data[lcha]['y'] = float(coord2)
                data[lcha]['z'] = float(coord3)
    return data


class ChannelNotFound(Exception):

    def __init__(self, l_cha):
        super().__init__(
            'Channel %s is not defined in the current montage' % l_cha)


class UnlocatedChannel(Exception):

    def __init__(self, l_cha):
        super().__init__(
            'Channel %s does not contain location coordinates' % l_cha)


if __name__ == '__main__':
    montage = get_standard_montage(
        '10-20', '2D', 'spherical')
    print(montage)
