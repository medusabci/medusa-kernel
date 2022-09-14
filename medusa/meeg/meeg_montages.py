import csv, math, os

# Synonyms
eeg_channel_synonyms = {
    'I1': 'O9',
    'I2': 'O10',
    'O9': 'I1',
    'O10': 'I2',
    'T3': 'T7',
    'T4': 'T8',
    'T7': 'T3',
    'T8': 'T4',
    'T5': 'P7',
    'T6': 'P8'
}


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
    montage = read_montage_file(
        path='%s/eeg_standard_%s_%s.tsv' % (folder, standard, dim),
        file_format='tsv', dim=dim, coord_system='cartesian')

    if coord_system == 'spherical':
        montage = switch_coordinates_system(montage, dim, 'cartesian')

    return montage


def switch_coordinates_system(montage, dim, coord_system):
    """Switch the coordinates system of the montage. If the the coordinates are
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
    # ACCEPTED PARAMSN
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
    montage = get_standard_montage('10-20', '2D', 'spherical')
    print(montage)
