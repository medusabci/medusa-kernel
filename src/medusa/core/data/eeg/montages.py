"""Standard EEG montages from a single source of truth.

All electrode coordinates come from **one** file, ``eeg_standard.tsv``
(unit-sphere cartesian; ``+x`` right, ``+y`` anterior, ``+z`` vertex). The
standard systems are subsets of it:

* :func:`get_standard_montage` (or :func:`montage_10_20` / :func:`montage_10_10`
  / :func:`montage_10_05`) returns a system as ``{label: coordinates}``.
* The 2-D topographic layout is **derived** from the 3-D master by azimuthal-
  equidistant projection (:func:`project_to_2d`) — there is no separate 2-D file.
* :data:`SYNONYMS` lets a montage be requested by either nomenclature (e.g. the
  Jasper 1958 names ``T3/T4/T5/T6`` resolve to the MCN names ``T7/T8/P7/P8``).
"""

import csv
import functools
import math
import os

import numpy as np

__all__ = [
    # Standard montages (subsets of the single 3-D coordinate master)
    "get_standard_montage",
    "montage_10_20",
    "montage_10_10",
    "montage_10_05",
    "get_standard_montage_labels",
    # Coordinate lookup / label resolution
    "get_coordinates",
    "resolve_label",
    # Projection / coordinate-system conversion
    "project_to_2d",
    "switch_coordinates_system",
    # Montage file I/O and label utilities
    "read_montage_file",
    "get_camel_case_labels",
    # Data constants
    "SYNONYMS",
    "REFERENCE_ELECTRODES",
]

# According to Jasper (1958)
EEG_10_20 = [
    'NZ',
    'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T3', 'C3', 'CZ', 'C4', 'T4',
    'T5', 'P3', 'PZ', 'P4', 'T6',
    'O1', 'O2',
    'A1', 'A2', 'M1', 'M2'
]

# According to Nuwer et al. (1998)
# doi.org/10.1016/S0013-4694(97)00106-5
EEG_10_10 = [
    'NZ',
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AFZ', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T3', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T4',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'T5', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'T6',
    'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
    'O1', 'OZ', 'O2',
    'A1', 'A2', 'M1', 'M2'
]

# According to Oostenveld & Praamstra (2001)
# doi.org/10.1016/S1388-2457(00)00527-7
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
    'PO4H', 'PO6H', 'PO3H', 'PO5H', 'POO4', 'POO3',
    'A1', 'A2', 'M1', 'M2'
]

# Alternative labels -> canonical name in the 3-D master. Lets a montage be
# requested by either nomenclature. Source: the historical
# _eeg_standard/_merge_all_montages.py synonym table.
SYNONYMS = {
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',    # Jasper 1958 -> MCN
    'A1': 'LPA', 'A2': 'RPA',                           # earlobe refs -> preauricular
    'O9': 'I1', 'O10': 'I2',                            # inferior-occipital naming
    'M1': 'TP9', 'M2': 'TP10',                          # mastoid refs ~ TP9/TP10
}

# Reference/earlobe/mastoid sites -- excluded from get_standard_montage_labels by
# default (they are reference options, not scalp recording electrodes).
REFERENCE_ELECTRODES = {'A1', 'A2', 'M1', 'M2'}


@functools.lru_cache(maxsize=1)
def _load_master():
    """Load the 3-D coordinate master (cached): ``{LABEL: {'x','y','z'}}``."""
    path = os.path.join(os.path.dirname(__file__), 'eeg_standard.tsv')
    return read_montage_file(path, file_format='tsv', dim='3D',
                             coord_system='cartesian')


def resolve_label(label):
    """Resolve a channel label to its canonical name in the coordinate master.

    Upper-cases and strips the label, then maps a known synonym (e.g. ``'T3'``
    -> ``'T7'``); unknown labels are returned unchanged.
    """
    label = str(label).upper().strip()
    return SYNONYMS.get(label, label)


def project_to_2d(positions):
    """Azimuthal-equidistant projection of 3-D scalp positions to 2-D.

    The vertex (``+z``) maps to the origin, ``+y`` is anterior (up) and ``+x``
    right; the equator (``z=0``) maps to unit radius. Accepts an ``(n, 3)``
    array (projected) or ``(n, 2)`` array (returned unchanged); ``NaN`` / origin
    rows pass through as ``NaN``. This is the canonical 3-D -> 2-D montage
    projection; it reproduces the historical 2-D layout to rounding, and
    ``medusa.plots`` re-exports it.
    """
    pos = np.asarray(positions, dtype=float)
    if pos.ndim != 2 or pos.shape[1] not in (2, 3):
        raise ValueError(
            'positions must be (n, 2) or (n, 3), got shape %r' % (pos.shape,))
    if pos.shape[1] == 2:
        return pos.copy()
    out = np.full((pos.shape[0], 2), np.nan, dtype=float)
    norm = np.linalg.norm(pos, axis=1)
    ok = np.isfinite(norm) & (norm > 0)
    unit = pos[ok] / norm[ok, None]
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))   # polar angle from +z
    phi = np.arctan2(unit[:, 1], unit[:, 0])            # +y anterior, +x right
    r = theta / (np.pi / 2.0)                            # equator -> 1.0
    out[ok, 0] = r * np.cos(phi)
    out[ok, 1] = r * np.sin(phi)
    return out


def _project_montage_to_2d(montage):
    """Project a ``{label: {'x','y','z'}}`` montage to ``{label: {'x','y'}}``."""
    out = {}
    for label, c in montage.items():
        xyz = np.array([c['x'], c['y'], c.get('z', 0.0)], dtype=float)
        xy = project_to_2d(xyz.reshape(1, 3))[0]
        out[label] = {'x': float(xy[0]), 'y': float(xy[1])}
    return out


def get_camel_case_labels(ch_labels):
    """ Converts a given list of channel labels to the common camel-case
    format (e.g., 'FPZ' -> 'FPz').
    """
    assert isinstance(ch_labels, list)
    camel_case_labels = []
    for ch in ch_labels:
        new_ch = ch.replace('Z', 'z').replace('H','h').replace('FP','Fp')
        camel_case_labels.append(new_ch)
    return camel_case_labels


def get_standard_montage(standard='10-05', dim='3D', coord_system='cartesian'):
    """Return a standard EEG montage as ``{label: coordinates}``.

    All coordinates come from the single 3-D master (``eeg_standard.tsv``);
    ``standard`` selects a subset of it. Each electrode is exposed under **both**
    its traditional and canonical name (e.g. both ``'T3'`` and ``'T7'``), via
    :data:`SYNONYMS`. For ``dim='2D'`` the positions are the azimuthal projection
    of the 3-D coordinates (:func:`project_to_2d`).

    Parameters
    ----------
    standard : {'10-20', '10-10', '10-05'}
        Electrode system (subset of the master).
    dim : {'2D', '3D'}
        ``'3D'`` returns master coordinates; ``'2D'`` returns the topographic
        projection.
    coord_system : {'cartesian', 'spherical'}
        Output coordinate system.

    Returns
    -------
    dict
        ``{label: {'x','y'[, 'z']}}`` (cartesian) or ``{label: {'r','theta'[,
        'phi']}}`` (spherical).
    """
    assert standard in ('10-20', '10-10', '10-05'), \
        'standard must be one of {"10-20", "10-10", "10-05"}'
    assert dim in ('2D', '3D'), 'dim must be one of {"2D", "3D"}'
    assert coord_system in ('cartesian', 'spherical'), \
        'coord_system must be one of {"cartesian", "spherical"}'

    membership = {'10-20': EEG_10_20, '10-10': EEG_10_10,
                  '10-05': EEG_10_05}[standard]
    master = _load_master()

    montage = {}
    for label in membership:
        canon = resolve_label(label)
        if canon in master:
            montage[str(label).upper().strip()] = dict(master[canon])

    # Expose each electrode under BOTH its traditional and canonical name.
    for syn, canon in SYNONYMS.items():
        if syn in montage:
            montage.setdefault(canon, dict(montage[syn]))
        elif canon in montage:
            montage.setdefault(syn, dict(montage[canon]))

    if dim == '2D':
        montage = _project_montage_to_2d(montage)
    if coord_system == 'spherical':
        montage = switch_coordinates_system(montage, dim, 'cartesian')
    return montage


def montage_10_20(dim='3D', coord_system='cartesian'):
    """The 10-20 montage (see :func:`get_standard_montage`)."""
    return get_standard_montage('10-20', dim, coord_system)


def montage_10_10(dim='3D', coord_system='cartesian'):
    """The 10-10 montage (see :func:`get_standard_montage`)."""
    return get_standard_montage('10-10', dim, coord_system)


def montage_10_05(dim='3D', coord_system='cartesian'):
    """The 10-05 montage (see :func:`get_standard_montage`)."""
    return get_standard_montage('10-05', dim, coord_system)


def get_standard_montage_labels(standard, include_references=False):
    """Return the electrode labels of a standard system.

    Because every electrode shares one coordinate in the 3-D master, a standard
    system is just a named set of labels. Pass the result straight to
    :meth:`~medusa.core.data.ChannelSet.add_unipolar_eeg_channels` to build the
    whole montage.

    Parameters
    ----------
    standard : {'10-20', '10-10', '10-05'}
        Electrode system.
    include_references : bool
        If ``False`` (default), the reference/earlobe/mastoid sites
        (:data:`REFERENCE_ELECTRODES`) are excluded -- they are reference
        options, not scalp recording electrodes.

    Returns
    -------
    list of str
        The electrode labels, in canonical order.
    """
    assert standard in ('10-20', '10-10', '10-05'), \
        'standard must be one of {"10-20", "10-10", "10-05"}'
    membership = {'10-20': EEG_10_20, '10-10': EEG_10_10,
                  '10-05': EEG_10_05}[standard]
    master = _load_master()
    labels = [lab for lab in membership if resolve_label(lab) in master]
    if not include_references:
        labels = [lab for lab in labels
                  if str(lab).upper().strip() not in REFERENCE_ELECTRODES]
    return labels


def get_coordinates(labels=None, dim='3D', coord_system='cartesian'):
    """Resolve labels to coordinates from the single 3-D master.

    With ``labels=None`` the whole master is returned (canonical names);
    otherwise only the requested labels, with synonyms resolved (e.g. ``'T3'``
    -> ``'T7'`` coordinates) and unknown labels omitted.

    Parameters
    ----------
    labels : list of str or None
        Channel labels to resolve, or ``None`` for the entire master.
    dim : {'2D', '3D'}
        ``'3D'`` master coordinates, or ``'2D'`` azimuthal projection.
    coord_system : {'cartesian', 'spherical'}
        Output coordinate system.

    Returns
    -------
    dict
        ``{label: {'x','y'[, 'z']}}`` (cartesian) or the spherical equivalent.
    """
    master = _load_master()
    if labels is None:
        montage = {key: dict(val) for key, val in master.items()}
    else:
        montage = {}
        for label in labels:
            canon = resolve_label(label)
            if canon in master:
                montage[str(label).upper().strip()] = dict(master[canon])
    if dim == '2D':
        montage = _project_montage_to_2d(montage)
    if coord_system == 'spherical':
        montage = switch_coordinates_system(montage, dim, 'cartesian')
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
    m = get_standard_montage('10-20', dim='3D')
    print('10-20 labels (incl. synonyms):', len(m))
    print("T3 resolves:", 'T3' in m, '| identical to T7:', m.get('T3') == m.get('T7'))
    print("A1 (-> LPA):", m.get('A1'))
    print("M1 (-> TP9):", m.get('M1'))
