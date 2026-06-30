"""Tests for the single-source-of-truth EEG montage system.

One 3-D master (``eeg_standard_3D.tsv``); 10-20/10-10/10-05 are subsets; the 2-D
layout is the azimuthal projection of the 3-D coordinates; synonyms (Jasper
T3-T6, ear/mastoid refs, ...) resolve to the same coordinates.
"""

import warnings

import numpy as np
import pytest

from medusa.core.data import ChannelSet
from medusa.core.data.eeg import montages as M

SYNONYM_PAIRS = [('T3', 'T7'), ('T4', 'T8'), ('T5', 'P7'), ('T6', 'P8'),
                 ('A1', 'LPA'), ('A2', 'RPA'), ('O9', 'I1'), ('O10', 'I2'),
                 ('M1', 'TP9'), ('M2', 'TP10')]


@pytest.mark.parametrize("syn, canon", SYNONYM_PAIRS)
def test_synonyms_resolve_to_identical_coordinates(syn, canon):
    m = M.get_standard_montage('10-05', dim='3D')
    assert syn in m and canon in m
    assert m[syn] == m[canon]


def test_resolve_label():
    assert M.resolve_label('t3') == 'T7'
    assert M.resolve_label(' a1 ') == 'LPA'
    assert M.resolve_label('Cz') == 'CZ'      # unknown synonym -> upper, unchanged


def test_standards_are_nested_in_size():
    n20 = len(M.get_standard_montage('10-20', dim='3D'))
    n10 = len(M.get_standard_montage('10-10', dim='3D'))
    n05 = len(M.get_standard_montage('10-05', dim='3D'))
    assert n20 < n10 < n05


def test_temporal_chain_present_in_10_20():
    m = M.get_standard_montage('10-20', dim='3D')
    for lab in ('T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'P7', 'P8'):
        assert lab in m


def test_build_set_with_jasper_temporals_and_linked_mastoids():
    # The previously-broken case: Jasper temporal names + linked-mastoid refs.
    labels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz',
              'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    cs = ChannelSet()
    with warnings.catch_warnings():
        warnings.simplefilter('error')        # any "missing coordinates" -> fail
        cs.add_unipolar_eeg_channels(labels, reference=['M1', 'M2'])
    pos = cs.get_positions('EEG')
    assert pos.shape == (19, 3)
    assert not np.isnan(pos).any()


def test_2d_is_projection_of_3d():
    m3 = M.get_standard_montage('10-05', dim='3D')
    m2 = M.get_standard_montage('10-05', dim='2D')
    assert set(m2) == set(m3)
    for lab in m3:
        xyz = np.array([m3[lab]['x'], m3[lab]['y'], m3[lab]['z']])
        proj = M.project_to_2d(xyz.reshape(1, 3))[0]
        assert np.allclose([m2[lab]['x'], m2[lab]['y']], proj, atol=1e-9)


def test_2d_landmarks_match_topographic_layout():
    # File-independent: reproduces the historical eeg_standard_2D.tsv layout.
    m2 = M.get_standard_montage('10-05', dim='2D')
    np.testing.assert_allclose([m2['CZ']['x'], m2['CZ']['y']], [0, 0], atol=1e-9)
    np.testing.assert_allclose([m2['NZ']['x'], m2['NZ']['y']], [0, 1], atol=1e-9)
    np.testing.assert_allclose([m2['T7']['x'], m2['T7']['y']], [-0.8, 0], atol=1e-3)
    np.testing.assert_allclose([m2['T8']['x'], m2['T8']['y']], [0.8, 0], atol=1e-3)


def test_convenience_functions():
    assert M.montage_10_20() == M.get_standard_montage('10-20')
    assert M.montage_10_10('2D') == M.get_standard_montage('10-10', '2D')
    assert M.montage_10_05() == M.get_standard_montage('10-05')


def test_spherical_output_shape():
    m = M.get_standard_montage('10-20', dim='3D', coord_system='spherical')
    assert set(m['CZ']) == {'r', 'theta', 'phi'}


def test_invalid_inputs():
    with pytest.raises(AssertionError):
        M.get_standard_montage('10-15')
    with pytest.raises(AssertionError):
        M.get_standard_montage('10-20', dim='4D')
    with pytest.raises(AssertionError):
        M.get_standard_montage('10-20', coord_system='polar')


def test_project_to_2d_passthrough_and_errors():
    p = np.array([[0.1, 0.2], [0.3, 0.4]])
    np.testing.assert_array_equal(M.project_to_2d(p), p)   # 2-D passthrough
    with pytest.raises(ValueError):
        M.project_to_2d(np.zeros((3, 4)))


def test_get_standard_montage_labels_excludes_refs_by_default():
    base = M.get_standard_montage_labels('10-20')
    assert not ({'A1', 'A2', 'M1', 'M2'} & set(base))
    full = M.get_standard_montage_labels('10-20', include_references=True)
    assert {'A1', 'A2', 'M1', 'M2'} <= set(full)


def test_get_coordinates_resolves_and_subsets():
    full = M.get_coordinates()
    assert 'T7' in full and 'FZ' in full
    sub = M.get_coordinates(['T3', 'Fz'])
    assert set(sub) == {'T3', 'FZ'}
    assert sub['T3'] == full['T7']            # synonym resolved to master coords


def test_add_unipolar_without_montage_uses_master():
    cs = ChannelSet().add_unipolar_eeg_channels(['Fz', 'Cz', 'Pz', 'T3'])
    pos = cs.get_positions('EEG')
    assert pos.shape == (4, 3) and not np.isnan(pos).any()


def test_add_unipolar_full_standard_via_labels():
    labels = M.get_standard_montage_labels('10-20')
    cs = ChannelSet().add_unipolar_eeg_channels(labels)
    assert cs.n_channels == len(labels)
    assert not np.isnan(cs.get_positions('EEG')).any()


def test_add_unipolar_rejects_standard_name():
    with pytest.raises(TypeError):
        ChannelSet().add_unipolar_eeg_channels(['Fz'], montage='10-20')
