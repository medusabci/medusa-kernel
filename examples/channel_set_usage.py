"""Build BIDS-aligned channel sets with ``medusa.core.data``.

Run:  python examples/channel_set_usage.py

Walks through the D1 channel model (``Channel`` / ``Sensor`` / ``ChannelSet``):

  1. A full unipolar EEG montage from a standard layout.
  2. A mixed-modality set (EEG + EOG + EMG + TRIG), built additively.
  3. Reference schemes: common, CAR (average of all), linked mastoids, bipolar.
  4. Selecting / reordering channels and reading sensor positions.
  5. Custom montages (3-D and 2-D) and standalone sensors.
  6. Serialization round-trip.
  7. The validation errors raised on malformed sets.

Recurring ideas: a *channel* is a data column (``channels.tsv``); a *sensor* is a
physical transducer (``electrodes.tsv``); they link by ``label``. Register sensors
first, then add channels additively -- each add enforces consistency eagerly, so
the set needs no separate finalisation step.
"""
import warnings

import numpy as np

from medusa.core.data import (Channel, Sensor, ChannelSet,
                              BIDS_CHANNEL_TYPES, REFERENCE_METHODS)
from medusa.core.data.eeg import (get_standard_montage,
                                  get_standard_montage_labels)


def section(title):
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


# ---------------------------------------------------------------------------- #
# 1. Full unipolar EEG montage (the common case)
# ---------------------------------------------------------------------------- #
# `add_unipolar_eeg_channels` builds one EEG Channel + one located Sensor per
# label, plus the reference and ground sensors. Coordinates come from the single
# standard master, so you just pass labels -- a hand-picked subset, or a whole
# standard system via `get_standard_montage_labels`.
section("1. Full unipolar EEG (standard labels, common reference)")

eeg = ChannelSet()
eeg.add_unipolar_eeg_channels(
    ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2'],
    reference='Cz', ground='Fpz')
print("the full 10-20 system has",
      len(get_standard_montage_labels('10-20')), "scalp labels (refs excluded):")
print(" ", get_standard_montage_labels('10-20'))

print("channels      :", eeg.labels)
print("n_channels    :", eeg.n_channels)
print("modalities    :", set(eeg.types))
print("reference     :", eeg.reference_method, "(derived from the channels)")
print("sensors (incl. ref/ground, not data columns):", list(eeg.sensors))
print("\nchannels.tsv (first 4 rows):")
print(eeg.to_channels_dataframe().head(4).to_string(index=False))
print("\nelectrodes.tsv (first 4 rows):")
print(eeg.to_sensors_dataframe().head(4).to_string(index=False))


# ---------------------------------------------------------------------------- #
# 2. Mixed-modality set: EEG + EOG + EMG + TRIG
# ---------------------------------------------------------------------------- #
# One Signal can mix channel types. Build the EEG block, then add other columns
# with `add_channels`. A channel links to its sensor(s) by label only, so register
# the sensors first with `add_sensors`; naming a sensor label that is not registered
# raises on add.
section("2. Mixed EEG + EOG + EMG + TRIG (built sensors-first)")

mixed = ChannelSet()
mixed.add_unipolar_eeg_channels(['F3', 'F4', 'C3', 'C4', 'P3', 'P4'],
                                reference='Cz', ground='Fpz')

# Bipolar VEOG/HEOG: register the active + subtracted sensors, then the channel.
mixed.add_sensors(Sensor('VEOG_up', coordinates=[0.0, 0.95, 0.30]))
mixed.add_sensors(Sensor('VEOG_down', coordinates=[0.0, 0.92, 0.10]))
mixed.add_channels(Channel('VEOG', 'VEOG', 'uV', sensor='VEOG_up',
                          reference='VEOG_down', reference_method='bipolar'))
mixed.add_sensors([Sensor('HEOG_l'), Sensor('HEOG_r')])
mixed.add_channels(Channel('HEOG', 'HEOG', 'uV', sensor='HEOG_l',
                          reference='HEOG_r', reference_method='bipolar'))
# Surface EMG placed by anatomy (no digitised position — `location` instead).
mixed.add_sensors(Sensor('EMG-neck', location='left trapezius',
                        sensor_type='surface'))
mixed.add_channels(Channel('EMG-neck', 'EMG', 'uV', sensor='EMG-neck'))
# A trigger column: a data column with no sensor and no reference.
mixed.add_channels(Channel('TRIG', 'TRIG', 'n/a'))

print("channels      :", mixed.labels)
print("modalities    :", mixed.types)
print("reference     :", mixed.reference_method, "(channels disagree -> 'mixed')")
print("VEOG links     -> sensor:", mixed.get_channel('VEOG').sensor,
      "reference:", mixed.get_channel('VEOG').reference)
print("EMG placement :", mixed.get_sensor('EMG-neck').location)


# ---------------------------------------------------------------------------- #
# 3. Reference schemes (scheme + operand are two separate fields)
# ---------------------------------------------------------------------------- #
# `reference_method` is the SCHEME (common/average/bipolar); `reference` is the
# OPERAND (sensor label(s) or None). Together they fold into BIDS' single
# `reference` column on export.
section("3. Reference schemes: common / CAR / linked mastoids / bipolar")

car = ChannelSet()
car.add_unipolar_eeg_channels(['F3', 'F4', 'C3', 'C4'],
                              reference=None, reference_method='average')  # CAR

# M1/M2 (mastoid refs) resolve to their canonical coordinates (TP9/TP10), so
# this needs no "missing coordinates" warning suppression any more.
linked = ChannelSet()
linked.add_unipolar_eeg_channels(['F3', 'F4'],
                                 reference=['M1', 'M2'])          # -> 'average'

print("CAR            : method =", car.reference_method,
      "| per-channel reference =", car.get_channel('F3').reference)
print("linked mastoids: method =", linked.reference_method,
      "| reference =", linked.get_channel('F3').reference)
print("\nchannels.tsv `reference` column folds scheme+operand:")
print("  CAR    ->", list(car.to_channels_dataframe()['reference'])[:1], "...")
print("  linked ->", list(linked.to_channels_dataframe()['reference'])[:1], "...")
mixed_df = mixed.to_channels_dataframe()
veog_ref = mixed_df.loc[mixed_df['name'] == 'VEOG', 'reference'].iloc[0]
print("  bipolar->", [veog_ref], "(VEOG-VEOG_down)")


# ---------------------------------------------------------------------------- #
# 4. Selecting channels and reading positions
# ---------------------------------------------------------------------------- #
# `pick` returns a sub-ChannelSet AND the column indices (to slice the matching
# signal columns). `get_positions` resolves each channel to its active sensor's
# 3-D coordinates; columns with no/located sensor come back as NaN.
section("4. pick / index / get_positions")

eeg_only, idx = mixed.pick(types='EEG')
print("pick('EEG')   : labels =", eeg_only.labels, "| signal column idx =", idx)
print("index(['C4','F3']) =", mixed.index(['C4', 'F3']))

pos = mixed.get_positions('EEG')
print("EEG positions shape:", pos.shape)
trig_pos = mixed.get_positions(None)[mixed.index('TRIG')[0]]
print("TRIG position (no sensor) is NaN:", bool(np.isnan(trig_pos).all()))


# ---------------------------------------------------------------------------- #
# 5. Custom montages and standalone sensors
# ---------------------------------------------------------------------------- #
# `montage` also accepts a custom dict {label: coords}. Standard montages are
# 3-D (BIDS requires x/y/z for EEG); a custom montage may be 2-D (it warns, since
# BIDS EEG export wants z). `add_sensors` registers standalone sensors (a ground
# or shared reference) that aren't carried by any single channel.
section("5. Custom montages (3-D / 2-D) + standalone sensors")

# Hand-built custom 3-D montage (values may be {x,y,z} dicts or [x,y,z] lists).
custom_3d = {'A1': {'x': 0.0, 'y': 0.0, 'z': 1.0}, 'A2': [0.5, 0.0, 0.86]}
cs3d = ChannelSet()
cs3d.add_unipolar_eeg_channels(['A1', 'A2'], montage=custom_3d)
print("custom 3-D montage positions:\n", cs3d.get_positions('EEG'))

# Unipolar EEG in 2-D: the standard montages are 3-D-only, so to work in a flat
# 2-D topographic layout, load the 2-D coordinates from the eeg module and pass
# them in as a *custom* montage. This warns (BIDS EEG export wants z), but is the
# supported way to get real EEG labels with 2-D positions.
layout_2d = get_standard_montage('10-20', dim='2D', coord_system='cartesian')
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    eeg_2d = ChannelSet()
    eeg_2d.add_unipolar_eeg_channels(['Fz', 'Cz', 'Pz', 'C3', 'C4', 'O1', 'O2'],
                                     montage=layout_2d)
    warned = any('2-D' in str(x.message) for x in w)
print("\nunipolar EEG in 2-D via custom montage (BIDS warning:", warned, ")")
print("  positions shape:", eeg_2d.get_positions('EEG').shape, "(n_channels x 2)")
print("  electrodes.tsv z:", list(eeg_2d.to_sensors_dataframe()['z'])[:3], "...")

ground_set = ChannelSet()
ground_set.add_unipolar_eeg_channels(['F3', 'F4', 'C3', 'C4'],
                                     reference=None, reference_method='average')
ground_set.add_sensors(Sensor('Fpz-gnd', coordinates=[0.0, 1.0, 0.0],
                             sensor_type='ground'))   # standalone, not a column
print("standalone ground registered:", 'Fpz-gnd' in ground_set.sensors,
      "| n_channels still", ground_set.n_channels)


# ---------------------------------------------------------------------------- #
# 6. Serialization round-trip (multiplatform .rec contract)
# ---------------------------------------------------------------------------- #
# Every component is a SerializableComponent: to/from a plain dict (json/bson/mat
# backends). Sensors persist as a list of rows and rebuild as a dict; channels
# persist their sensor links as plain labels.
section("6. Serialization round-trip")

obj = mixed.to_serializable_obj()
restored = ChannelSet.from_serializable_obj(obj)
print("labels preserved      :", restored.labels == mixed.labels)
print("sensors rebuilt as dict:", isinstance(restored.sensors, dict),
      "| keys =", list(restored.sensors)[:4], "...")
print("VEOG reference kept   :", restored.get_channel('VEOG').reference)


# ---------------------------------------------------------------------------- #
# 7. Errors raised when building a malformed channel set
# ---------------------------------------------------------------------------- #
# The hard invariants are checked eagerly, at the offending call.
section("7. Validation errors")

# (a) Duplicate channel label.
try:
    cs = ChannelSet()
    cs.add_channels(Channel('C3', 'EEG', 'uV'))
    cs.add_channels(Channel('C3', 'EEG', 'uV'))
except ValueError as e:
    print("duplicate channel label ->", e)

# (b) Registering two DIFFERENT sensors under the same label.
try:
    cs = ChannelSet()
    cs.add_sensors(Sensor('C3', coordinates=[0, 0, 1]))
    cs.add_sensors(Sensor('C3', coordinates=[1, 1, 1]))
except ValueError as e:
    print("conflicting sensor     ->", e)

# (c) Invalid reference scheme.
try:
    Channel('x', 'EEG', 'uV', reference_method='monopolar')
except ValueError as e:
    print("bad reference_method   ->", e)

# (d) Bad montage type.
try:
    cs = ChannelSet()
    cs.add_unipolar_eeg_channels(['Fz'], montage=42)
except TypeError as e:
    print("bad montage type       ->", e)

# (e) Unknown channel label lookups.
try:
    cs = ChannelSet()
    cs.add_channels(Channel('Fz', 'EEG', 'uV'))
    cs.index(['Pz'])
except KeyError as e:
    print("unknown channel lookup ->", e)

# (f) Unknown ch_type is *not* an error: it is coerced to OTHER with a warning.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    ch = Channel('weird', 'FOO', 'uV')
    print(f"unknown ch_type 'FOO'  -> coerced to {ch.ch_type!r} (+ warning: "
          f"{any('vocabulary' in str(x.message) for x in w)})")

# (g) Naming a sensor label that was never registered raises on add (sensors-first).
try:
    cs = ChannelSet()
    cs.add_channels(Channel('O1', 'EEG', 'uV', sensor='O1'))   # 'O1' not registered
except ValueError as e:
    print("unregistered sensor    ->", e)


section("Reference: BIDS vocabularies")
print(f"{len(BIDS_CHANNEL_TYPES)} BIDS channel types, e.g.:",
      BIDS_CHANNEL_TYPES[:8], "...")
print("reference schemes:", REFERENCE_METHODS)


# ---------------------------------------------------------------------------- #
# 8. Plot the three standard systems (medusa.plots.scalp)
# ---------------------------------------------------------------------------- #
# The same coordinates drive the plots: build a ChannelSet per standard from its
# labels and draw each on a head with `scalp.plot_scalp`. Density grows
# 10-20 -> 10-10 -> 10-05; labels are shown only for the sparse 10-20.
section("8. Plot the three standard montages (medusa.plots.scalp)")

from pathlib import Path

import matplotlib.pyplot as plt
import medusa_style

from medusa.plots import plot_scalp, save_figure

with plt.style.context(medusa_style.mpl.rcparams()):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), layout='constrained')
    for ax, standard in zip(axes, ('10-20', '10-10', '10-05')):
        montage_set = ChannelSet().add_unipolar_eeg_channels(
            get_standard_montage_labels(standard))
        plot_scalp(montage_set, ax=ax, show_labels=(standard == '10-20'),
                   sensor_size=12 if standard != '10-05' else 4)
        ax.set_title(f"{standard}   ({montage_set.n_channels} channels)")
    out = Path(__file__).resolve().parent / "channel_set_montages.png"
    save_figure(fig, out)
print("saved montage figure ->", out)
if "agg" not in plt.get_backend().lower():
    plt.show()