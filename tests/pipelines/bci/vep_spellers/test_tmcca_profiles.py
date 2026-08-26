"""Tests for the ``TMCCAPipeline`` configuration profiles.

Three things are checked, because they are the three ways a profile goes wrong:

1. **it does not build** -- a profile is a settings payload written by hand, so a schema
   rename silently rots it. ``update_from_dict`` only *warns* on an unknown key, so the
   drift test turns warnings into errors;
2. **its values are not its defaults** -- if a profile sets values on top of the stock tree
   instead of building its own defaults, ``reset()`` wipes it and ``user_overrides()``
   reports the profile as if the user had typed it;
3. **the required mode stops being required** -- ``TMCCAPipeline()`` must keep raising, and
   the error must name the profiles.
"""
import pathlib
import warnings

import numpy as np
import pytest

from medusa.pipelines.bci.vep_spellers.decoding import template_matching
from medusa.pipelines.bci.vep_spellers import (
    generate_freq_codebook,
    TMCCAPipeline,
    tm_cca_settings,
    zerocal_ssvep_settings,
    cal_ssvep_settings,
    cvep_settings,
    uniform_weights,
    decaying_power_law_weights,
)

#: Every shipped profile: its builder, the name it records, and the mode it selects.
PROFILES = [
    (zerocal_ssvep_settings, "zerocal_ssvep", "synthetic_harmonics"),
    (cal_ssvep_settings, "cal_ssvep", "mixed_harmonics_template"),
    (cvep_settings, "cvep", "calibrated_template"),
]

CHANNELS = ["O1", "OZ", "O2", "POZ"]


class TestProfileSettings:

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_builds_a_valid_tree_without_warnings(self, builder, name, mode):
        """A profile builds cleanly and picks its mode (catches schema drift)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")          # an unknown/renamed key must fail loudly
            settings = builder()
        assert settings.validate() == []
        assert settings.to_dict()["reference"]["mode"] == mode

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_returns_a_fresh_tree_each_call(self, builder, name, mode):
        """Profiles are functions, not shared constants: the ctor stores trees by reference."""
        first, second = builder(), builder()
        assert first is not second
        first.set_value("car", value=True)
        assert second.to_dict()["car"] is False

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_constructs_a_pipeline(self, builder, name, mode):
        pipe = TMCCAPipeline(settings=builder(), channels=CHANNELS)
        assert pipe.cfg["reference"]["mode"] == mode
        assert pipe.cfg["channels"] == CHANNELS
        assert pipe.cfg["profile"] == name

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_profile_values_are_the_defaults(self, builder, name, mode):
        """The profile is the baseline: only the user's own edits are overrides."""
        pipe = TMCCAPipeline(settings=builder(), channels=CHANNELS, car=True)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS, "car": True}

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_reset_restores_the_profile_not_the_stock_schema(self, builder, name, mode):
        """``reset()`` (and the GUI's 'Reset all') must return to the profile."""
        profiled = builder().to_dict()
        pipe = TMCCAPipeline(settings=builder(), channels=CHANNELS)
        pipe.settings.set_value("car", value=True)
        pipe.settings.get_item("freq_filtering", "filterbank").set_elements(
            [{"filt_type": "iir", "band_type": "bandpass", "cutoff": [3.0, 30.0], "order": 2}])
        pipe.settings.reset()

        assert pipe.cfg["car"] == profiled["car"]
        assert pipe.cfg["freq_filtering"] == profiled["freq_filtering"]
        assert pipe.cfg["reference"]["mode"] == mode


class TestProfileProvenance:
    """The ``profile`` leaf records which recipe the settings came from -- a label only."""

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_profile_leaf_records_the_name(self, builder, name, mode):
        assert builder().to_dict()["profile"] == name

    def test_hand_written_settings_have_no_profile(self):
        assert tm_cca_settings(mode="calibrated_template").to_dict()["profile"] is None
        assert TMCCAPipeline.default_settings().to_dict()["profile"] is None

    def test_profile_never_blocks_construction(self):
        """Unlike reference.mode, an unset profile has no value_options, so it is not required."""
        pipe = TMCCAPipeline(channels=CHANNELS, reference={"mode": "calibrated_template"})
        assert pipe.cfg["profile"] is None

    @pytest.mark.parametrize("builder, name, mode", PROFILES)
    def test_profile_survives_save_and_load(self, builder, name, mode, tmp_path):
        """Provenance rides along in the settings, so it needs no bundle key of its own."""
        pipe = TMCCAPipeline(settings=builder(), channels=CHANNELS)
        path = tmp_path / "pipe.pkl"
        pipe.save(str(path))
        assert TMCCAPipeline.load(str(path)).cfg["profile"] == name

    def test_two_profiles_can_share_one_reference_mode(self):
        """The mode does not identify the recipe -- which is why the name is worth recording."""
        template_ssvep = tm_cca_settings(mode="calibrated_template", profile="cal_ssvep_template",
                                         bands=[(6.0, 40.0)], order=5).to_dict()
        cvep = cvep_settings().to_dict()
        assert template_ssvep["reference"]["mode"] == cvep["reference"]["mode"]
        assert template_ssvep["profile"] != cvep["profile"]
        assert template_ssvep["freq_filtering"] != cvep["freq_filtering"]

    def test_no_decoding_code_reads_the_profile_name(self):
        """A label, never a switch: renaming it must not change what the pipeline computes."""
        source = pathlib.Path(
            template_matching.__file__).read_text(encoding="utf-8")
        body = source.split("class TMCCAPipeline", 1)[1]
        # the only mentions inside the class are the settings schema check and docstrings
        assert 'cfg["profile"]' not in body
        assert "cfg['profile']" not in body


class TestFilterBank:

    def test_multi_band_bank_keeps_per_band_defaults(self):
        """Each sub-band is its own default, so a per-leaf reset cannot collapse the bank."""
        bands = [(6.0, 40.0), (14.0, 40.0), (22.0, 40.0)]
        settings = zerocal_ssvep_settings(bands=bands)
        bank = settings.to_dict()["freq_filtering"]["filterbank"]
        assert [f["cutoff"] for f in bank] == [list(b) for b in bands]

        elements = settings.get_item("freq_filtering", "filterbank").elements
        for element, band in zip(elements, bands):
            assert element.get_item("cutoff").tree["default"] == list(band)

        pipe = TMCCAPipeline(settings=settings, channels=CHANNELS)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS}

    def test_band_and_order_are_keyword_knobs(self):
        settings = cvep_settings(bands=[(2.0, 60.0)], order=9)
        band = settings.to_dict()["freq_filtering"]["filterbank"][0]
        assert (band["cutoff"], band["order"]) == ([2.0, 60.0], 9)

    def test_empty_bank_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            cvep_settings(bands=[])


class TestRequiredMode:

    def test_no_settings_raises_and_names_the_profiles(self):
        with pytest.raises(ValueError) as excinfo:
            TMCCAPipeline()
        message = str(excinfo.value)
        assert "reference.mode is required" in message
        # both designed ways of building settings, and no stale ad-hoc advice
        for name in ("zerocal_ssvep_settings", "cal_ssvep_settings", "cvep_settings",
                     "tm_cca_settings"):
            assert name in message
        assert "reference={" not in message

    def test_default_settings_leaves_the_mode_unset(self):
        settings = TMCCAPipeline.default_settings()
        assert settings.to_dict()["reference"]["mode"] is None
        assert "default" not in settings.get_item("reference", "mode").tree

    def test_setting_the_mode_by_hand_still_works(self):
        """Profiles are the easy path, not the only one."""
        pipe = TMCCAPipeline(channels=CHANNELS,
                             reference={"mode": "calibrated_template"})
        assert pipe.cfg["reference"]["mode"] == "calibrated_template"

    def test_general_builder_accepts_any_mode(self):
        settings = tm_cca_settings(mode="calibrated_template", bands=[(1.0, 45.0)], order=6)
        pipe = TMCCAPipeline(settings=settings, channels=CHANNELS)
        assert pipe.cfg["freq_filtering"]["filterbank"][0]["order"] == 6
        assert pipe.settings.user_overrides() == {"channels": CHANNELS}


class TestBandWeights:
    """``band_weights`` is a plain list of one weight per sub-band, summing to 1."""

    BANDS = [(6.0, 40.0), (14.0, 40.0), (22.0, 40.0)]

    @staticmethod
    def _legacy_fbcca(n_bands):
        """The weighting that used to be hard-coded in ``_fbcca_weights``."""
        k = np.arange(1, n_bands + 1, dtype=float)
        w = k ** -1.25 + 0.25
        return w / w.sum()

    # ---- the two helper functions ----
    @pytest.mark.parametrize("n_bands", [1, 2, 3, 5, 8])
    def test_decaying_power_law_matches_the_old_hard_coded_weights(self, n_bands):
        bank = [(1.0, 2.0)] * n_bands
        np.testing.assert_allclose(decaying_power_law_weights(bank),
                                   self._legacy_fbcca(n_bands))

    @pytest.mark.parametrize("n_bands", [1, 2, 3, 5, 8])
    def test_uniform_splits_evenly(self, n_bands):
        np.testing.assert_allclose(uniform_weights([(1.0, 2.0)] * n_bands),
                                   np.full(n_bands, 1.0 / n_bands))

    @pytest.mark.parametrize("builder", [uniform_weights, decaying_power_law_weights])
    def test_helpers_return_plain_floats_that_sum_to_one(self, builder):
        weights = builder(self.BANDS)
        assert isinstance(weights, list)
        assert all(isinstance(w, float) for w in weights)
        assert sum(weights) == pytest.approx(1.0)

    @pytest.mark.parametrize("builder", [uniform_weights, decaying_power_law_weights])
    def test_single_band_always_gets_weight_one(self, builder):
        assert builder([(6.0, 40.0)]) == pytest.approx([1.0])

    @pytest.mark.parametrize("builder", [uniform_weights, decaying_power_law_weights])
    def test_empty_bank_is_rejected(self, builder):
        with pytest.raises(ValueError, match="at least one sub-band"):
            builder([])

    def test_decay_knobs_change_the_shape(self):
        """A bigger exponent concentrates weight on the first sub-band."""
        steep = decaying_power_law_weights(self.BANDS, exponent=3.0)
        shallow = decaying_power_law_weights(self.BANDS, exponent=0.5)
        assert steep[0] > shallow[0]
        assert sum(steep) == pytest.approx(1.0)
        # exponent 0 with no offset floor is the uniform case
        np.testing.assert_allclose(
            decaying_power_law_weights(self.BANDS, exponent=0.0, offset=0.0),
            uniform_weights(self.BANDS))

    @pytest.mark.parametrize("exponent, offset", [
        (1.25, 0.25),      # the defaults
        (1.25, 0.0),       # offset alone
        (1.25, 2.0),
        (0.0, 0.25),       # exponent alone
        (2.5, 0.25),
        (0.25, 1.25),      # the two swapped, which must NOT give the default result
    ])
    def test_follows_the_formula_for_every_pair_of_knobs(self, exponent, offset):
        """Pin w_k = k**-exponent + offset exactly, so neither knob can be ignored or swapped."""
        n_bands = 4
        k = np.arange(1, n_bands + 1, dtype=float)
        expected = k ** -exponent + offset
        expected = expected / expected.sum()
        np.testing.assert_allclose(
            decaying_power_law_weights([(1.0, 2.0)] * n_bands, exponent=exponent, offset=offset),
            expected)

    def test_offset_alone_flattens_the_weights(self):
        """The offset is a floor: raise it and the sub-bands even out."""
        steep = decaying_power_law_weights(self.BANDS, offset=0.0)
        flat = decaying_power_law_weights(self.BANDS, offset=10.0)
        assert steep[0] > flat[0]
        assert max(flat) - min(flat) < max(steep) - min(steep)
        # with no floor at all the weights are the bare normalised power law
        np.testing.assert_allclose(steep, [1.0, 2.0 ** -1.25, 3.0 ** -1.25]
                                   / np.sum([1.0, 2.0 ** -1.25, 3.0 ** -1.25]))

    def test_weights_decrease_with_a_positive_exponent(self):
        weights = decaying_power_law_weights([(1.0, 2.0)] * 6)
        assert all(a > b for a, b in zip(weights, weights[1:]))

    def test_the_helpers_accept_the_filterbank_config(self):
        """The list from the settings works as well as the raw bands."""
        settings = zerocal_ssvep_settings(bands=self.BANDS)
        bank = settings.to_dict()["freq_filtering"]["filterbank"]
        assert decaying_power_law_weights(bank) == decaying_power_law_weights(self.BANDS)

    # ---- the setting ----
    @pytest.mark.parametrize("builder, rule", [
        (zerocal_ssvep_settings, decaying_power_law_weights),
        (cal_ssvep_settings, decaying_power_law_weights),
        (cvep_settings, uniform_weights),
    ])
    def test_each_profile_pins_the_weighting_its_paradigm_calls_for(self, builder, rule):
        """SSVEP favours the lower sub-bands (a response has a fundamental); c-VEP does not."""
        assert builder(bands=self.BANDS).to_dict()["band_weights"] ==             pytest.approx(rule(self.BANDS))
        assert builder().to_dict()["band_weights"] == [1.0]      # one band: always [1.0]

    @pytest.mark.parametrize("builder", [zerocal_ssvep_settings, cal_ssvep_settings,
                                         cvep_settings])
    def test_profiles_do_not_take_a_weighting_argument(self, builder):
        """The weighting is the profile's opinion, not a knob; tm_cca_settings is the way out."""
        with pytest.raises(TypeError, match="band_weights"):
            builder(bands=self.BANDS, band_weights=[0.5, 0.3, 0.2])

    def test_the_general_builder_still_accepts_explicit_weights(self):
        weights = [0.5, 0.3, 0.2]
        settings = tm_cca_settings(mode="synthetic_harmonics", bands=self.BANDS,
                                   band_weights=weights)
        assert settings.to_dict()["band_weights"] == pytest.approx(weights)
        pipe = TMCCAPipeline(settings=settings, channels=CHANNELS)
        assert pipe.cfg["band_weights"] == pytest.approx(weights)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS}

    def test_weights_can_be_overridden_at_construction(self):
        pipe = TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                             channels=CHANNELS, band_weights=[0.5, 0.3, 0.2])
        assert pipe.cfg["band_weights"] == [0.5, 0.3, 0.2]

    # ---- validation at construction ----
    def test_wrong_length_is_rejected(self):
        with pytest.raises(ValueError, match="one weight per filter-bank sub-band"):
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[0.5, 0.5])

    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[3.0, 2.0, 1.0])

    def test_negative_weights_are_rejected(self):
        """A negative weight subtracts a sub-band's evidence and breaks the 0-1 score scale."""
        with pytest.raises(ValueError, match="zero or positive"):
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[-10.0, 5.5, 5.5])

    @pytest.mark.parametrize("kwargs", [{"exponent": -1.0}, {"offset": -0.5843}])
    def test_helper_refuses_knobs_that_would_produce_negative_weights(self, kwargs):
        """The helper must not emit a list its own checker would refuse."""
        with pytest.raises(ValueError, match="zero or positive"):
            decaying_power_law_weights(self.BANDS, **kwargs)

    def test_helpers_always_produce_acceptable_lists(self):
        """Whatever knobs are allowed, the output must pass validation."""
        for n_bands in (1, 2, 3, 7):
            bands = [(1.0, 2.0)] * n_bands
            for weights in (uniform_weights(bands),
                            decaying_power_law_weights(bands),
                            decaying_power_law_weights(bands, exponent=0.0, offset=0.0),
                            decaying_power_law_weights(bands, exponent=4.0, offset=3.0)):
                TMCCAPipeline(settings=tm_cca_settings(mode="synthetic_harmonics", bands=bands,
                                                       band_weights=weights),
                              channels=CHANNELS)

    def test_non_finite_weights_are_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[float("nan"), 0.5, 0.5])

    def test_settings_editor_rounding_is_tolerated(self):
        """The Qt editor rounds floats to 6 decimals, which must not invalidate a bank.

        Opening a multi-sub-band configuration in the settings editor and reading it straight
        back -- with no edit at all -- returns weights quantised to 6 decimals, so an exact
        sum-to-1 check would make such a configuration unopenable.
        """
        quantised = [round(w, 6) for w in uniform_weights(self.BANDS)]
        assert sum(quantised) != 1.0                    # 0.999999
        pipe = TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                             channels=CHANNELS, band_weights=quantised)
        assert pipe.cfg["band_weights"] == quantised

    def test_a_real_mistake_is_still_rejected(self):
        """The tolerance is generous enough for the editor, not for a wrong list."""
        with pytest.raises(ValueError, match="must sum to 1"):
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[0.333, 0.333, 0.333])

    def test_sum_error_shows_the_real_total_and_the_deviation(self):
        """The message must not round the discrepancy away with %g."""
        with pytest.raises(ValueError) as excinfo:
            TMCCAPipeline(settings=zerocal_ssvep_settings(bands=self.BANDS),
                          channels=CHANNELS, band_weights=[0.5, 0.3, 0.3])
        message = str(excinfo.value)
        assert "off by" in message
        assert "overwrite" in message          # tells the user who keeps the list in sync

    def test_floating_point_rounding_is_tolerated(self):
        """The helpers' own output does not always sum to exactly 1, and must be accepted.

        With 9 sub-bands ``decaying_power_law_weights`` sums to 0.9999999999999999, so an
        exact ``== 1`` check would reject the list this very library produced.
        """
        bands = [(6.0 + 2 * i, 40.0) for i in range(9)]
        weights = decaying_power_law_weights(bands)
        assert sum(weights) != 1.0                      # genuinely off by one ulp
        pipe = TMCCAPipeline(settings=zerocal_ssvep_settings(bands=bands),
                             channels=CHANNELS, band_weights=weights)
        assert pipe.cfg["band_weights"] == weights


class TestBandWeightsReachPredict:
    """The weights must actually reach ``predict``, in the right order.

    Everything above checks lists and settings. These two decode a real synthetic recording,
    so they fail if the multiplication is dropped, if the loop indexes the wrong weight, or if
    the sub-band order and the weight order ever drift apart.
    """

    FPS, FS, N_CMDS, N_CYCLES = 60.0, 250.0, 4, 4
    BANDS = [(6.0, 40.0), (14.0, 40.0), (22.0, 40.0)]

    @pytest.fixture
    def ssvep_recording(self, ssvep_recording_factory):
        with warnings.catch_warnings():          # a few freqs snap to frame-lockable values
            warnings.simplefilter("ignore")
            commands = generate_freq_codebook(self.N_CMDS, freq_range=(8.0, 15.0), t_stim=1.0,
                                              fps_resolution=self.FPS)
        uids = list(commands)
        return ssvep_recording_factory(commands, uids, fps=self.FPS, fs=self.FS,
                                       n_cycles=self.N_CYCLES, resp_amp=0.25, seed=3,
                                       mode="test"), uids

    @staticmethod
    def _predict(recording, bands, weights):
        channels = list(recording.signals["eeg"].channel_set.labels)
        pipe = TMCCAPipeline(
            settings=tm_cca_settings(mode="synthetic_harmonics", bands=bands,
                                     band_weights=weights),
            channels=channels)
        return pipe.predict(recording)

    @pytest.mark.parametrize("kept", [0, 1, 2])
    def test_a_one_hot_weighting_equals_that_sub_band_alone(self, ssvep_recording, kept):
        """Weight 1 on sub-band ``kept`` must reproduce a pipeline built on only that band.

        This pins the index mapping: if ``weights[b]`` were paired with the wrong segment set,
        only ``kept=0`` would survive.
        """
        recording, _ = ssvep_recording
        one_hot = [1.0 if b == kept else 0.0 for b in range(len(self.BANDS))]
        combined = self._predict(recording, self.BANDS, one_hot)
        alone = self._predict(recording, [self.BANDS[kept]], [1.0])
        np.testing.assert_allclose(combined, alone, rtol=1e-9, atol=1e-12)

    def test_different_weights_give_different_scores(self, ssvep_recording):
        """A weighting change must move the scores -- otherwise the setting is decorative."""
        recording, _ = ssvep_recording
        uniform = self._predict(recording, self.BANDS, uniform_weights(self.BANDS))
        decaying = self._predict(recording, self.BANDS,
                                 decaying_power_law_weights(self.BANDS))
        assert not np.allclose(uniform, decaying)

    def test_the_combination_is_the_weighted_sum_of_the_sub_bands(self, ssvep_recording):
        """Pin the arithmetic itself: combined == sum_b w_b * (that sub-band alone)."""
        recording, _ = ssvep_recording
        weights = decaying_power_law_weights(self.BANDS)
        combined = self._predict(recording, self.BANDS, weights)
        expected = sum(w * self._predict(recording, [band], [1.0])
                       for w, band in zip(weights, self.BANDS))
        np.testing.assert_allclose(combined, expected, rtol=1e-9, atol=1e-12)
