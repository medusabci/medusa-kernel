"""Tests for the ``BWREEGInceptionPipeline`` stimulation profiles.

The same three failure modes the ``TMCCAPipeline`` profiles are checked against
(``test_tmcca_profiles.py``), minus the required-mode one, since this pipeline does have a
working bare schema:

1. **it does not build** -- a profile is a settings payload written by hand, so a schema
   rename silently rots it. ``update_from_dict`` only *warns* on an unknown key, so the
   drift test turns warnings into errors;
2. **its values are not its defaults** -- if a profile sets values on top of the stock tree
   instead of building its own defaults, ``reset()`` wipes it and ``user_overrides()``
   reports the profile as if the user had typed it;
3. **the name becomes a switch** -- ``profile`` is provenance only, so no decoding code may
   read it.

Nothing here pins the *values* a profile picks: those are tuned per paradigm and are meant
to move. What is pinned is that a profile is a real settings baseline and that the pipeline
behaves the same whatever the name says.

Skipped on the no-extras CI job: this pipeline is torch-gated.
"""
import pathlib
import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.pipelines.bci._torch_backbones import ARCHITECTURES
from medusa.pipelines.bci.vep_spellers.decoding import bwr_eeg_inception
from medusa.pipelines.bci.vep_spellers import (
    BWREEGInceptionPipeline,
    bwr_eeg_inception_settings,
    mseq_cvep_settings,
    burst_cvep_settings,
)

#: Every shipped profile: its builder and the name it records.
PROFILES = [
    (mseq_cvep_settings, "mseq_cvep"),
    (burst_cvep_settings, "burst_cvep"),
]

CHANNELS = ["Fz", "Cz", "Pz", "Oz"]


class TestProfileSettings:

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_builds_a_valid_tree_without_warnings(self, builder, name):
        """A profile builds cleanly (catches schema drift)."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")      # an unknown/renamed key must fail loudly
            settings = builder()
        assert settings.validate() == []

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_returns_a_fresh_tree_each_call(self, builder, name):
        """Profiles are functions, not shared constants: the ctor stores trees by reference."""
        first, second = builder(), builder()
        assert first is not second
        first.set_value("car", value=False)
        assert second.to_dict()["car"] is True

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_constructs_a_pipeline(self, builder, name):
        pipe = BWREEGInceptionPipeline(settings=builder(), channels=CHANNELS)
        assert pipe.cfg["channels"] == CHANNELS
        assert pipe.cfg["profile"] == name

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_profile_values_are_the_defaults(self, builder, name):
        """The profile is the baseline: only the user's own edits are overrides."""
        pipe = BWREEGInceptionPipeline(settings=builder(), channels=CHANNELS, car=False)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS, "car": False}

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_reset_restores_the_profile_not_the_stock_schema(self, builder, name):
        """``reset()`` (and the GUI's 'Reset all') must return to the profile."""
        profiled = builder().to_dict()
        pipe = BWREEGInceptionPipeline(settings=builder(), channels=CHANNELS)
        pipe.settings.set_value("car", value=False)
        pipe.settings.set_value("segmentation", "w_segment_t", value=[0.0, 123.0])
        pipe.settings.get_item("freq_filtering", "filterbank").set_elements(
            [{"filt_type": "iir", "band_type": "bandpass", "cutoff": [3.0, 30.0], "order": 2}])
        pipe.settings.reset()

        assert pipe.cfg["car"] == profiled["car"]
        assert pipe.cfg["segmentation"] == profiled["segmentation"]
        assert pipe.cfg["freq_filtering"] == profiled["freq_filtering"]

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_the_filter_bank_holds_exactly_one_band(self, builder, name):
        """A conv backbone cannot fuse sub-bands, so a profile must never build a bank."""
        assert len(builder().to_dict()["freq_filtering"]["filterbank"]) == 1

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_band_window_and_arch_are_keyword_knobs(self, builder, name):
        settings = builder(band=(2.0, 45.0), order=9, w_segment_t=(0.0, 300.0),
                           arch="eeg_inception_v2").to_dict()
        band = settings["freq_filtering"]["filterbank"][0]
        assert (band["cutoff"], band["order"]) == ([2.0, 45.0], 9)
        assert settings["segmentation"]["w_segment_t"] == [0.0, 300.0]
        assert settings["classifier"]["arch"] == "eeg_inception_v2"

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_knobs_passed_to_a_profile_are_defaults_too(self, builder, name):
        """A knob is part of the recipe, not a user edit, so it must survive ``reset()``."""
        pipe = BWREEGInceptionPipeline(settings=builder(w_segment_t=(0.0, 300.0)),
                                       channels=CHANNELS)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS}
        pipe.settings.reset()
        assert pipe.cfg["segmentation"]["w_segment_t"] == [0.0, 300.0]


class TestProfileProvenance:
    """The ``profile`` leaf records which recipe the settings came from -- a label only."""

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_profile_leaf_records_the_name(self, builder, name):
        assert builder().to_dict()["profile"] == name

    def test_hand_written_settings_have_no_profile(self):
        assert bwr_eeg_inception_settings().to_dict()["profile"] is None
        assert BWREEGInceptionPipeline.default_settings().to_dict()["profile"] is None

    def test_profile_never_blocks_construction(self):
        """The leaf has no value_options, so an unset profile is not a required setting."""
        pipe = BWREEGInceptionPipeline(channels=CHANNELS)
        assert pipe.cfg["profile"] is None

    @pytest.mark.parametrize("builder, name", PROFILES)
    def test_profile_survives_save_and_load(self, builder, name, tmp_path):
        """Provenance rides along in the settings, so it needs no bundle key of its own."""
        pipe = BWREEGInceptionPipeline(settings=builder(), channels=CHANNELS)
        path = tmp_path / "pipe.pkl"
        pipe.save(str(path))
        assert BWREEGInceptionPipeline.load(str(path)).cfg["profile"] == name

    def test_two_profiles_can_share_one_set_of_values(self):
        """The values do not identify the recipe -- which is why the name is worth recording."""
        a = bwr_eeg_inception_settings(profile="one").to_dict()
        b = bwr_eeg_inception_settings(profile="two").to_dict()
        assert a["profile"] != b["profile"]
        assert {k: v for k, v in a.items() if k != "profile"} == \
               {k: v for k, v in b.items() if k != "profile"}

    def test_no_decoding_code_reads_the_profile_name(self):
        """A label, never a switch: renaming it must not change what the pipeline computes."""
        source = pathlib.Path(
            bwr_eeg_inception.__file__).read_text(encoding="utf-8")
        body = source.split("class BWREEGInceptionPipeline", 1)[1]
        assert 'cfg["profile"]' not in body
        assert "cfg['profile']" not in body


class TestLazyTorchGating:
    """The profiles live in the torch-gated module, so they must be resolved lazily too."""

    def test_the_package_exports_every_profile(self):
        from medusa.pipelines.bci.vep_spellers import decoding
        for name in ("bwr_eeg_inception_settings", "mseq_cvep_settings",
                     "burst_cvep_settings"):
            assert name in decoding.__all__
            assert getattr(decoding, name) is getattr(bwr_eeg_inception, name)

    def test_an_unknown_name_still_raises_attribute_error(self):
        from medusa.pipelines.bci.vep_spellers import decoding
        with pytest.raises(AttributeError):
            decoding.no_such_profile_settings


class TestArchitectureGroups:
    """Each architecture owns a group of its real hyper-parameters, named after itself.

    The flattened schema this replaced could only offer the three knobs v1 and v2 happen to
    share, and it wrote one ``filters_per_branch`` onto both of v2's independent filter
    counts. These pin the shape that fixed it.
    """

    def test_one_group_per_architecture(self):
        classifier = bwr_eeg_inception_settings().to_dict()["classifier"]
        assert set(ARCHITECTURES) <= set(classifier)

    @pytest.mark.parametrize("arch", list(ARCHITECTURES))
    def test_the_group_key_is_the_arch_value(self, arch):
        """build_backbone indexes the classifier group by the arch string, with no table."""
        classifier = bwr_eeg_inception_settings(arch=arch).to_dict()["classifier"]
        assert classifier["arch"] == arch
        assert isinstance(classifier[arch], dict) and classifier[arch]

    def test_v2_filter_counts_are_independent(self):
        pipe = BWREEGInceptionPipeline(
            channels=CHANNELS,
            classifier={"arch": "eeg_inception_v2",
                        "eeg_inception_v2": {"dil_filt_per_branch": 16}})
        v2 = pipe.cfg["classifier"]["eeg_inception_v2"]
        assert (v2["temp_filt_per_branch"], v2["dil_filt_per_branch"]) == (8, 16)

    def test_dil_branch_specs_is_an_editable_group_list(self):
        v2 = bwr_eeg_inception_settings().to_dict()["classifier"]["eeg_inception_v2"]
        assert [(b["kernel"], b["dilation"]) for b in v2["dil_branch_specs"]] == \
               [(5, 1), (5, 5), (5, 10), (5, 15)]

    def test_every_branch_of_the_group_list_is_its_own_default(self):
        """Otherwise reset() would collapse the four branches onto the template's values."""
        settings = bwr_eeg_inception_settings()
        pipe = BWREEGInceptionPipeline(settings=settings, channels=CHANNELS)
        pipe.settings.reset()
        v2 = pipe.cfg["classifier"]["eeg_inception_v2"]
        assert [b["dilation"] for b in v2["dil_branch_specs"]] == [1, 5, 10, 15]

    def test_scales_ms_reaches_every_architecture(self):
        """One millisecond opinion, written into whichever leaf each architecture calls it."""
        classifier = bwr_eeg_inception_settings(
            scales_ms=(200.0, 100.0)).to_dict()["classifier"]
        assert classifier["eeg_inception_v1"]["scales_ms"] == [200.0, 100.0]
        assert classifier["eeg_inception_v2"]["temp_scales_ms"] == [200.0, 100.0]

    def test_an_unknown_arch_is_rejected(self):
        with pytest.raises(ValueError, match="arch must be one of"):
            bwr_eeg_inception_settings(arch="eeg_inception_v3")

    @pytest.mark.parametrize("arch", list(ARCHITECTURES))
    def test_the_unused_architecture_is_still_a_default(self, arch):
        """The inactive group is inert, not an edit: it must not show in user_overrides()."""
        pipe = BWREEGInceptionPipeline(settings=bwr_eeg_inception_settings(arch=arch),
                                       channels=CHANNELS)
        assert pipe.settings.user_overrides() == {"channels": CHANNELS}
