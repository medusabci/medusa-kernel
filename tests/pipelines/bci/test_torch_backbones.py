"""Tests for the shared backbone schema/mapping module.

Two of these come straight from defects an adversarial review of the module found, and both
were reachable only *because* the module exposes every constructor argument:

* ``output_pooling_factor`` accepted 1, which makes EEG-Inception v2 derive its output-block
  count from ``log(samples, 1)`` and die with a bare ``ZeroDivisionError`` after the whole
  feature path has already been computed;
* the temporal-scale leaf name lived in a second lookup table beside ``ARCHITECTURES``, so
  registering a third architecture raised ``KeyError`` while building the schema -- for
  *every* architecture, not just the new one.

The temporal scales are also the one place the two architectures genuinely differ in units:
v1 states them as durations (``scales_ms``, converted at build time), v2 as sample counts
(``temp_scales_samples``, used as given), so ``scales_ms_leaf`` is ``None`` for v2 and the
``scales_ms`` shorthand must not reach it.

Skipped on the no-extras CI job: this module imports the torch backbones.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")

from medusa.core.settings_tree import SettingsTree
from medusa.pipelines.bci import _torch_backbones as tb
from medusa.pipelines.bci.vep_spellers import BWREEGInceptionPipeline

CHANNELS = ["Fz", "Cz", "Pz", "Oz"]


def _classifier_settings(**kwargs):
    s = SettingsTree()
    clf = s.add_group("classifier")
    tb.add_architecture_settings(clf, **kwargs)
    return s


class TestOneRegistry:
    """``ARCHITECTURES`` must be the only place an architecture is registered."""

    def test_registering_an_architecture_needs_no_second_table(self, monkeypatch):
        """A third entry must not break the schema -- least of all for the other two."""
        def add_settings(group, *, scales_ms=(10.0,)):
            group.add_item("scales_ms", value=[float(ms) for ms in scales_ms])

        extra = dict(tb.ARCHITECTURES)
        extra["fake_arch"] = tb._Architecture(add_settings, lambda cfg, **kw: None,
                                              "scales_ms")

        monkeypatch.setattr(tb, "ARCHITECTURES", extra)

        classifier = _classifier_settings(
            arch="eeg_inception_v1", scales_ms=(200.0, 100.0)).to_dict()["classifier"]
        assert set(classifier) == {"arch", *extra}
        assert classifier["fake_arch"]["scales_ms"] == [200.0, 100.0]

    @pytest.mark.parametrize("name", list(tb.ARCHITECTURES))
    def test_every_entry_names_a_leaf_its_own_schema_has(self, name):
        """``scales_ms_leaf``, when set, must exist in the group that architecture builds."""
        spec = tb.ARCHITECTURES[name]
        group = SettingsTree().add_group("g")
        spec.add_settings(group)
        if spec.scales_ms_leaf is not None:
            assert spec.scales_ms_leaf in group.to_dict()

    def test_defaults_rejects_an_unknown_architecture(self):
        with pytest.raises(ValueError, match="defaults has no architecture"):
            _classifier_settings(defaults={"nope": {"dropout_rate": 0.5}})

    def test_build_backbone_rejects_an_unknown_arch(self):
        with pytest.raises(ValueError, match="classifier.arch must be one of"):
            tb.build_backbone({"arch": "nope"}, input_samples=64, n_cha=4, rate=128.0)


class TestOutputPoolingFactorFloor:
    """v2 derives ``log(samples, output_pooling_factor)``, so the factor cannot be 1."""

    @staticmethod
    def _v2_cfg(**overrides):
        cfg = _classifier_settings().to_dict()["classifier"]["eeg_inception_v2"]
        cfg.update(overrides)
        return cfg

    def test_the_leaf_range_starts_at_two(self):
        settings = _classifier_settings()
        settings.set_value("classifier", "eeg_inception_v2", "output_pooling_factor",
                           value=1)
        issues = settings.validate()
        assert any("output_pooling_factor" in path for path, _ in issues)

    def test_building_with_one_fails_with_a_message_not_a_zerodivision(self):
        with pytest.raises(ValueError, match="output_pooling_factor"):
            tb.build_eeg_inception_v2(self._v2_cfg(output_pooling_factor=1),
                                      input_samples=64, n_cha=4, rate=128.0)

    @pytest.mark.parametrize("factor", [2, 3, 4])
    def test_a_legal_factor_still_builds(self, factor):
        backbone = tb.build_eeg_inception_v2(
            self._v2_cfg(output_pooling_factor=factor),
            input_samples=64, n_cha=4, rate=128.0)
        assert backbone.backbone_features > 0

    def test_the_pipeline_refuses_it_at_construction(self):
        """The whole point of a range: it is caught before fit() computes any features."""
        with pytest.raises(ValueError, match="output_pooling_factor"):
            BWREEGInceptionPipeline(
                channels=CHANNELS,
                classifier={"arch": "eeg_inception_v2",
                            "eeg_inception_v2": {"output_pooling_factor": 1}})


class TestScalesToSamples:

    def test_milliseconds_become_samples_at_the_epoch_rate(self):
        assert tb.scales_to_samples([500.0, 250.0, 125.0], 128.0) == (64, 32, 16)

    def test_the_same_duration_survives_a_change_of_rate(self):
        assert tb.scales_to_samples([500.0], 256.0) == (128,)

    def test_a_scale_never_rounds_to_zero_samples(self):
        assert tb.scales_to_samples([0.1], 128.0) == (1,)

    def test_an_empty_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one temporal scale"):
            tb.scales_to_samples([], 128.0)


class TestV2KernelsAreSizedInSamples:
    """v2 states every kernel size directly, so nothing about it depends on the epoch rate."""

    def test_the_leaf_holds_the_sample_counts_it_was_given(self):
        classifier = _classifier_settings(
            defaults={"eeg_inception_v2": {"temp_scales_samples": (50, 25, 15)}}
        ).to_dict()["classifier"]
        assert classifier["eeg_inception_v2"]["temp_scales_samples"] == [50, 25, 15]

    def test_the_milliseconds_shorthand_does_not_reach_it(self):
        """``scales_ms`` is a duration; v2 has no leaf to put one in."""
        classifier = _classifier_settings(
            scales_ms=(500.0, 250.0)).to_dict()["classifier"]
        assert classifier["eeg_inception_v1"]["scales_ms"] == [500.0, 250.0]
        assert classifier["eeg_inception_v2"]["temp_scales_samples"] == [100, 75, 50]

    @pytest.mark.parametrize("rate", [128.0, 200.0, 512.0])
    def test_the_kernels_built_ignore_the_epoch_rate(self, rate):
        cfg = _classifier_settings(
            defaults={"eeg_inception_v2": {"temp_scales_samples": (21, 11)}}
        ).to_dict()["classifier"]["eeg_inception_v2"]
        backbone = tb.build_eeg_inception_v2(cfg, input_samples=64, n_cha=4, rate=rate)
        assert backbone.temp_scales_samples == (21, 11)

    def test_an_empty_scale_list_is_rejected(self):
        cfg = _classifier_settings().to_dict()["classifier"]["eeg_inception_v2"]
        cfg["temp_scales_samples"] = []
        with pytest.raises(ValueError, match="temp_scales_samples"):
            tb.build_eeg_inception_v2(cfg, input_samples=64, n_cha=4, rate=200.0)
