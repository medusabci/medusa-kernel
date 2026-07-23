"""Template-matching speller pipeline based on canonical correlation analysis (CCA).

The template-matching Layer-1 pipeline (:class:`TMCCAPipeline`). It scores each command by
the correlation between a cycle's multichannel EEG segment and that command's *reference*,
accumulating evidence over cycles by coherently averaging the segments
(:func:`~medusa.pipelines.bci.vep_spellers.decoding.tm_command_scores`). A ``reference.mode``
setting picks how the reference is built, named for *what the reference is made of*:

* ``synthetic_harmonics`` -- an analytic sin/cos harmonic bank (calibration-free SSVEP);
* ``calibrated_template`` -- the subject's learned average EEG (calibrated; c-VEP any code +
  calibrated SSVEP);
* ``mixed_harmonics_template`` -- both fused, the extended-CCA a.k.a. eCCA score (calibrated
  SSVEP).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from medusa.core.settings_tree import SettingsTree
from medusa.core.data.recording import Recording
from medusa.core.data.signal import Signal
from medusa.signal.spatial_filtering import car, CCA
from medusa.signal.segmentation import segment_signal_around_events

from medusa.pipelines.base import DecodingPipeline, harmonize_channels
from medusa.pipelines.bci.vep_spellers.data import SpellerData, validate_speller_events
from medusa.pipelines.bci._filtering import (
    add_notch_and_filterbank_settings, apply_notch_and_filterbank)
from medusa.pipelines.bci.vep_spellers.decoding._common import (
    _fbcca_weights, _cycle_arrays)
from medusa.pipelines.bci.vep_spellers.decoding.scores import tm_command_scores

__all__ = ["TMCCAPipeline"]

#: The ``reference.mode`` options, named for what the reference is made of. See the class
#: docstring for the full description of each.
SYNTHETIC_HARMONICS = "synthetic_harmonics"     # analytic sines; calibration-free SSVEP
CALIBRATED_TEMPLATE = "calibrated_template"     # learned EEG template; c-VEP + calibrated SSVEP
MIXED_HARMONICS_TEMPLATE = "mixed_harmonics_template"    # harmonics + template fused (eCCA)

_REFERENCE_MODES = (SYNTHETIC_HARMONICS, CALIBRATED_TEMPLATE, MIXED_HARMONICS_TEMPLATE)
#: Modes that learn a template in :meth:`~TMCCAPipeline.fit` (need ``spell_target``).
_CALIBRATED_MODES = (CALIBRATED_TEMPLATE, MIXED_HARMONICS_TEMPLATE)
#: Modes that build a synthetic harmonic reference (need ``extra['stim_freq']`` per command).
_HARMONIC_MODES = (SYNTHETIC_HARMONICS, MIXED_HARMONICS_TEMPLATE)


def _cca_reference(freq: float, n_samples: int, fs: float, n_harmonics: int) -> NDArray:
    """Sine/cosine harmonic reference for CCA, shape ``(n_samples, 2 * n_harmonics)``.

    For ``k = 0 .. n_harmonics - 1``, column pair ``(2k, 2k+1)`` holds the ``sin`` / ``cos``
    of harmonic ``k + 1`` of ``freq``. Stacking harmonics lets CCA fit the harmonic structure
    of the flicker response.
    """
    t = np.arange(n_samples) / fs
    cols = []
    for h in range(1, n_harmonics + 1):
        cols.append(np.sin(2 * np.pi * h * freq * t))
        cols.append(np.cos(2 * np.pi * h * freq * t))
    return np.column_stack(cols)


def _cca_corr(segment: NDArray, reference: NDArray) -> float:
    """Top CCA canonical correlation ``|r|`` between a segment and a reference.

    ``segment`` is ``(n_samples, n_channels)`` and ``reference`` is ``(n_samples, n_ref)``.
    CCA finds the combination of channels and reference columns that maximises their
    correlation, and this returns the largest canonical correlation (``-inf`` if the fit is
    degenerate). This is the CCA scoring *method* of :class:`TMCCAPipeline`. The family-level
    accumulation lives in :func:`~medusa.pipelines.bci.vep_spellers.decoding.tm_command_scores`.
    """
    cca = CCA()
    cca.fit(segment, reference)
    r = cca.r
    return float(abs(r[0])) if r is not None and np.isfinite(r[0]) else -np.inf


def _best_code_shift(code: NDArray, learned_code: NDArray) -> "tuple[int, float]":
    """Circular lag ``k`` (and its correlation) for which ``roll(learned_code, k) ~= code``.

    In a shift-coded paradigm (c-VEP: every command's code is a circular shift of one
    m-sequence), this finds how far a command's code is rolled from a *learned* code. Its
    EEG template can then be built by rolling the learned template by the same lag.
    """
    a = np.asarray(code, dtype=float)
    a = a - a.mean()
    b = np.asarray(learned_code, dtype=float)
    b = b - b.mean()
    n = len(a)
    rolls = np.stack([np.roll(b, k) for k in range(n)])       # (n, n): roll(b, k) per lag
    num = rolls @ a
    den = np.sqrt((rolls ** 2).sum(axis=1) * (a ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = num / den
    lag = int(np.nanargmax(corr)) if np.isfinite(corr).any() else 0
    return lag, float(corr[lag]) if np.isfinite(corr).any() else -np.inf


#: Circular-shift correlation above which two command codes are treated as the *same*
#: base sequence rolled (m-sequence c-VEP: shifts correlate at 1.0; distinct codes -- Gold
#: c-VEP, SSVEP frequencies -- stay well below). Used to pool shift-family calibration.
_SHIFT_MATCH_TOL = 0.999


def _corr1d(a: NDArray, b: NDArray) -> float:
    """``|Pearson correlation|`` between two 1-D signals (``-inf`` if either is constant)."""
    a = np.asarray(a, dtype=float)
    a = a - a.mean()
    b = np.asarray(b, dtype=float)
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(abs(np.dot(a, b) / den)) if den > 0 else -np.inf


def _pearson_signed(a: NDArray, b: NDArray) -> float:
    """Signed Pearson correlation between two 1-D signals (``0`` if either is constant)."""
    a = np.asarray(a, dtype=float)
    a = a - a.mean()
    b = np.asarray(b, dtype=float)
    b = b - b.mean()
    den = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / den) if den > 0 else 0.0


def _fit_cca(x: NDArray, y: NDArray) -> "CCA | None":
    """Fit ``CCA(x, y)``; return the fitted object, or ``None`` if the fit is degenerate/fails.

    ``CCA.fit`` raises on rank-deficient input (e.g. a constant segment), so this swallows that
    and reports it as ``None`` -- the caller then treats that view as abstaining.
    """
    cca = CCA()
    try:
        cca.fit(x, y)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if cca.wx is None or cca.r is None or not np.all(np.isfinite(cca.wx[:, 0])):
        return None
    return cca


def _ecca_score(segment: NDArray, harmonics: NDArray, template: NDArray) -> float:
    """Extended-CCA (eCCA) score fusing a segment against synthetic harmonics + a template.

    Given the test ``segment`` ``X`` ``(n_samples, n_channels)``, the synthetic harmonic
    reference ``harmonics`` ``Y`` ``(n_samples, 2 * n_harmonics)`` and the calibrated
    multichannel ``template`` ``X_hat`` ``(n_samples, n_channels)``, this computes the classic
    four correlations (Nakanishi et al. 2018) and fuses them as ``sum(sign(r) * r**2)``:

    1. the top canonical correlation of ``X`` vs ``Y`` (test against the synthetic harmonics);
    2. ``corr(X @ w, X_hat @ w)`` with ``w = CCA(X, X_hat)`` (each on the test-template filter);
    3. ``corr(X @ w, X_hat @ w)`` with ``w = CCA(X, Y)`` (each on the test-harmonics filter);
    4. ``corr(X @ w, X_hat @ w)`` with ``w = CCA(X_hat, Y)`` (each on the template-harmonics filter).

    Squaring makes strong correlations dominate; keeping the sign stops an anti-correlated
    component (evidence for the *wrong* command) from being flipped positive by the square.
    A degenerate CCA fit contributes ``0`` (that view simply abstains).
    """
    cca_xy = _fit_cca(segment, harmonics)             # test vs harmonics (shared by 1 and 3)
    r1 = float(cca_xy.r[0]) if cca_xy is not None and np.isfinite(cca_xy.r[0]) else 0.0

    rs = [r1]
    for cca in (_fit_cca(segment, template),          # (2) test-template filter
                cca_xy,                                # (3) test-harmonics filter
                _fit_cca(template, harmonics)):        # (4) template-harmonics filter
        if cca is None:
            rs.append(0.0)
        else:
            w = cca.wx[:, 0]
            rs.append(_pearson_signed(segment @ w, template @ w))
    return float(sum(np.sign(r) * r * r for r in rs))


class TMCCAPipeline(DecodingPipeline):
    """Template-matching speller pipeline based on canonical correlation analysis (CCA).

    Scores each command by the correlation between a cycle's multichannel EEG segment and
    that command's **reference**. Evidence builds up over cycles by coherently averaging the
    segments, so :meth:`predict` returns the cumulative ``(n_cycles, n_commands)`` matrix
    (:func:`~medusa.pipelines.bci.vep_spellers.decoding.tm_command_scores`) that the
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.command_decoder.VEPCommandDecoder`
    selects from. Configuration is levelled: ``freq_filtering`` (notch + filter bank) and
    ``reference``. With a multi-sub-band filter bank, the per-band scores are combined with
    FBCCA weights (:func:`~medusa.pipelines.bci.vep_spellers.decoding._common._fbcca_weights`).

    The ``reference.mode`` setting is **required** (no default). Each mode is named for what
    the reference is made of:

    * ``"synthetic_harmonics"`` -- **calibration-free SSVEP**. The reference is the command's
      sin/cos harmonic bank at ``extra['stim_freq']`` (from
      :func:`~medusa.pipelines.bci.vep_spellers.encoding.generate_freq_codebook`). The score is the
      top canonical correlation. There is no training: construct the pipeline and call
      :meth:`predict` directly.
    * ``"calibrated_template"`` -- **calibrated**. :meth:`fit` learns, per shift-family and per
      sub-band, a coherent-average EEG template and a spatial filter (it needs ``spell_target``).
      Scoring projects both the template and the test segment to 1-D with that filter and
      takes their ``|Pearson|``. A full multichannel template would let CCA align any command,
      so the 1-D projection is what tells commands apart. Shift-coded commands (c-VEP) are
      built from one pooled base template by rolling it by the command's code lag. Distinct
      codes (Gold c-VEP, SSVEP) each keep their own template. One mode serves both
      SSVEP-with-calibration and c-VEP.
    * ``"mixed_harmonics_template"`` -- **calibrated SSVEP**, the extended-CCA (**eCCA**) score.
      It *mixes* the two references above: for each command it fuses the canonical correlation
      against the synthetic harmonics with correlations against the learned template, through
      several CCA spatial filters, as ``sum(sign(r) * r**2)`` (see :func:`_ecca_score`). This
      pools the calibration-free harmonic view and the subject-specific template view, and is
      the accuracy sweet spot for SSVEP. Needs both ``spell_target`` (to learn the template)
      **and** ``extra['stim_freq']`` (to build the harmonics).

    A *template-matching* sibling of
    :class:`~medusa.pipelines.bci.vep_spellers.decoding.bwr_lda.BWRLDAPipeline`. Like it, it
    is a **direct** :class:`~medusa.pipelines.base.DecodingPipeline` (no shared template-matching
    base): its feature path (one multichannel segment per *cycle*, no per-frame epoching) has
    nothing in common with BWR's. TRCA (a different spatial-filter objective) will be a separate
    pipeline.
    """

    fs = None            # sampling rate adopted at fit/predict
    # calibrated modes: per sub-band {base: (spatial_filter, 1-D template, code, multichannel template)}
    templates = None

    # ---- configuration schema (SettingsTree) ----
    @classmethod
    def default_settings(cls) -> SettingsTree:
        """A GUI-editable, levelled schema; ``reference.mode`` is required (no default)."""
        s = SettingsTree()
        s.add_item("channels", value=[], info="Channels to decode (required)")
        s.add_item("signal_key", value="eeg", info="Recording stream key to decode")
        add_notch_and_filterbank_settings(s)
        s.add_item("car", value=False,
                   info="Common-average reference before scoring (CCA already "
                        "spatially filters; CAR makes the montage rank-deficient)")
        ref = s.add_group("reference", info="Template-matching reference")
        ref.add_item("mode", value=None, value_options=list(_REFERENCE_MODES),
                     info="Reference mode (REQUIRED): 'synthetic_harmonics' (calibration-free "
                          "SSVEP), 'calibrated_template' (learned template; c-VEP + SSVEP, needs "
                          "fit), or 'mixed_harmonics_template' (eCCA; calibrated SSVEP, needs fit)")
        ref.add_item("n_harmonics", value=3, value_range=[1, None],
                     info="Sine/cosine harmonics per reference (harmonic modes)")
        return s

    # ---- validation ----
    def check_consistency(self, recording: Recording) -> None:
        """Check the recording has the configured signal and channels, a matching ``fs``,
        valid speller events, and (for the harmonic modes) a frequency per command;
        raise ``ValueError`` if not."""
        cfg = self.cfg
        sig = recording.signals.get(cfg["signal_key"])
        if sig is None:
            raise ValueError(f"recording has no {cfg['signal_key']!r} signal.")
        if not cfg["channels"]:
            raise ValueError("no channels configured; set the 'channels' setting.")
        if self.fs is None:
            self.fs = sig.fs
        elif sig.fs != self.fs:
            raise ValueError(f"fs mismatch: pipeline={self.fs}, recording={sig.fs}.")
        missing = [c for c in cfg["channels"] if c not in sig.channel_set.labels]
        if missing:
            raise ValueError(f"recording is missing channels {missing}.")
        sd = SpellerData.from_recording(recording)
        mode = cfg["reference"]["mode"]
        if mode not in _REFERENCE_MODES:
            raise ValueError(
                f"reference.mode must be one of {list(_REFERENCE_MODES)}, got {mode!r}.")
        if mode in _HARMONIC_MODES:
            missing_freq = [uid for uid, cmd in sd.commands_info.items()
                            if cmd.extra.get("stim_freq") is None]
            if missing_freq:
                raise ValueError(
                    f"reference.mode={mode!r} needs a stimulation frequency per command; "
                    f"commands {missing_freq} lack extra['stim_freq'] (build the codebook "
                    f"with generate_freq_codebook).")
        validate_speller_events(recording.events)

    # ---- feature path (one segment set per filter-bank sub-band) ----
    def _cycle_segments_per_band(self, signal: Signal, cycle_onsets: NDArray,
                                 n_frames: int, fps: float,
                                 cfg: dict) -> "list[NDArray]":
        """Per sub-band, one multichannel segment per cycle: list of ``(n_cycles, n_s, n_ch)``."""
        x = harmonize_channels(signal, cfg["channels"])
        raw = car(x.signal) if cfg["car"] else x.signal
        window = (0.0, 1000.0 * n_frames / fps)     # the full stimulation cycle, in ms
        return [segment_signal_around_events(x.times, xf, cycle_onsets, x.fs, window)
                for xf in apply_notch_and_filterbank(raw, x.fs, cfg["notch_filtering"],
                                                     cfg["freq_filtering"]["filterbank"])]

    # ---- offline ----
    def fit(self, recordings=()) -> "TMCCAPipeline":
        """Calibrate the pipeline and return ``self``.

        For ``reference.mode='synthetic_harmonics'`` there is nothing to learn: it only
        validates the recordings and adopts ``fs``. The calibrated modes
        (``'calibrated_template'``, ``'mixed_harmonics_template'``) call
        :meth:`_learn_templates`.
        """
        self._check_settings()
        if self.cfg["reference"]["mode"] == SYNTHETIC_HARMONICS:
            for rec in recordings:
                self.check_consistency(rec)
        else:
            self._learn_templates(recordings)
        self._fitted = True
        return self

    def predict(self, recording: Recording) -> NDArray:
        """Cumulative ``(n_cycles, n_commands)`` correlations (FBCCA-combined over sub-bands)."""
        cfg = self.cfg
        if cfg["reference"]["mode"] in _CALIBRATED_MODES and not self._fitted:
            raise RuntimeError(
                f"reference.mode={cfg['reference']['mode']!r} pipeline is not fitted; "
                f"call fit() first.")
        self.check_consistency(recording)          # adopts fs
        sd = SpellerData.from_recording(recording)
        onsets, trial, cycle, _ = _cycle_arrays(recording.events)
        n_frames = sd.codes.shape[2]
        band_segs = self._cycle_segments_per_band(
            recording.signals[cfg["signal_key"]],
            onsets, n_frames, sd.fps_resolution, cfg)
        weights = _fbcca_weights(len(band_segs))
        combined = None                              # weighted sum of per-sub-band scores
        for b, segs in enumerate(band_segs):
            score_fn = self._build_score_fn(sd, segs.shape[1], n_frames, b)
            m = np.nan_to_num(tm_command_scores(segs, score_fn, trial, cycle), neginf=0.0)
            combined = weights[b] * m if combined is None else combined + weights[b] * m
        return combined

    # ------------------------------------------------------------------ #
    # Reference modes -- each a self-contained scoring strategy. _build_score_fn
    # dispatches per sub-band; each mode owns its reference construction + scoring.
    # ------------------------------------------------------------------ #
    def _build_score_fn(self, sd: SpellerData, n_samples: int, n_frames: int, band: int):
        """A ``score_fn(avg_segment) -> (n_commands,)`` for the mode, on sub-band ``band``."""
        mode = self.cfg["reference"]["mode"]
        if mode == SYNTHETIC_HARMONICS:
            return self._harmonic_score_fn(sd, n_samples)
        if mode == MIXED_HARMONICS_TEMPLATE:
            return self._mixed_score_fn(sd, n_samples, n_frames, band)
        return self._template_score_fn(sd, n_samples, n_frames, band)

    def _harmonic_score_fn(self, sd: SpellerData, n_samples: int):
        """Calibration-free SSVEP: CCA of the segment against each command's sin/cos harmonics.

        The harmonic reference is low-dimensional (``2 * n_harmonics`` columns), so CCA's
        spatial fit is well-posed. The score is the top canonical correlation. It is
        band-independent: every sub-band scores against the same references.
        """
        n_harmonics = int(self.cfg["reference"]["n_harmonics"])
        refs = [_cca_reference(float(cmd.extra["stim_freq"]),
                               n_samples, self.fs, n_harmonics)
                for cmd in sd.commands_info.values()]
        return lambda avg: np.array([_cca_corr(avg, ref) for ref in refs])

    def _template_score_fn(self, sd: SpellerData, n_samples: int, n_frames: int, band: int):
        """Calibrated: a learned spatial filter and 1-D correlation against shifted templates.

        Uses sub-band ``band``'s learned templates. A full multichannel template would let CCA
        align any command, so each learned base owns a spatial filter ``w`` that projects both
        its template and the test segment to 1-D. The score is the ``|Pearson|`` between them.
        """
        templates = self.templates[band]
        scorers = [
            self._template_for(
                templates, np.asarray(sd.codes[i, 0]),
                n_samples, n_frames
            )
            for i in range(len(sd.command_uids))
        ]
        return lambda avg: np.array([_corr1d(avg @ w, t1d) for w, t1d in scorers])

    def _mixed_score_fn(self, sd: SpellerData, n_samples: int, n_frames: int, band: int):
        """Calibrated SSVEP (eCCA): fuse the synthetic harmonics and the learned template.

        For each command it builds the synthetic harmonic reference ``Y`` at
        ``extra['stim_freq']`` and picks sub-band ``band``'s learned multichannel template
        ``X_hat`` (rolled to the command's code lag). The score is :func:`_ecca_score`, which
        fuses the test-vs-harmonics canonical correlation with correlations against the template
        via ``sum(sign(r) * r**2)``.
        """
        n_harmonics = int(self.cfg["reference"]["n_harmonics"])
        templates = self.templates[band]
        per_cmd = []                                 # (harmonic reference Y, template X_hat)
        for i, cmd in enumerate(sd.commands_info.values()):
            harmonics = _cca_reference(
                float(cmd.extra["stim_freq"]), n_samples, self.fs, n_harmonics)
            entry, lag = self._matched_base(templates, np.asarray(sd.codes[i, 0]))
            lag_samples = int(round(lag * n_samples / n_frames))
            template = np.roll(entry[3], lag_samples, axis=0)      # multichannel template
            per_cmd.append((harmonics, template))
        return lambda avg: np.array([_ecca_score(avg, h, t) for h, t in per_cmd])

    @staticmethod
    def _matched_base(templates: dict, code: NDArray) -> "tuple[tuple, int]":
        """The learned-base ``(entry, lag)`` whose code best matches ``code`` (identical -> lag 0).

        ``entry`` is the base's ``(w, 1-D template, code, multichannel template)`` tuple. Shared
        by :meth:`_template_for` and :meth:`_mixed_score_fn` so both resolve a command to the
        same learned base (a c-VEP command decoded from one pooled, rolled template).
        """
        best = None
        for _base, entry in templates.items():
            lag, corr = _best_code_shift(code, np.asarray(entry[2]))
            if best is None or corr > best[0]:
                best = (corr, entry, lag)
        _, entry, lag = best
        return entry, lag

    @classmethod
    def _template_for(cls, templates: dict, code: NDArray, n_samples: int,
                      n_frames: int) -> "tuple[NDArray, NDArray]":
        """``(spatial filter w, shifted 1-D template)`` for a command's ``code``, from ``templates``.

        Picks the learned base whose code best matches ``code`` (an identical code means lag 0)
        and rolls its 1-D template by the code lag, mapped to samples
        (``lag * n_samples / n_frames``). So a c-VEP command decoded from one pooled template
        lands on its own response.
        """
        entry, lag = cls._matched_base(templates, code)
        w, t1d = entry[0], entry[1]
        lag_samples = int(round(lag * n_samples / n_frames))
        return w, np.roll(t1d, lag_samples)

    def _learn_templates(self, recordings) -> None:
        """Calibrate the template modes: templates per sub-band and per shift-family.

        Used by both ``calibrated_template`` and ``mixed_harmonics_template``. For each
        filter-bank sub-band, attended commands whose codes are circular shifts (an m-sequence
        family) are pooled: their epochs are rolled to a common base and averaged into one
        template and spatial filter. Distinct codes each form their own base. Every sub-band
        learns its own template set (see :meth:`_pool_templates`).
        """
        cfg = self.cfg
        band_epochs, codes, n_samples, n_frames = None, {}, None, None
        for rec in recordings:
            self.check_consistency(rec)
            sd = SpellerData.from_recording(rec)
            if sd.spell_target is None:
                raise ValueError(
                    f"reference.mode={cfg['reference']['mode']!r} needs spell_target in each "
                    f"calibration recording.")
            onsets, trial, _, _ = _cycle_arrays(rec.events)
            band_segs = self._cycle_segments_per_band(
                rec.signals[cfg["signal_key"]],
                onsets, sd.codes.shape[2],
                sd.fps_resolution, cfg)
            if band_epochs is None:
                band_epochs = [{} for _ in band_segs]
            n_samples, n_frames = band_segs[0].shape[1], sd.codes.shape[2]
            row = {u: i for i, u in enumerate(sd.command_uids)}
            for b, segs in enumerate(band_segs):
                for i, t in enumerate(trial):
                    uid = str(sd.spell_target[int(t)])
                    band_epochs[b].setdefault(uid, []).append(segs[i])
                    codes[uid] = np.asarray(sd.codes[row[uid], 0])
        if band_epochs is None or not codes:
            raise ValueError("no calibration segments found to learn templates.")
        self.templates = [self._pool_templates(eps, codes, n_samples, n_frames)
                          for eps in band_epochs]

    @staticmethod
    def _pool_templates(epochs: dict, codes: dict, n_samples: int, n_frames: int) -> dict:
        """Pool shift-family epochs into ``{base: (w, 1-D template, code, template)}`` per sub-band.

        Commands whose codes are circular shifts (corr >= ``_SHIFT_MATCH_TOL``) are aligned to
        a common base and averaged into one template. A spatial filter ``w`` (the first CCA
        component between the single epochs and the pooled template) projects each base to its
        1-D template. The multichannel template is kept too (the 4th tuple element), for the
        ``mixed_harmonics_template`` (eCCA) mode which needs it to build its own CCA filters.
        """
        bases = []                                       # [(base_code, [aligned epochs]), ...]
        for uid, eps in epochs.items():
            for base_code, pool in bases:
                lag, corr = _best_code_shift(codes[uid], base_code)
                if corr >= _SHIFT_MATCH_TOL:
                    shift = int(round(lag * n_samples / n_frames))
                    pool.extend(np.roll(ep, -shift, axis=0) for ep in eps)
                    break
            else:
                bases.append((codes[uid], list(eps)))
        templates = {}
        for k, (base_code, pool) in enumerate(bases):
            eps = np.stack(pool)                         # (n_epochs, n_samples, n_channels)
            template = eps.mean(axis=0)                  # (n_samples, n_channels)
            cca = CCA()                                  # spatial filter: epochs <-> template
            cca.fit(eps.reshape(-1, eps.shape[2]), np.tile(template, (len(eps), 1)))
            templates[str(k)] = (cca.wx[:, 0], template @ cca.wx[:, 0], base_code, template)
        return templates

    # ---- persistence (settings + adopted fs + learned templates) ----
    def to_pickleable_obj(self) -> dict:
        """Bundle the settings, the fitted flag, ``fs``, and the learned templates for saving."""
        return {"settings": self.settings.to_dict(), "fitted": self._fitted, "fs": self.fs,
                "templates": self.templates}

    @classmethod
    def from_pickleable_obj(cls, obj: dict) -> "TMCCAPipeline":
        """Rebuild the pipeline from a bundle made by :meth:`to_pickleable_obj`."""
        self = cls(settings=obj["settings"])
        self.fs, self._fitted = obj["fs"], obj["fitted"]
        self.templates = obj.get("templates")
        return self
