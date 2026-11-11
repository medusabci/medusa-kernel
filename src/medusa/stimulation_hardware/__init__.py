"""Models of the hardware that presents stimuli to the participant.

Each device type has its own subpackage:

* :mod:`medusa.stimulation_hardware.display` -- monitors (for example, the raster latency
  of a stimulus).

These modules describe the device: when and how a stimulus reaches the participant. Using
that in a paradigm, for example to correct the onsets of a speller, belongs to that
paradigm's pipeline.
"""

__all__: list[str] = []
