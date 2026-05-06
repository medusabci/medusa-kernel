"""Backward-compatibility shim for medusa.components.

All classes have moved to their canonical locations in core/.
This module re-exports them so that existing code continues to work
during the K1 migration. It will be removed in a future release.
"""
from medusa.core.serialization import SerializableComponent, PickleableComponent
from medusa.core.data.recording import (
    Recording, ConsistencyChecker, BiosignalData, CustomBiosignalData,
)
from medusa.core.data.experiment import ExperimentData, CustomExperimentData
from medusa.core.pipeline import (
    ProcessingMethod, ProcessingFuncWrapper, ProcessingClassWrapper,
    PipelineConnector, Pipeline, Algorithm,
)
from medusa.ml.dataset import Dataset
from medusa.core.utils import ThreadWithReturnValue

__all__ = [
    'SerializableComponent', 'PickleableComponent',
    'Recording', 'ConsistencyChecker', 'BiosignalData', 'CustomBiosignalData',
    'ExperimentData', 'CustomExperimentData',
    'ProcessingMethod', 'ProcessingFuncWrapper', 'ProcessingClassWrapper',
    'PipelineConnector', 'Pipeline', 'Algorithm',
    'Dataset',
    'ThreadWithReturnValue',
]
