"""Created on Monday March 15 19:27:14 2021

This module provides

@author: Eduardo Santamaría-Vázquez
"""

import tensorflow as tf
import os
import warnings


class NoGPU(Exception):
    def __init__(self):
        super().__init__('No GPU available')


class NoCPU(Exception):
    def __init__(self):
        super().__init__('No CPU available')


class DeviceNotAvailable(Exception):
    def __init__(self, device):
        super().__init__('Device %s not available' % device)


def config_tensorflow(gpu_acceleration=True, device=None):
    """
    This function makes a very simple configuration of tensorflow, checking if
    Tensorflow >= 2.0.0 and GPU acceleration is available. For more advanced
    options, use the tensorflow native API.

    Parameters
    ----------
    gpu_acceleration: bool
        Enables GPU acceleration. The function will check if this feature is
        available and select the proper device.
    device: str, optional
        Device that will be selected. Use tf.config.list_logical_devices() to
        list the current available devices.
    """
    try:
        if int(tf.__version__.split('.')[0]) < 2:
            raise ImportError('Tensorflow >= 2.0.0 is required for GPU '
                              'acceleration')
        # Get available devices
        gpus = tf.config.list_logical_devices(device_type='GPU')
        cpus = tf.config.list_logical_devices(device_type='CPU')
        if gpu_acceleration:
            if len(gpus) > 0:
                os.environ["MEDUSA_TF_GPU_ACCELERATION"] = "1"
                if device is None:
                    os.environ["MEDUSA_TF_DEVICE"] = gpus[0].name
                else:
                    check = False
                    for gpu in gpus:
                        if gpu.name == device:
                            check = True
                    if check:
                        os.environ["MEDUSA_TF_DEVICE"] = device
                    else:
                        raise Exception('Device %s not available' % device)
            else:
                raise NoGPU()
        else:
            if len(cpus) > 0:
                os.environ["MEDUSA_TF_GPU_ACCELERATION"] = "0"
                if device is None:
                    os.environ["MEDUSA_TF_DEVICE"] = cpus[0].name
                else:
                    check = False
                    for cpu in cpus:
                        if cpu.name == device:
                            check = True
                    if check:
                        os.environ["MEDUSA_TF_DEVICE"] = device
                    else:
                        raise DeviceNotAvailable(device)
            else:
                raise NoCPU()

            warnings.warn('Some medusa modules take great advantage from GPU '
                          'acceleration to increase the performance of medusa. '
                          'You should consider execute your program in a device'
                          'with available GPU.')

        print('GPU acceleration: %s' % os.environ["MEDUSA_TF_GPU_ACCELERATION"])
        print('Selected device: %s' % os.environ["MEDUSA_TF_DEVICE"])

    except ModuleNotFoundError:
        raise ModuleNotFoundError('Tensorflow is not installed')


def check_tf_config(autoconfig=False):
    """Checks if tensorflow has been configured

     Parameters
    ----------
    autoconfig: bool
        If tensorflow has not been configured and autoconfig is True,
        tensorflow is configured automatically, trying GPU first.
    """
    check = True if os.environ.get("MEDUSA_TF_GPU_ACCELERATION") is not None \
        else False
    if not check and autoconfig:
        __auto_config_tensorflow()
        check = True if os.environ.get("MEDUSA_TF_GPU_ACCELERATION") is not \
                        None else False
    return check


def check_gpu_acceleration():
    """Checks if there is GPU acceleration available"""
    # Check configuration
    if not check_tf_config():
        raise Exception('Tensorflow has not been configured. Call function '
                        'medusa.gpu_acceleration.config_tensorflow')
    # Check GPU
    return True if os.environ.get("MEDUSA_TF_GPU_ACCELERATION") == '1' \
        else False


def get_tf_device_name():
    """Returns the tensorflow device used by medusa"""
    # Check configuration
    if not check_tf_config():
        raise Exception('Tensorflow has not been configured. Call function '
                        'medusa.gpu_acceleration.config_tensorflow')
    return os.environ["MEDUSA_TF_DEVICE"]


def __auto_config_tensorflow():
    """
    This function tries to configure tensorflow automatically. It is meant for
    internal use within medusa-core and avoid errors, but should not be used in
    custom scripts. Use config_tensorflow instead.
    """
    # Configure tensorflow integration automatically if it is not done yet
    try:
        config_tensorflow()
    except NoGPU as e:
        config_tensorflow(gpu_acceleration=False)
    warnings.warn('Tensorflow configured automatically. Function '
                  'tensorflow_integration.config_tensorflow should be'
                  ' called before for custom behaviour.')