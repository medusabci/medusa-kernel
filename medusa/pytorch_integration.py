import os
import warnings

try:
    import torch
    os.environ["MEDUSA_TORCH_INTEGRATION"] = "1"
except ImportError:
    os.environ["MEDUSA_TORCH_INTEGRATION"] = "0"


class TorchExtrasNotInstalled(Exception):
    """
    Exception raised when a required PyTorch package or dependency is not
    installed.

    Parameters
    ----------
    msg : str, optional
        Custom error message. If not provided, a default message is used.

    Notes
    -----
    - This exception is typically raised when an external package that
        integrates with PyTorch is missing.
    - Users should check https://pytorch.org/ for installation instructions.

    Examples
    --------
    >>> raise TorchExtrasNotInstalled()
    Traceback (most recent call last):
        ...
    TorchExtrasNotInstalled: This functionality requires PyTorch package. Check
    https://pytorch.org/ for installation instructions.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = (
                "This functionality requires the PyTorch package. "
                "Visit https://pytorch.org/ for installation instructions."
            )
        super().__init__(msg)


class TorchNotConfiguredError(Exception):
    """
    Exception raised when PyTorch has not been properly configured.

    This error occurs when an operation requiring PyTorch is attempted
    without first configuring the PyTorch environment.

    Parameters
    ----------
    msg : str, optional
        Custom error message. If not provided, a default message is used.

    Notes
    -----
    - This exception is typically raised when `config_pytorch()` has not been
        called.
    - Users should ensure that PyTorch integration is properly initialized
        before  executing GPU-dependent or Medusa-related operations.

    Examples
    --------
    >>> raise TorchNotConfiguredError()
    Traceback (most recent call last):
        ...
    TorchNotConfiguredError: PyTorch has not been configured. Call
    `config_pytorch()` before using PyTorch-related features.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = (
                "PyTorch has not been configured. "
                "Call `config_pytorch()` before using PyTorch-related features."
            )
        super().__init__(msg)


class NoGPUError(Exception):
    """
    Exception raised when no GPU compatible with Pytorch is available for
    computation.

    This error occurs when a GPU is required but not detected in the system.

    Parameters
    ----------
    msg : str, optional
        Custom error message. If not provided, a default message is used.

    Notes
    -----
    - This exception is typically raised when attempting to use CUDA, but no
        GPU is available.
    - Users should verify their hardware and ensure that CUDA is properly
        installed.
    - To check GPU availability in PyTorch, use `torch.cuda.is_available()`.

    Examples
    --------
    >>> raise NoGPUError()
    Traceback (most recent call last):
        ...
    NoGPUError: No GPU available. Ensure your system has a compatible GPU
        and that CUDA is installed.
    """
    def __init__(self, msg=None):
        if msg is None:
            msg = (
                "No GPU available. Ensure your system has a compatible GPU "
                "and that CUDA is installed."
            )
        super().__init__(msg)



class DeviceNotAvailableError(Exception):
    """
    Exception raised when the requested computing device is not available.

    This error occurs when attempting to use a specific device (e.g., 'cuda',
    'cuda:0', 'mps') that is either not detected or unsupported on the system.

    Parameters
    ----------
    device : str, optional
        The name of the device that is unavailable (e.g., `'cuda'`, `'cuda:0'`,
        `'mps'`).
    msg : str, optional
        Custom error message. If not provided, a default message is generated.

    Notes
    -----
    - If `device` is provided, the error message includes the unavailable
        device name.
    - Users should verify device availability using PyTorch methods like
        `torch.cuda.is_available()`.

    Examples
    --------
    >>> raise DeviceNotAvailableError(device="cuda:0")
    Traceback (most recent call last):
        ...
    DeviceNotAvailableError: Device 'cuda:0' is not available. Check
        system configuration.

    >>> raise DeviceNotAvailableError()
    Traceback (most recent call last):
        ...
    DeviceNotAvailableError: Device not available. Ensure the required device
        is connected and supported.
    """
    def __init__(self, device=None, msg=None):
        if msg is None:
            if device is None:
                msg = ("Device not available. Ensure the required device is "
                       "connected and is supported by PyTorch.")
            else:
                msg = (f"Device '{device}' is not available. Check system "
                       f"configuration.")
        super().__init__(msg)



def config_pytorch(device_name=None):
    """
    Configures PyTorch, checking for GPU acceleration and setting the
    appropriate device.

    This function automatically selects a device based on availability or uses
    the user-specified device. It also sets environment variables to indicate
    the selected device and whether GPU acceleration is enabled.

    Parameters
    ----------
    device_name : str, optional
        The specific device to use (e.g., `'cuda'`, `'cuda:0'`, or `'cpu'`).
        If `None`, the function will automatically choose `'cuda:0'` if a GPU
        is available; otherwise, it defaults to `'cpu'`.

    Returns
    -------
    torch.device
        The configured PyTorch device.

    Raises
    ------
    ImportError
        If PyTorch is not properly integrated (`MEDUSA_TORCH_INTEGRATION`
        environment variable is missing).
    TorchExtrasNotInstalled
        If additional PyTorch-related dependencies are not installed.

    Notes
    -----
    - If `device_name` is not specified, the function defaults to `'cuda:0'`
        if CUDA is available, otherwise `'cpu'`.
    - The function sets two environment variables:
        - `"MEDUSA_TORCH_DEVICE"`: Stores the selected device.
        - `"MEDUSA_TORCH_GPU_ACCELERATION"`: `"1"` if CUDA is used, `"0"`
            otherwise.
    - If GPU is unavailable and no device is specified,
        `warn_gpu_not_available()` is called.

    Examples
    --------
    >>> device = config_pytorch()
    Selected device: NVIDIA GeForce RTX 3090
      - CUDA Device Index: 0
      - Compute Capability: 8.6
      - Total Memory: 24.00 GB

    >>> device = config_pytorch("cpu")
    Selected device: CPU
      - Cores: 8 (Threads used by PyTorch)
    """
    try:
        # Check PyTorch integration
        if os.environ.get("MEDUSA_TORCH_INTEGRATION") != '1':
            raise TorchExtrasNotInstalled()

        # Configure PyTorch device
        if device_name is None:
            if torch.cuda.is_available():
                device_name = 'cuda:0'
                device = torch.device(device_name)
            else:
                device_name = 'cpu'
                device = torch.device(device_name)
                warn_gpu_not_available()
        else:
            device = torch.device(device_name)

        # Set environment variables
        os.environ["MEDUSA_TORCH_DEVICE"] = device_name
        os.environ["MEDUSA_TORCH_GPU_ACCELERATION"] = "1" \
            if device.type == 'cuda' else "0"

        # Print device info
        print_device_info(device)

        return device

    except ImportError:
        raise TorchExtrasNotInstalled()


def print_device_info(device):
    """
    Prints detailed information about the selected computing device.

    Parameters
    ----------
    device : torch.device
        The PyTorch device object (CPU or CUDA).

    Notes
    -----
    - If the device is CUDA, it prints the device name, CUDA device index,
      compute capability, and total memory.
    - If the device is CPU, it prints the number of threads used by PyTorch.

    Examples
    --------
    >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    >>> print_device_info(device)
    Selected device: NVIDIA GeForce RTX 3090
      - CUDA Device Index: 0
      - Compute Capability: 8.6
      - Total Memory: 24.00 GB

    >>> print_device_info(torch.device("cpu"))
    Selected device: CPU
      - Cores: 8 (Threads used by PyTorch)
    """
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else 0
        device_name = torch.cuda.get_device_name(device_index)
        total_memory = torch.cuda.get_device_properties(
            device_index).total_memory / (1024 ** 3)  # Convert bytes to GB
        capability = torch.cuda.get_device_capability(device_index)

        print(f"Selected device: {device_name}")
        print(f"  - CUDA Device Index: {device_index}")
        print(f"  - Compute Capability: {capability[0]}.{capability[1]}")
        print(f"  - Total Memory: {total_memory:.2f} GB\n")
    else:
        print("Selected device: CPU")
        print(f"  - Cores: {torch.get_num_threads()} (Threads used by "
              f"PyTorch)\n")


def check_pytorch_config():
    """
    Checks if PyTorch has been configured within the Medusa environment.

    Returns
    -------
    int
        - `1` if PyTorch is configured and GPU acceleration status is set.
        - `0` if PyTorch is configured but GPU acceleration status is missing.
        - `-1` if PyTorch is not configured.

    Notes
    -----
    - The function checks the `MEDUSA_TORCH_INTEGRATION` environment variable.
    - GPU acceleration is checked using `MEDUSA_TORCH_GPU_ACCELERATION`.
    """
    if os.environ.get("MEDUSA_TORCH_INTEGRATION") == '1':
        return 1 if "MEDUSA_TORCH_GPU_ACCELERATION" in os.environ else 0
    return -1


def check_gpu_acceleration():
    """
    Checks if GPU acceleration is available and properly configured.

    Returns
    -------
    bool
        `True` if GPU acceleration is enabled, `False` otherwise.

    Raises
    ------
    PyTorchNotConfiguredError
        If PyTorch has not been properly configured.

    Notes
    -----
    - Calls `check_pytorch_config()` to verify PyTorch integration.
    - If PyTorch is not configured, an exception is raised.
    - Uses the `MEDUSA_TORCH_GPU_ACCELERATION` environment variable to determine GPU availability.
    """
    if check_pytorch_config() == -1:
        raise TorchNotConfiguredError()
    return os.environ.get("MEDUSA_TORCH_GPU_ACCELERATION") == '1'


def get_torch_device():
    """
    Retrieves the PyTorch device currently used by Medusa.

    Returns
    -------
    str
        The name of the configured PyTorch device (e.g., `'cuda:0'` or `'cpu'`).

    Raises
    ------
    PyTorchNotConfiguredError
        If PyTorch has not been properly configured.

    Notes
    -----
    - Calls `check_pytorch_config()` to verify PyTorch integration.
    - If PyTorch is not configured, an exception is raised.
    - Returns the value stored in the `MEDUSA_TORCH_DEVICE` environment
        variable.
    """
    if check_pytorch_config() == -1:
        raise TorchNotConfiguredError()
    return torch.device(os.environ["MEDUSA_TORCH_DEVICE"])


def warn_gpu_not_available():
    """
    Issues a warning when GPU acceleration is not available.

    Notes
    -----
    - This function is called when `config_pytorch()` detects that no GPU is
    available.
    - Uses `warnings.warn()` to provide a non-blocking alert.
    """
    warnings.warn(
"GPU acceleration is not available. PyTorch is running on CPU, "
        "which may result in slower performance.", UserWarning)
