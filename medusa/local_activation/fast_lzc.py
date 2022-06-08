import numpy as np
import ctypes


def lempelziv_fast(signal):

    # Adapt the signal to the format required by the DLL
    n_sample = signal.shape[0]
    n_cha = signal.shape[1]
    signal = signal.T.reshape(-1)

    # Get function from dll
    dll_file = ".\computeLZC.dll"
    lib = ctypes.cdll.LoadLibrary(dll_file)
    lzc_func = lib.computeLempelZivCmplx  # Access function

    # Create empty output vector
    lz_channel_values = np.zeros(int(n_cha), dtype=np.double)

    # Define inputs and outputs
    lzc_func.restype = None
    array_1d_double = np.ctypeslib.ndpointer(dtype=np.double, ndim=1,
                                             flags='CONTIGUOUS')
    lzc_func.argtypes = [array_1d_double, array_1d_double, ctypes.c_int32,
                         ctypes.c_int32]

    # Call the function in the dll
    lzc_func(signal, lz_channel_values, int(n_sample*n_cha), int(n_sample))

    return lz_channel_values


if __name__ == "__main__":

    import scipy.io

    mat = scipy.io.loadmat('P:/Usuarios/Victor_R/0001_Control.mat')
    vector = np.array(mat["signal"])[:, 0:5000]

    out = lempelziv_fast(vector.T)

