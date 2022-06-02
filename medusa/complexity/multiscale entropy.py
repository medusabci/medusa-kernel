import numpy as np
from scipy.signal import  decimate
import threading


from entropy_algorithms import sample_entropy

def MSE(signal,MaxScale,m = 1,r = 0.3):

    mse_result = np.empty(MaxScale)
    working_threads = list()

    for i in range(1,MaxScale+1):
        if i == 1:
            t = threading.Thread(target=sample_entropy, args=(signal,m,r,'chebyshev',mse_result,i))
            # mse_result[i-1] = SampEn(signal,m,r)
            working_threads.append(t)
            t.start()
        else:
            t = threading.Thread(target=sample_entropy, args=(coarse_grain(signal,i),m,r,'chebyshev', mse_result,i))
            # mse_result[i-1] = SampEn(signal,m,r)
            working_threads.append(t)
            t.start()
    for t in reversed(working_threads):
        t.join()

    return mse_result


def coarse_grain(signal, scale, decimate_mode=True):

    if decimate_mode:
        return decimate(signal, scale)
    else:
        N = len(signal)  # Signal length
        tau = int(round(N / scale))  # Number of coarse grains in which the
        # signal is splitted
        y = np.empty(tau)  # Returned signal
        for i in range(tau):
            y[i] = np.mean(signal[i*scale:(i*scale + scale)])
        return y

if __name__ == '__main__':
    import time
    import scipy.signal as ss
    import matplotlib.pyplot as plt

    t = np.linspace(0, 100, 60000)
    w = ss.chirp(t, f0=6, f1=16, t1=100, method='linear')
    start = time.time()
    mse = MSE(w,20)
    end = time.time()
    plt.plot(mse,'-o')
    plt.show()
    print(end-start)
