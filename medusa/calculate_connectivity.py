import numpy as np
import scipy as sp
from medusa.connectivity import aec
from medusa.connectivity import pli
import warnings

def calculate_connectivity(signal,mode,signal_type,fs,param,trial_len,trial_overlapping,filt_mode,**config):
    
    # Variable debugging
    signal = np.array(signal)
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError("'Signal' contains non-numeric values")
    
    fs = np.array(fs)
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError("'fs' contains non-numeric values")
    elif fs.size != 1:
        raise ValueError("'fs' must be a number, not an array")
    
    param = param.upper()
    param = 'AEC' if param == 'AEC_ORT' else param # 'AEC' is rarely used, so we bypass its calculation to "AEC_ORT", and call "AEC" to "AEC_ORT"
    if param not in ("AEC","AEC_ORT","PLI"):
        raise ValueError("Unknown parameter")
    
    trial_len = np.array(trial_len)
    if not np.issubdtype(trial_len.dtype, np.integer):
        raise ValueError("'trial_len' must be an integer")
    elif trial_len > signal.shape[0]:
        raise ValueError("'trial_len' is bigger than signal length")
    elif trial_len < 1:
        raise ValueError("'trial_len' must be greater than 0")
    elif np.mod(signal.shape[0],trial_len):
        warnings.warn("The signal's sample number is not divisible by 'trial_len'. Signal will be truncated")
    
    trial_overlapping = np.array(trial_overlapping)
    if not np.issubdtype(signal.dtype, np.number):
        raise ValueError("'trial_overlapping' contains non-numeric values")        
    elif trial_overlapping >= 100 or trial_overlapping < 0:
        raise ValueError("'trial_overlapping' must be defined between 0 and 100 (and not equal to 100)")
    
    if filt_mode not in (0,1,2):
        raise ValueError("Unknown filtering mode")
    
    # Bands definition...
    # If no filter is wanted
    if filt_mode == 0:
        freq_size = 1
        flag = 0
    # If band filtering mode    
    elif filt_mode == 1:
        flag = 1
        
        # Variable debugging
        config["bands"] = config["bands"].lower()
        if "bands" not in config:
            raise ValueError("Band type not defined")
        elif config["bands"] not in ("classic","custom"):            
            raise ValueError("Unknown band definition mode")
        # Classical band definition    
        if config["bands"] == 'classic':
            bands_init = np.array([1, 4, 8, 13, 19, 30])
            bands_end = np.array([4, 8, 13, 19, 30, 70])
        elif config["bands"] == 'custom':
            # Variable debugging
            if "bands_init" not in config or "bands_end" not in config:
                raise ValueError("Bands not defined")
            bands_init = np.array(config["bands_init"])
            bands_end = np.array(config["bands_end"])  
            if (not np.issubdtype(bands_init.dtype, np.number) or not np.issubdtype(bands_end.dtype, np.number)):
                raise ValueError("Bands contain non-nummeric values")
              
    # If window filtering mode            
    elif filt_mode == 2:
        flag = 1
        # Variable debugging
        if "bandwidth" not in config or "overlapping" not in config or "limits" not in config:
            raise ValueError("In win mode, you have to define bandwidth, overlapping, and limits")
        overlapping = np.array(config["overlapping"])
        limits = np.array(config["limits"])
        bandwidth = np.array(config["bandwidth"])        
        if (not np.issubdtype(overlapping.dtype, np.number) or not np.issubdtype(bandwidth.dtype, np.number)
              or not np.issubdtype(limits.dtype, np.number)):
            raise ValueError("Bandwidth, overlapping or limits contains non-nummeric values")
        elif limits.size != 2 or bandwidth.size != 1 or overlapping.size != 1:
            raise ValueError("Limits must be a two-value array. Bandwidth and overlapping must be a number, not an array")
        elif limits[0] > limits[1]:
            raise ValueError("'limits' must be an increasing array")
        elif overlapping >= 100 or overlapping < 0:
            raise ValueError("'overlapping' must be defined between 0 and 100 (and not equal to 100)")
        else:
            # Bands definition
            overlapping = 1 - overlapping/100
            bands_init = np.arange(limits[0],(limits[1]-bandwidth)*1.0000001,bandwidth*overlapping) # I have to add 1% to reach the last freq. bin                                        
            bands_end = bands_init + bandwidth
    
    # Re-Referencing (Only for EEG signals)
    if signal_type == 'EEG':                
        signal_average = []
        matriz_ref = []
        signal_average = np.nanmean(signal, axis = 1)
        matriz_ref = signal - np.transpose(np.tile(signal_average,(signal.shape[1],1)))
        signal = []
        signal = matriz_ref
    
    # Parameter definition
    trial_len = trial_len * fs
    trial_overlapping = 1.0 - trial_overlapping/100
    freq_size = bands_init.shape[0] if flag else 1
    filter_order = 300 if 300 < np.trunc(trial_len/3) else np.trunc(trial_len/3).astype(int)     
    num_chan = signal.shape[1]
    trial_size = np.arange(0,signal.shape[0]-trial_len+1,np.trunc(trial_len*trial_overlapping).astype(int))
    adj_matrix = np.empty((freq_size,num_chan,num_chan,trial_size.shape[0]))
    adj_matrix[:] = np.nan

    idx = 0
    for jj in trial_size:
        
        for kk in range(0,freq_size):
            signal_trial = signal[jj.astype(int) : (jj + trial_len).astype(int), :]
            
            if flag:
                f_init = bands_init[kk]
                f_end = bands_end[kk]                
                # Filter design
                fc1  = f_init # First cutoff freq
                fc2  = f_end # Second cutoff freq
                num  = sp.signal.firwin(filter_order+1, np.array([fc1, fc2])/(fs/2), window='hamming', pass_zero='bandpass', scale=True) # Bandpass Design        
                # Filter the current trial
                signal_trial = sp.signal.filtfilt(num,1,signal_trial,axis=0,padtype='odd', padlen=3*(max(len(num),1)-1))           

            if param == 'WillNeverBeCalculated':
                adj_matrix[kk,:,:,idx] = aec.aec(signal_trial,mode,False)
            if param == 'AEC':
                adj_matrix[kk,:,:,idx] = aec.aec(signal_trial,mode,True)
            if param == 'PLI':
                adj_matrix[kk,:,:,idx] = pli.pli(signal_trial,mode)
                    
        idx = idx + 1
		
    info = dict()
    if filt_mode == 0:
        info["freq_overlap"] = 'no_filter' 
        info["freq_resolution"] = 'no_filter' 
        info["f_start"] = 'no_filter' 
        info["f_end"] = 'no_filter'
        info["bands_init"] = 'no_filter'
        info["bands_end"] = 'no_filter'       
    elif filt_mode == 1:
        info["freq_overlap"] = []
        info["freq_resolution"] = []
        info["f_start"] = bands_init[0] 
        info["f_end"] = bands_end[-1]
        info["bands_init"] = bands_init
        info["bands_end"] = bands_end
    elif filt_mode == 2:
        info["freq_overlap"] = (1 - overlapping) * 100 
        info["freq_resolution"] = bandwidth 
        info["f_start"] = bands_init[0] 
        info["f_end"] = bands_end[-1]
        info["bands_init"] = bands_init
        info["bands_end"] = bands_end
    
    info["trial_overlap"] = (1 - trial_overlapping) * 100
    info["length_trial"] = trial_len/fs
    info["measure"] = param
    info["info"] = 'Overlap is in percentage'
    return adj_matrix, info