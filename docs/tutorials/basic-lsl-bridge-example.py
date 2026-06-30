"""
Author:   Eduardo Santamaría-Vázquez
Date:     10 Dec. 2022
Version:  0.1

Requirements: pylsl, numpy

"""

import time, threading
import pylsl
import numpy as np

#%% PARAMETERS

# LSL parameters
stream_name = 'SignalGenerator'
stream_type = 'EEG'
source_id = '433b4d0a-78ae-11ed-a1eb-0242ac120002'
chunk_size = 16
format = 'float32'
n_cha = 8
l_cha = [str(i) for i in range(8)]
units = 'uV'
manufacturer = 'MEDUSA'
sample_rate = 250

# Random signal parameters
mean = 0
std = 1

#%% CREATE LSL OUTLET

# Create the stream info
lsl_info = pylsl.StreamInfo(name=stream_name,
                            type=stream_type,
                            channel_count=n_cha,
                            nominal_srate=sample_rate,
                            channel_format=format,
                            source_id=source_id)

# Modify description to include additional information (e.g., manufacturer)
lsl_info.desc().append_child_value("manufacturer", manufacturer)

# Append channel information. By default, MEDUSA© Platform expects this
# information in the "channels" section of the LSL stream description
channels = lsl_info.desc().append_child("channels")
for l in l_cha:
    channels.append_child("channel") \
        .append_child_value("label", l) \
        .append_child_value("units", units) \
        .append_child_value("type", stream_type)

# Create LSL outlet
lsl_outlet = pylsl.StreamOutlet(info=lsl_info,
                                chunk_size=chunk_size,
                                max_buffered=100*chunk_size)

#%% STREAM DATA


def send_data():
    """Function that generates random data and sends it through LSL
    """
    while io_run.is_set():
        try:
            if lsl_outlet is not None:
                # Get data
                # --------------------------------------------------------------
                # TODO: Get the data from an actual device using its API
                # For this tutorial, we will generate random data
                sample = std * np.random.randn(chunk_size, n_cha) + mean
                sample = sample.tolist()
                # --------------------------------------------------------------
                # Get the timestamp of the chunk
                timestamp = pylsl.local_clock()
                # Send the chunk through LSL
                lsl_outlet.push_chunk(sample, timestamp)
                # Wait for the next chunk. This timer is not particularly
                # accurate
                time.sleep(chunk_size / sample_rate)
        except Exception as e:
            raise e


# Run data source in other thread so the execution can be stopped on demand
io_run = threading.Event()
io_run.set()

lsl_thread = threading.Thread(
    name='SignalGeneratorThread',
    target=send_data
)
lsl_thread.start()

# Streaming data...
print('SignalGenerator is streaming data. Open MEDUSA and check that the '
      'stream is received correctly')

#%% STOP EXECUTION AND CLEAR

# Pause the main thread until the user presses enter
input("Press enter to finish...")

# Stop the thread and join
io_run.clear()
lsl_thread.join()

# Final message
print('SignalGenerator finished successfully')

