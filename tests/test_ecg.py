import numpy as np
import json

# Import the ECG module (assuming it's saved as 'ecg_module.py')
from medusa.ecg import *


# Function to test get_standard_montage()
def test_get_standard_montage():
    print("\nTesting get_standard_montage()...")

    # Test retrieving electrode placements
    electrodes = get_standard_montage("12leads", "electrodes")
    assert isinstance(electrodes, dict), "Electrode data should be a dictionary"
    assert "RA" in electrodes and "V6" in electrodes, "Missing expected electrodes"
    print("✅ Electrode retrieval test passed!")

    # Test retrieving lead configurations
    leads = get_standard_montage("12leads", "leads")
    assert isinstance(leads, dict), "Lead data should be a dictionary"
    assert "I" in leads and "V6" in leads, "Missing expected leads"
    print("✅ Lead retrieval test passed!")

    # Test invalid standard
    try:
        get_standard_montage("leads", "invalid_standard")
    except ValueError as e:
        print("✅ Invalid standard test passed!")
    else:
        print("❌ Invalid standard test failed!")


# Function to test ECGChannelSet class
def test_ecg_channel_set():
    print("\nTesting ECGChannelSet class...")

    # Create an ECG channel set with 'leads' mode
    ecg_set = ECGChannelSet(channel_mode='leads')

    # Test adding channels
    ecg_set.add_channel("I", "Lead I: LA - RA")
    ecg_set.add_channel("II", "Lead II: LL - RA")

    assert len(ecg_set.channels) == 2, "Channel addition failed"
    assert ecg_set.channels[0]['label'] == "I", "Incorrect first channel label"
    print("✅ Channel addition test passed!")

    # Test setting ground electrode
    ecg_set.set_ground("RL")
    assert ecg_set.ground == "RL", "Ground setting failed"
    print("✅ Ground setting test passed!")

    # Test serialization
    ser_str = json.dumps(ecg_set.to_serializable_obj())
    print("✅ Serialization test passed!")


# Function to test ECG class
def test_ecg_class():
    print("\nTesting ECG class...")

    # Generate dummy ECG data
    times = np.linspace(0, 10, 500)  # 10-second signal with 500 samples
    signal = np.random.randn(500, 12)  # 12-channel simulated ECG data
    fs = 50  # Sample rate in Hz

    # Create ECGChannelSet with leads
    ecg_channel_set = ECGChannelSet(channel_mode='leads')
    for lead in ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]:
        ecg_channel_set.add_channel(lead, f"Lead {lead}")

    # Create ECG object
    ecg_data = ECG(times, signal, fs, ecg_channel_set)

    # Test basic attributes
    assert ecg_data.fs == fs, "Incorrect sample rate"
    assert ecg_data.signal.shape == (500, 12), "Incorrect signal shape"
    assert ecg_data.channel_set.n_cha == 12, "Incorrect number of channels"
    print("✅ ECG class instantiation test passed!")


# Run tests
if __name__ == "__main__":
    test_get_standard_montage()
    test_ecg_channel_set()
    test_ecg_class()
    print("\nAll tests completed successfully! ✅")
