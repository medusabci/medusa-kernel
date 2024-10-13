# files = ["10-05.tsv", "10-10.tsv", "10-20.tsv"]
# output = "international_system.tsv"
#
# all_channels = dict()
# for f in files:
#     print(f"Reading {f}...")
#     with open(f, 'r') as file:
#         for line in file:
#             label, x, y = line.strip().split('\t')
#             if label != "label":
#                 all_channels[label]= (float(x), float(y))
#
# print(f"Saving into {output}")
# with open(output, 'w') as file:
#     file.write("label\tx\ty")
#     for label, coords in all_channels.items():
#         file.write("\n%s\t%.4f\t%.4f" % (label, coords[0], coords[1]))
# print("Done!")

#%%
eeg_channel_synonyms = {
    'I1': 'O9',
    'I2': 'O10',
    'O9': 'I1',
    'O10': 'I2',
    'T3': 'T7',
    'T4': 'T8',
    'T7': 'T3',
    'T8': 'T4',
    'T5': 'P7',
    'T6': 'P8',
    'A1': 'LPA',
    'A2': 'RPA',
    'NAS': 'NZ'
}

# eeg2d = "../eeg_standard_2D.tsv"
eeg2d = "../eeg_standard_10-05_3D.tsv"

# all_channels = dict()
# with open(eeg2d, 'r') as file:
#     for line in file:
#         label, x, y, z = line.strip().split('\t')
#         if label != "label":
#             all_channels[label] = (float(x), float(y), float(z))
#
# for ch1, ch2 in eeg_channel_synonyms.items():
#     if ch1 not in all_channels:
#         if ch2 in all_channels:
#             all_channels[ch1] = all_channels[ch2]
#             print(f"Added ch {ch1}")
#     if ch2 not in all_channels:
#         if ch1 in all_channels:
#             all_channels[ch2] = all_channels[ch1]
#             print(f"Added ch {ch2}")
#
# with open("eeg_standard_3D.tsv", 'w') as file:
#     file.write("label\tx\ty\tz")
#     for label, coords in all_channels.items():
#         file.write("\n%s\t%.4f\t%.4f\t%.4f" % (label, coords[0], coords[1], coords[2]))
# print("Done!")

eeg2d = "10-05.tsv"
eeg3d = "../eeg_standard_10-05_3D.tsv"

all_channels = dict()
with open(eeg2d, 'r') as file:
    for line in file:
        label, x, y = line.strip().split('\t')
        if label != "label":
            all_channels[label] = (float(x), float(y))
#
# all_channels3d = dict()
# with open(eeg3d, 'r') as file:
#     for line in file:
#         label, x, y, z = line.strip().split('\t')
#         if label != "label":
#             all_channels3d[label.upper()] = (float(x), float(y), float(z))
#
# with open("eeg_standard_3D.tsv", 'w') as file:
#     file.write("label\tx\ty\tz")
#     for label, coords in all_channels3d.items():
#         file.write("\n%s\t%.4f\t%.4f\t%.4f" % (label, coords[0], coords[1], coords[2]))
# print("Done!")
#
#
# print("\nChannels in 2D:")
# for ch3d in all_channels3d.keys():
#     if ch3d not in all_channels:
#         print(f"> Channel {ch3d} not found in 2D")
#
# print("\nChannels in 3D:")
# for ch2d in all_channels.keys():
#     if ch2d not in all_channels3d:
#         print(f"> Channel {ch2d} not found in 3D")
