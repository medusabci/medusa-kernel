files = ["10-05.tsv", "10-10.tsv", "10-20.tsv"]
output = "eeg_standard_2D.tsv"

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

all_channels = dict()
for f in files:
    with open(f, 'r') as file:
        for line in file:
            label, x, y = line.strip().split('\t')
            if label != "label":
                all_channels[label] = (float(x), float(y))

for ch1, ch2 in eeg_channel_synonyms.items():
    if ch1 not in all_channels:
        if ch2 in all_channels:
            all_channels[ch1] = all_channels[ch2]
            print(f"Added ch {ch1}")
    if ch2 not in all_channels:
        if ch1 in all_channels:
            all_channels[ch2] = all_channels[ch1]
            print(f"Added ch {ch2}")

print(f"> Total number of channels: {len(all_channels)}")

print(f"Saving into {output}")
with open(output, 'w') as file:
    file.write("label\tx\ty")
    for label, coords in all_channels.items():
        file.write("\n%s\t%.4f\t%.4f" % (label, coords[0], coords[1]))
print("Done!")
