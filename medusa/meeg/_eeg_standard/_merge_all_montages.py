files = ["10-05.tsv", "10-10.tsv", "10-20.tsv"]
output = "international_system.tsv"

all_channels = dict()
for f in files:
    print(f"Reading {f}...")
    with open(f, 'r') as file:
        for line in file:
            label, x, y = line.strip().split('\t')
            if label != "label":
                all_channels[label]= (float(x), float(y))

print(f"Saving into {output}")
with open(output, 'w') as file:
    file.write("label\tx\ty")
    for label, coords in all_channels.items():
        file.write("\n%s\t%.4f\t%.4f" % (label, coords[0], coords[1]))
print("Done!")
