import pandas as pd

CLASSIFICATION = "data/classification/classification.csv"
CONNECTIONS = "data/connection/connections_princeton.csv"

# Classes we are interested in
TARGET_CLASSES = {
    "olfactory",
    "Kenyon_Cell",
    "MBON",
    "MBIN",
    "DAN",
}

# Load neuron classifications
classes = pd.read_csv(CLASSIFICATION)

selected = classes[
    classes["class"].isin(TARGET_CLASSES)
]

neuron_ids = set(selected["root_id"])

print("\n🪰 LEARNING CIRCUIT")
print("=" * 50)

print(f"Selected neurons: {len(neuron_ids):,}")

print("\nNeurons by class:")
print(selected["class"].value_counts().to_string())

# Scan connections in chunks
total = 0
internal = 0

for chunk in pd.read_csv(CONNECTIONS, chunksize=100_000):

    total += len(chunk)

    internal += (
        chunk["pre_root_id"].isin(neuron_ids)
        & chunk["post_root_id"].isin(neuron_ids)
    ).sum()

print("\nConnections:")
print(f"Total brain connections scanned : {total:,}")
print(f"Connections inside our circuit  : {internal:,}")