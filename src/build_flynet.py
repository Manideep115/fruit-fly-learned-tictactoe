import pandas as pd
from pathlib import Path

CLASSIFICATION = "data/classification/classification.csv"
CELL_TYPES = "data/cell_types/consolidated_cell_types.csv"
CONNECTIONS = "data/connection/connections_princeton.csv"

OUTPUT_DIR = Path("data/flynet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CLASSES = {
    "olfactory",
    "Kenyon_Cell",
    "DAN",
    "MBON",
    "MBIN",
}

print("🪰 Building FlyNet...")
print("=" * 50)

# --------------------------------------------------
# 1. Find neurons belonging to our learning circuit
# --------------------------------------------------

classification = pd.read_csv(CLASSIFICATION)

selected = classification[
    classification["class"].isin(TARGET_CLASSES)
].copy()

neuron_ids = set(selected["root_id"])

print(f"Selected neurons: {len(neuron_ids):,}")

# --------------------------------------------------
# 2. Save neuron information
# --------------------------------------------------

cell_types = pd.read_csv(CELL_TYPES)

neurons = selected.merge(
    cell_types,
    on="root_id",
    how="left"
)

neurons.to_csv(
    OUTPUT_DIR / "neurons.csv",
    index=False
)

print("Saved neurons.csv")

# --------------------------------------------------
# 3. Extract connections between selected neurons
# --------------------------------------------------

output_connections = OUTPUT_DIR / "connections.csv"

first_chunk = True
total = 0

for chunk in pd.read_csv(CONNECTIONS, chunksize=100_000):

    filtered = chunk[
        chunk["pre_root_id"].isin(neuron_ids)
        & chunk["post_root_id"].isin(neuron_ids)
    ]

    if len(filtered) > 0:

        filtered.to_csv(
            output_connections,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False
        total += len(filtered)

        print(f"Extracted: {total:,} connections")

print("\n" + "=" * 50)
print("🪰 FLYNET READY")
print("=" * 50)
print(f"Neurons:     {len(neurons):,}")
print(f"Connections: {total:,}")
print(f"Location:    {OUTPUT_DIR}")