import pandas as pd

CLASSIFICATION = "data/classification/classification.csv"
CELL_TYPES = "data/cell_types/consolidated_cell_types.csv"
CONNECTIONS = "data/connection/connections_princeton.csv"

TARGET_CLASSES = {
    "olfactory",
    "Kenyon_Cell",
    "DAN",
    "MBON",
    "MBIN",
}

print("🪰 Finding FlyNet boundaries...")
print("=" * 50)

# --------------------------------------------------
# Load classification + cell type information
# --------------------------------------------------

classification = pd.read_csv(CLASSIFICATION)
cell_types = pd.read_csv(CELL_TYPES)

# Add primary neuron type to classification
classification = classification.merge(
    cell_types[["root_id", "primary_type"]],
    on="root_id",
    how="left"
)

# --------------------------------------------------
# Get our FlyNet neurons
# --------------------------------------------------

selected = classification[
    classification["class"].isin(TARGET_CLASSES)
].copy()

fly_ids = set(selected["root_id"])

print(f"FlyNet neurons: {len(fly_ids):,}")

# --------------------------------------------------
# Find connections crossing our boundary
# --------------------------------------------------

incoming = {}
outgoing = {}

for chunk in pd.read_csv(CONNECTIONS, chunksize=100_000):

    # Outside → FlyNet
    mask_in = (
        ~chunk["pre_root_id"].isin(fly_ids)
        & chunk["post_root_id"].isin(fly_ids)
    )

    for _, row in chunk[mask_in].iterrows():
        neuron = row["post_root_id"]
        incoming[neuron] = incoming.get(neuron, 0) + row["syn_count"]

    # FlyNet → Outside
    mask_out = (
        chunk["pre_root_id"].isin(fly_ids)
        & ~chunk["post_root_id"].isin(fly_ids)
    )

    for _, row in chunk[mask_out].iterrows():
        neuron = row["pre_root_id"]
        outgoing[neuron] = outgoing.get(neuron, 0) + row["syn_count"]

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\n📥 Strongest INPUT candidates")
print("-" * 60)

for neuron, strength in sorted(
    incoming.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    row = selected[selected["root_id"] == neuron].iloc[0]

    print(
        f"{neuron} | "
        f"{row['primary_type']} | "
        f"{row['class']} | "
        f"{strength:.0f}"
    )

print("\n📤 Strongest OUTPUT candidates")
print("-" * 60)

for neuron, strength in sorted(
    outgoing.items(),
    key=lambda x: x[1],
    reverse=True
)[:20]:

    row = selected[selected["root_id"] == neuron].iloc[0]

    print(
        f"{neuron} | "
        f"{row['primary_type']} | "
        f"{row['class']} | "
        f"{strength:.0f}"
    )

print("\nDone.")