import json
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

print("🪰 Selecting biological FlyNet interface...")
print("=" * 55)

# Load metadata
classification = pd.read_csv(CLASSIFICATION)
cell_types = pd.read_csv(CELL_TYPES)

classification = classification.merge(
    cell_types[["root_id", "primary_type"]],
    on="root_id",
    how="left",
)

selected = classification[
    classification["class"].isin(TARGET_CLASSES)
].copy()

fly_ids = set(selected["root_id"])

# --------------------------------------------------
# Input candidates
# Restrict to olfactory neurons
# --------------------------------------------------

olfactory_ids = set(
    selected.loc[
        selected["class"] == "olfactory",
        "root_id"
    ]
)

# Output candidates = MBONs
mbon_ids = set(
    selected.loc[
        selected["class"] == "MBON",
        "root_id"
    ]
)

incoming = {}
outgoing = {}

print(f"Olfactory candidates: {len(olfactory_ids):,}")
print(f"MBON candidates:      {len(mbon_ids):,}")

# --------------------------------------------------
# Find external connections
# --------------------------------------------------

for chunk in pd.read_csv(CONNECTIONS, chunksize=100_000):

    # Outside brain → olfactory neuron
    mask_in = (
        ~chunk["pre_root_id"].isin(fly_ids)
        & chunk["post_root_id"].isin(olfactory_ids)
    )

    for _, row in chunk[mask_in].iterrows():
        neuron = row["post_root_id"]
        incoming[neuron] = (
            incoming.get(neuron, 0)
            + row["syn_count"]
        )

    # MBON → outside brain
    mask_out = (
        chunk["pre_root_id"].isin(mbon_ids)
        & ~chunk["post_root_id"].isin(fly_ids)
    )

    for _, row in chunk[mask_out].iterrows():
        neuron = row["pre_root_id"]
        outgoing[neuron] = (
            outgoing.get(neuron, 0)
            + row["syn_count"]
        )

# --------------------------------------------------
# Pick strongest unique neurons
# --------------------------------------------------

input_neurons = sorted(
    incoming,
    key=incoming.get,
    reverse=True
)[:9]

output_neurons = sorted(
    outgoing,
    key=outgoing.get,
    reverse=True
)[:9]

print("\n📥 SELECTED INPUT NEURONS")
print("-" * 55)

for i, neuron in enumerate(input_neurons):
    row = selected[
        selected["root_id"] == neuron
    ].iloc[0]

    print(
        f"Input {i}: "
        f"{neuron} | "
        f"{row['primary_type']} | "
        f"incoming={incoming[neuron]}"
    )

print("\n📤 SELECTED OUTPUT NEURONS")
print("-" * 55)

for i, neuron in enumerate(output_neurons):
    row = selected[
        selected["root_id"] == neuron
    ].iloc[0]

    print(
        f"Move {i}: "
        f"{neuron} | "
        f"{row['primary_type']} | "
        f"outgoing={outgoing[neuron]}"
    )

# --------------------------------------------------
# Save configuration
# --------------------------------------------------

config = {
    "input_neurons": [int(x) for x in input_neurons],
    "output_neurons": [int(x) for x in output_neurons],
}

with open(
    "data/flynet/interface.json",
    "w"
) as f:
    json.dump(config, f, indent=2)

print("\n✅ Saved:")
print("data/flynet/interface.json")