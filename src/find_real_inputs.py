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

print("🪰 Finding real FlyNet input neurons...")
print("=" * 60)

classification = pd.read_csv(CLASSIFICATION)
cell_types = pd.read_csv(CELL_TYPES)

classification = classification.merge(
    cell_types[["root_id", "primary_type"]],
    on="root_id",
    how="left",
)

# Our current internal FlyNet
selected = classification[
    classification["class"].isin(TARGET_CLASSES)
].copy()

fly_ids = set(selected["root_id"])

print(f"FlyNet neurons: {len(fly_ids):,}")

# Candidate neurons must be OUTSIDE the FlyNet
# but send connections INTO it.
incoming = {}

for chunk in pd.read_csv(CONNECTIONS, chunksize=100_000):

    mask = (
        ~chunk["pre_root_id"].isin(fly_ids)
        & chunk["post_root_id"].isin(fly_ids)
    )

    boundary = chunk[mask]

    for _, row in boundary.iterrows():

        source = row["pre_root_id"]

        incoming[source] = (
            incoming.get(source, 0)
            + row["syn_count"]
        )

# Metadata for candidate neurons
candidate_ids = set(incoming)

candidate_info = classification[
    classification["root_id"].isin(candidate_ids)
].copy()

# Score candidates by total incoming-to-FlyNet strength
candidate_info["strength"] = (
    candidate_info["root_id"]
    .map(incoming)
)

candidate_info = candidate_info.sort_values(
    "strength",
    ascending=False,
)

print("\nTop biological input candidates:")
print("-" * 60)

for _, row in candidate_info.head(30).iterrows():

    print(
        f"{int(row['root_id'])} | "
        f"{row['primary_type']} | "
        f"{row['class']} | "
        f"{int(row['strength'])}"
    )