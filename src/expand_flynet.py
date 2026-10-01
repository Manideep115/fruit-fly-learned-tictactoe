import pandas as pd
from pathlib import Path

CLASSIFICATION = "data/classification/classification.csv"
CELL_TYPES = "data/cell_types/consolidated_cell_types.csv"
CONNECTIONS = "data/connection/connections_princeton.csv"

OUTPUT_DIR = Path("data/flynet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORE_CLASSES = {
    "olfactory",
    "Kenyon_Cell",
    "DAN",
    "MBON",
    "MBIN",
}

INPUT_NEURONS = {
    720575940618308825,
    720575940630770042,
    720575940622726271,
    720575940619071005,
    720575940637056887,
    720575940621529435,
    720575940648650884,
    720575940611079236,
    720575940623528925,

    720575940635983770,
    720575940609460491,
    720575940636873791,
    720575940618295454,
    720575940617668747,
    720575940628467611,
    720575940637594334,
    720575940614727903,
    720575940632698797,
}

print("🪰 Expanding FlyNet for 18-input interface...")
print("=" * 60)

classification = pd.read_csv(CLASSIFICATION)
cell_types = pd.read_csv(CELL_TYPES)

classification = classification.merge(
    cell_types[["root_id", "primary_type"]],
    on="root_id",
    how="left",
)

core = classification[
    classification["class"].isin(CORE_CLASSES)
].copy()

core_ids = set(core["root_id"])
all_ids = core_ids | INPUT_NEURONS

print(f"Core neurons       : {len(core_ids):,}")
print(f"Input neurons      : {len(INPUT_NEURONS):,}")
print(f"Expanded neurons   : {len(all_ids):,}")

expanded_neurons = classification[
    classification["root_id"].isin(all_ids)
].copy()

expanded_neurons.to_csv(
    OUTPUT_DIR / "neurons_18input.csv",
    index=False,
)

output_file = OUTPUT_DIR / "connections_18input.csv"

first_chunk = True
total_connections = 0

for chunk in pd.read_csv(
    CONNECTIONS,
    chunksize=100_000,
):

    mask = (
        chunk["pre_root_id"].isin(all_ids)
        & chunk["post_root_id"].isin(all_ids)
    )

    filtered = chunk[mask]

    if not filtered.empty:

        filtered.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False
        total_connections += len(filtered)

print()
print("=" * 60)
print("🪰 18-INPUT FLYNET READY")
print("=" * 60)
print(f"Neurons     : {len(expanded_neurons):,}")
print(f"Connections : {total_connections:,}")