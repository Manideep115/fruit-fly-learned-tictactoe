import pandas as pd

connections = pd.read_csv(
    "data/connection/connections_princeton.csv",
    nrows=1000
)

cell_types = pd.read_csv(
    "data/cell_types/consolidated_cell_types.csv"
)

classification = pd.read_csv(
    "data/classification/classification.csv"
)

print("\n=== CONNECTIONS ===")
print(connections.columns.tolist())
print(connections.head())

print("\n=== CELL TYPES ===")
print(cell_types.columns.tolist())
print(cell_types.head())

print("\n=== CLASSIFICATION ===")
print(classification.columns.tolist())
print(classification.head())