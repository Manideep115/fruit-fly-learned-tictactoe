import pandas as pd

FILE = "data/connection/connections_princeton.csv"

total_connections = 0
neurons = set()
nt_counts = {}

for chunk in pd.read_csv(FILE, chunksize=100_000):
    total_connections += len(chunk)

    neurons.update(chunk["pre_root_id"])
    neurons.update(chunk["post_root_id"])

    counts = chunk["nt_type"].value_counts()

    for nt, count in counts.items():
        nt_counts[nt] = nt_counts.get(nt, 0) + count

print("\n🪰 FLY BRAIN SUMMARY")
print("--------------------")
print(f"Connections : {total_connections:,}")
print(f"Neurons     : {len(neurons):,}")

print("\nNeurotransmitters:")
for nt, count in sorted(nt_counts.items(), key=lambda x: -x[1]):
    print(f"  {nt:10} {count:,}")