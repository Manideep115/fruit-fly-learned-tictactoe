import pandas as pd
import networkx as nx

CONNECTIONS = "data/flynet/connections_expanded.csv"
CLASSIFICATION = "data/classification/classification.csv"

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
}

OUTPUT_NEURONS = {
    720575940607155890,
    720575940611344078,
    720575940629981440,
    720575940643863496,
    720575940626315010,
    720575940632943277,
    720575940628734376,
    720575940620464321,
    720575940637902938,
}

print("🪰 Checking paths through FlyNet...")
print("=" * 60)

connections = pd.read_csv(CONNECTIONS)

# Build directed graph
graph = nx.DiGraph()

graph.add_edges_from(
    zip(
        connections["pre_root_id"],
        connections["post_root_id"],
    )
)

print(f"Graph nodes: {graph.number_of_nodes():,}")
print(f"Graph edges: {graph.number_of_edges():,}")

print("\nInput → MBON reachability")
print("-" * 60)

reachable_pairs = 0

for input_neuron in INPUT_NEURONS:

    reached = []

    for output_neuron in OUTPUT_NEURONS:

        if nx.has_path(
            graph,
            input_neuron,
            output_neuron,
        ):
            reached.append(output_neuron)
            reachable_pairs += 1

    print(
        f"{input_neuron} → "
        f"{len(reached)}/{len(OUTPUT_NEURONS)} MBONs reachable"
    )

print("\nTotal reachable input/output pairs:")
print(reachable_pairs)

# Find shortest paths for reachable pairs
print("\nShortest path examples")
print("-" * 60)

count = 0

for input_neuron in INPUT_NEURONS:

    for output_neuron in OUTPUT_NEURONS:

        try:
            path = nx.shortest_path(
                graph,
                input_neuron,
                output_neuron,
            )

            print(
                f"{input_neuron} → "
                f"{output_neuron}: "
                f"{len(path) - 1} hops"
            )

            count += 1

            if count >= 10:
                break

        except nx.NetworkXNoPath:
            pass

    if count >= 10:
        break