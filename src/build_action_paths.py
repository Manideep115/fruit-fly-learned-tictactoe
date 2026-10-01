import json
from collections import deque

import networkx as nx
import pandas as pd


NEURON_FILE = "data/flynet/neurons_18input.csv"
CONNECTION_FILE = "data/flynet/connections_18input.csv"
INTERFACE_FILE = "data/flynet/interface.json"
OUTPUT_FILE = "data/flynet/action_paths.json"

MAX_HOPS = 3


def main():
    print("Loading neurons...")
    neurons = pd.read_csv(NEURON_FILE)

    print("Loading connections...")
    conns = pd.read_csv(
        CONNECTION_FILE,
        usecols=["pre_root_id", "post_root_id"]
    )

    interface = json.load(open(INTERFACE_FILE))
    output_neurons = interface["output_neurons"]

    valid_ids = set(neurons["root_id"].astype(int))

    conns = conns[
        conns["pre_root_id"].isin(valid_ids)
        & conns["post_root_id"].isin(valid_ids)
    ]

    print(f"Connections used: {len(conns):,}")

    # Reverse graph:
    # MBON <- upstream <- upstream <- ...
    graph = nx.DiGraph()

    graph.add_edges_from(
        zip(
            conns["pre_root_id"].astype(int),
            conns["post_root_id"].astype(int),
        )
    )

    action_paths = {}

    for action, output_id in enumerate(output_neurons):
        output_id = int(output_id)

        # BFS backwards from the MBON.
        distances = {output_id: 0}
        queue = deque([output_id])

        while queue:
            current = queue.popleft()
            depth = distances[current]

            if depth >= MAX_HOPS:
                continue

            for upstream in graph.predecessors(current):
                if upstream not in distances:
                    distances[upstream] = depth + 1
                    queue.append(upstream)

        # Keep actual edges connecting nodes inside this
        # reverse upstream region.
        path_edges = []

        for pre, post in graph.edges():
            if pre in distances and post in distances:
                # Edge must move toward the MBON.
                if distances[pre] == distances[post] + 1:
                    distance = distances[pre]

                    # Stronger weight for closer-to-output paths.
                    if distance == 1:
                        path_weight = 1.0
                    elif distance == 2:
                        path_weight = 0.7
                    else:
                        path_weight = 0.5

                    path_edges.append({
                        "pre": int(pre),
                        "post": int(post),
                        "distance": int(distance),
                        "weight": path_weight,
                    })

        action_paths[str(action)] = {
            "output_neuron": output_id,
            "neurons": {
                str(node): int(distance)
                for node, distance in distances.items()
            },
            "edges": path_edges,
        }

        print(
            f"Action {action}: "
            f"{len(distances):,} upstream neurons, "
            f"{len(path_edges):,} pathway edges"
        )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(action_paths, f)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()