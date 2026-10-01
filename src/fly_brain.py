import numpy as np
import pandas as pd
import json

class FlyBrain:
    """
    FlyWire-derived spiking network.

    The FlyNet itself learns through reward-modulated
    eligibility traces.

    No external policy network is used.
    """

    def __init__(
        self,
        neuron_file="data/flynet/neurons_18input.csv",
        connection_file="data/flynet/connections_18input.csv",
        seed=42,
    ):
        with open("data/flynet/action_paths.json", "r") as f:
            self.action_paths = json.load(f)
        self.rng = np.random.default_rng(seed)

        print("🪰 Loading FlyNet...")

        # -----------------------------
        # Neurons
        # -----------------------------

        neurons = pd.read_csv(neuron_file)

        self.neuron_ids = (
            neurons["root_id"]
            .astype(np.int64)
            .tolist()
        )

        self.id_to_index = {
            neuron_id: i
            for i, neuron_id in enumerate(self.neuron_ids)
        }

        self.num_neurons = len(self.neuron_ids)

        # -----------------------------
        # Connections
        # -----------------------------

        connections = pd.read_csv(connection_file)

        self.pre = np.array(
            [
                self.id_to_index[x]
                for x in connections["pre_root_id"]
            ],
            dtype=np.int32,
        )

        self.post = np.array(
            [
                self.id_to_index[x]
                for x in connections["post_root_id"]
            ],
            dtype=np.int32,
        )

        syn_count = connections[
            "syn_count"
        ].to_numpy(dtype=np.float32)

        nt_type = (
            connections["nt_type"]
            .fillna("")
            .to_numpy()
        )

        # -----------------------------
        # Biological sign
        # -----------------------------

        excitatory = np.isin(
            nt_type,
            ["ACH", "GLUT"],
        )

        inhibitory = nt_type == "GABA"

        self.synapse_sign = np.zeros(
            len(nt_type),
            dtype=np.float32,
        )

        self.synapse_sign[excitatory] = 1.0
        self.synapse_sign[inhibitory] = -1.0

        active = excitatory | inhibitory

        weights = syn_count.copy()
        weights[~active] = 0.0

        # --------------------------------------------
        # Normalize incoming strength per neuron
        # --------------------------------------------

        incoming_strength = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        np.add.at(
            incoming_strength,
            self.post,
            np.abs(weights),
        )

        incoming_strength[
            incoming_strength == 0
        ] = 1.0

        weights /= incoming_strength[
            self.post
        ]

        # Global gain
        weights *= 1.5

        self.weights = (
            weights * self.synapse_sign
        )

        self.max_magnitude = 2.0

        # -----------------------------
        # Neural state
        # -----------------------------

        self.potential = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        self.spikes = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        # -----------------------------
        # Simulation
        # -----------------------------

        self.decay = 0.92
        self.threshold = 0.50

        # -----------------------------
        # Learning
        # -----------------------------

        self.learning_rate = 0.001

        self.eligibility_decay = 0.90
        self.spike_trace_decay = 0.85

        print(
            f"Neurons:     {self.num_neurons:,}"
        )
        print(
            f"Connections: {len(self.weights):,}"
        )

    def reset(self):
        self.potential.fill(0)
        self.spikes.fill(0)

    def step(self, external_input):
        current = external_input.astype(
            np.float32
        ).copy()

        if np.any(self.spikes):
            np.add.at(
                current,
                self.post,
                self.spikes[self.pre]
                * self.weights,
            )

        self.potential *= self.decay
        self.potential += current

        self.spikes = (
            self.potential >= self.threshold
        ).astype(np.float32)

        self.potential[
            self.spikes > 0
        ] = 0.0

        return self.spikes.copy()

    def run(
        self,
        external_input,
        steps=20,
    ):
        """
        Present a brief input pulse and let the
        FlyNet process it.

        Returns:
            activity
            eligibility
            final membrane potential
        """

        self.reset()

        activity = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        eligibility = np.zeros(
            len(self.weights),
            dtype=np.float32,
        )

        neuron_trace = np.zeros(
            self.num_neurons,
            dtype=np.float32,
        )

        for step in range(steps):

            if step < 5:
                current_input = external_input
            else:
                current_input = np.zeros(
                    self.num_neurons,
                    dtype=np.float32,
                )

            spikes = self.step(
                current_input
            )

            activity += spikes

            # ----------------------------------
            # Eligibility trace
            #
            # Presynaptic activity paired with postsynaptic
            # activation (spikes + subthreshold depolarization).
            # ----------------------------------

            post_act = spikes[self.post] + np.maximum(
                0.0, self.potential[self.post]
            )

            coactivity = (
                neuron_trace[self.pre]
                * post_act
            )

            eligibility *= (
                self.eligibility_decay
            )

            eligibility += coactivity

            neuron_trace *= self.spike_trace_decay
            neuron_trace += spikes

        return (
            activity,
            eligibility,
            self.potential.copy(),
            neuron_trace.copy(),
        )

    def learn(self, eligibility, reward, action, neuron_trace=None):
        """
        Reward-modulated learning on the selected action pathway.

        Only synapses that:
        1. belong to the selected action pathway, AND
        2. actually acquired eligibility during the game
        are modified.
        """

        if reward == 0:
            return

        action_data = self.action_paths[str(action)]

        if not hasattr(self, "_edge_to_index"):
            self._edge_to_index = {}

            for i, (pre, post) in enumerate(
                zip(self.pre, self.post)
            ):
                pre_id = int(self.neuron_ids[pre])
                post_id = int(self.neuron_ids[post])

                self._edge_to_index[
                    (pre_id, post_id)
                ] = i

        guided = np.zeros(
            len(self.weights),
            dtype=np.float32,
        )

        for edge in action_data["edges"]:
            idx = self._edge_to_index.get(
                (edge["pre"], edge["post"])
            )

            if idx is not None:
                dist = edge.get("distance", 3)
                if dist == 1 and neuron_trace is not None:
                    guided[idx] = edge["weight"] * neuron_trace[self.pre[idx]]
                else:
                    guided[idx] = edge["weight"] * eligibility[idx]

        # Only modify excitatory synapses to prevent rewarding an action from increasing inhibition
        guided *= (self.synapse_sign > 0)

        scale = np.max(
            np.abs(guided)
        )

        if scale <= 0:
            return

        normalized = guided / scale

        magnitude = np.abs(
            self.weights
        )

        learning_step = (
            self.learning_rate
            * 2.0
            * reward
            * normalized
        )

        magnitude += learning_step

        magnitude = np.clip(
            magnitude,
            0.0,
            self.max_magnitude,
        )

        self.weights = (
            magnitude
            * self.synapse_sign
        )

    def save_weights(self, path):
        np.save(
            path,
            self.weights,
        )

        print(
            f"💾 Saved FlyNet weights to: {path}"
        )

    def load_weights(self, path):
        self.weights = np.load(
            path
        ).astype(np.float32)

        print(
            f"🧠 Loaded FlyNet weights from: {path}"
        )
