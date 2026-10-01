import json
import numpy as np

from fly_brain import FlyBrain


class FlyInterface:

    def __init__(
        self,
        interface_file="data/flynet/interface.json",
        calibration_file="data/flynet/output_calibration.json",
        seed=42,
    ):
        self.rng = np.random.default_rng(seed)

        self.brain = FlyBrain(seed=seed)

        with open(
            interface_file,
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        self.x_inputs = np.array(
            [
                self.brain.id_to_index[int(x)]
                for x in config["x_inputs"]
            ],
            dtype=np.int32,
        )

        self.o_inputs = np.array(
            [
                self.brain.id_to_index[int(x)]
                for x in config["o_inputs"]
            ],
            dtype=np.int32,
        )

        self.output_neurons = np.array(
            [
                self.brain.id_to_index[int(x)]
                for x in config["output_neurons"]
            ],
            dtype=np.int32,
        )

        with open(
            calibration_file,
            "r",
            encoding="utf-8",
        ) as f:
            calibration = json.load(f)

        self.output_mean = np.array(
            calibration["mean"],
            dtype=np.float32,
        )

        self.output_scale = np.array(
            calibration["scale"],
            dtype=np.float32,
        )

        print(
            "🧬 Using biological FlyWire interface"
        )

        print(
            f"X input neurons : "
            f"{len(self.x_inputs)}"
        )

        print(
            f"O input neurons : "
            f"{len(self.o_inputs)}"
        )

        print(
            f"Output neurons  : "
            f"{len(self.output_neurons)}"
        )

        print(
            "Output readout  : calibrated MBON z-score"
        )

    def board_to_input(
        self,
        board,
        player,
    ):
        """
        18-channel representation:

        Current player's position -> corresponding X neuron
        Opponent's position       -> corresponding O neuron
        """

        signal = np.zeros(
            self.brain.num_neurons,
            dtype=np.float32,
        )

        for position, value in enumerate(
            board
        ):

            if value == player:
                signal[
                    self.x_inputs[position]
                ] = 1.0

            elif value == -player:
                signal[
                    self.o_inputs[position]
                ] = 1.0

        return signal

    def choose_move(
        self,
        board,
        player,
        epsilon=0.0,
        return_learning_data=False,
    ):
        """
        The FlyNet itself produces the action scores.

        No trainable external policy exists.
        """

        input_signal = self.board_to_input(
            board,
            player,
        )

        (
            activity,
            eligibility,
            potential,
            neuron_trace,
        ) = self.brain.run(
            input_signal,
            steps=20,
        )

        # Combine spikes and membrane potential.
        output_activity = (
            activity[
                self.output_neurons
            ]
            + potential[
                self.output_neurons
            ]
        )

        # Unbiased centering across 9 MBON outputs
        output_activity = (
            output_activity - np.mean(output_activity)
        )

        legal_moves = [
            i
            for i, value in enumerate(board)
            if value == 0
        ]

        if not legal_moves:
            return None

        # Exploration during training.
        if (
            epsilon > 0
            and self.rng.random() < epsilon
        ):
            move = int(
                self.rng.choice(
                    legal_moves
                )
            )
        else:
            scores = np.full(
                9,
                -np.inf,
                dtype=np.float32,
            )

            scores[legal_moves] = (
                output_activity[
                    legal_moves
                ]
            )

            move = int(
                np.argmax(scores)
            )

        output_neuron = int(
            self.output_neurons[move]
        )

        if return_learning_data:
            return (
                move,
                eligibility,
                output_neuron,
                neuron_trace,
            )

        return move
