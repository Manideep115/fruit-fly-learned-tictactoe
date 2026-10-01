import numpy as np

from fly_interface import FlyInterface


def main():

    fly = FlyInterface(seed=42)

    board = [
        1, 1, 0,
        -1, 0, 0,
        0, -1, 0,
    ]

    signal = fly.board_to_input(
        board,
        player=1,
    )

    print("Input neurons activated:")
    print(
        np.flatnonzero(signal)
    )

    # Run manually for more steps.
    fly.brain.reset()

    total_activity = np.zeros(
        fly.brain.num_neurons,
        dtype=np.float32,
    )

    for step in range(200):

        spikes = fly.brain.step(signal)

        total_activity += spikes

        if np.count_nonzero(spikes) > 0:

            print(
                f"Step {step:3d}: "
                f"{np.count_nonzero(spikes):4d} neurons fired"
            )

    print("\nTotal active neurons:")
    print(
        np.count_nonzero(total_activity)
    )

    print("\nTotal spikes:")
    print(
        int(total_activity.sum())
    )

    print("\nMBON activity:")
    print(
        total_activity[
            fly.output_neurons
        ]
    )


if __name__ == "__main__":
    main()