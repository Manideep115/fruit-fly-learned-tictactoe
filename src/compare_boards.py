import numpy as np

from fly_interface import FlyInterface


BOARDS = [
    (
        "Top-row threat",
        [1, 1, 0,
         -1, 0, 0,
         0, -1, 0],
        1,
    ),
    (
        "Bottom-row threat",
        [1, -1, 0,
         1, -1, 0,
         0, 0, 0],
        1,
    ),
    (
        "Center occupied",
        [0, 1, -1,
         0, 1, 0,
         -1, 0, 0],
        -1,
    ),
    (
        "Mostly empty",
        [0, 0, 0,
         0, 1, 0,
         0, 0, 0],
        -1,
    ),
]


def main():
    fly = FlyInterface(seed=42)

    print("🪰 Board → FlyNet → MBON patterns")
    print("=" * 70)

    for name, board, player in BOARDS:

        scores, activity = fly.get_scores(
            board,
            player,
        )

        print(f"\n{name}")
        print(f"Board: {board}")

        print(
            "MBON:",
            np.round(activity, 2)
        )


if __name__ == "__main__":
    main()