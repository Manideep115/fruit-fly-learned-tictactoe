import numpy as np
from fly_interface import FlyInterface


POSITIONS = [
    {
        "name": "WIN top",
        "board": [1, 1, 0, -1, 0, 0, 0, -1, 0],
        "player": 1,
    },
    {
        "name": "BLOCK top",
        "board": [-1, -1, 0, 1, 0, 0, 1, 0, 0],
        "player": 1,
    },
    {
        "name": "WIN left",
        "board": [1, -1, 0, 1, -1, 0, 0, 0, 0],
        "player": 1,
    },
    {
        "name": "BLOCK middle",
        "board": [1, 0, 0, -1, -1, 0, 1, 0, 0],
        "player": 1,
    },
]


def main():
    fly = FlyInterface(seed=42)

    policy = np.load("data/flynet/policy.npz")
    fly.policy_weights = policy["weights"]
    fly.policy_bias = policy["bias"]

    for test in POSITIONS:
        scores, activity = fly.get_scores(
            test["board"],
            test["player"],
        )

        print("\n" + "=" * 60)
        print(test["name"])
        print("Board:", test["board"])

        print("\nFly MBON activity:")
        print(np.round(activity, 3))

        print("\nPolicy scores:")
        print(np.round(scores, 3))

        print("\nChosen move:")
        print(np.argmax(scores))


if __name__ == "__main__":
    main()