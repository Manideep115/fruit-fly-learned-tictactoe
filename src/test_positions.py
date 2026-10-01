from fly_interface import FlyInterface


WEIGHTS = "data/flynet/flynet_learned_weights.npy"


POSITIONS = [
    {
        "name": "🟢 Immediate WIN — top row",
        "board": [
            1, 1, 0,
            -1, 0, 0,
            0, -1, 0,
        ],
        "player": 1,
        "expected": 2,
    },
    {
        "name": "🔴 Must BLOCK — top row",
        "board": [
            -1, -1, 0,
            1, 0, 0,
            1, 0, 0,
        ],
        "player": 1,
        "expected": 2,
    },
    {
        "name": "🟢 Immediate WIN — left column",
        "board": [
            1, -1, 0,
            1, -1, 0,
            0, 0, 0,
        ],
        "player": 1,
        "expected": 6,
    },
    {
        "name": "🔴 Must BLOCK — middle row",
        "board": [
            1, 0, 0,
            -1, -1, 0,
            1, 0, 0,
        ],
        "player": 1,
        "expected": 5,
    },
    {
        "name": "🟢 Immediate WIN — right column",
        "board": [
            1, -1, 0,
            0, -1, 1,
            0, 0, 1,
        ],
        "player": 1,
        "expected": 2,
    },
]


def main():

    print("🪰 FlyNet Tactical Test")
    print("=" * 60)

    fly = FlyInterface(seed=42)

    # Load the actual learned FlyNet weights.
    fly.brain.load_weights(WEIGHTS)

    passed = 0

    for i, test in enumerate(POSITIONS, 1):

        move = fly.choose_move(
            test["board"],
            test["player"],
            epsilon=0.0,
        )

        success = move == test["expected"]

        if success:
            passed += 1

        print()
        print(f"Test {i}: {test['name']}")
        print(f"Board   : {test['board']}")
        print(f"Expected: {test['expected']}")
        print(f"Fly     : {move}")
        print(
            f"Result  : "
            f"{'✅ PASS' if success else '❌ FAIL'}"
        )

    print()
    print("=" * 60)
    print(
        f"Score: {passed}/{len(POSITIONS)}"
    )


if __name__ == "__main__":
    main()