from fly_interface import FlyInterface
from tictactoe import TicTacToe


WEIGHTS = "data/flynet/flynet_learned_weights.npy"


def print_board(board):
    symbols = {
        1: "X",
        -1: "O",
        0: " ",
    }

    print()

    for row in range(3):
        i = row * 3

        print(
            f" {symbols[board[i]]} |"
            f" {symbols[board[i + 1]]} |"
            f" {symbols[board[i + 2]]}"
        )

        if row < 2:
            print("---+---+---")

    print()


def main():
    print("🪰 TRAINED FLYNET vs YOU")
    print("=" * 45)

    # Load the FlyNet with the same biological architecture
    # used during training.
    fly = FlyInterface(seed=42)

    fly.brain.load_weights(WEIGHTS)

    print("🧠 Loaded trained FlyNet weights")
    print("You = X")
    print("Fly = O")

    print("\nPositions:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")

    game = TicTacToe()

    # Human always starts as X.
    player = 1

    while not game.done:

        print_board(game.board)

        # -----------------------------
        # Human
        # -----------------------------
        if player == 1:

            while True:
                try:
                    move = int(
                        input("Your move (0-8): ")
                    )
                except ValueError:
                    print("Enter a number from 0 to 8.")
                    continue

                if not 0 <= move <= 8:
                    print("Choose a position from 0 to 8.")
                    continue

                if game.board[move] != 0:
                    print("That position is occupied.")
                    continue

                break

        # -----------------------------
        # Fly
        # -----------------------------
        else:

            print("🪰 Fly is thinking...")

            move = fly.choose_move(
                game.board,
                player,
                epsilon=0.0,
            )

            print(
                f"🪰 Fly chooses position: {move}"
            )

        # -----------------------------
        # Apply move
        # -----------------------------

        _, reward, done = game.make_move(
            move,
            player,
        )

        if done:

            print_board(game.board)

            if reward == 1:

                if player == 1:
                    print("🎉 YOU WIN!")
                else:
                    print("🪰 FLY WINS!")

            else:
                print("🤝 DRAW!")

            break

        player *= -1


if __name__ == "__main__":
    main()