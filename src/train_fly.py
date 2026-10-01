import random

from fly_interface import FlyInterface
from tictactoe import TicTacToe


def random_move(board):
    legal_moves = [
        i
        for i, value in enumerate(board)
        if value == 0
    ]

    if not legal_moves:
        return None

    return random.choice(legal_moves)


def play_game(fly, epsilon, fly_player):
    """
    Fly plays against a random opponent.

    The fly alternates between X and O across games.
    """

    game = TicTacToe()

    player = 1

    history = []

    while not game.done:

        # -----------------------------------------
        # Fly's turn
        # -----------------------------------------

        if player == fly_player:

            (
                move,
                eligibility,
                output_neuron,
                neuron_trace,
            ) = fly.choose_move(
                game.board,
                player,
                epsilon=epsilon,
                return_learning_data=True,
            )

            if move is None:
                raise RuntimeError(
                    "Fly could not choose a move."
                )

            history.append((move, eligibility, neuron_trace))

        # -----------------------------------------
        # Random opponent
        # -----------------------------------------

        else:

            move = random_move(
                game.board
            )

        _, reward, done = game.make_move(
            move,
            player,
        )

        if done:

            if reward == 1:

                if player == fly_player:
                    return 1, history
                else:
                    return -1, history

            # Draw
            return 0, history

        player *= -1

    return 0, history


def train(games=500, seed=42):

    print(
        "🪰 Teaching FlyNet against a random opponent"
    )

    print("=" * 60)

    random.seed(seed)

    fly = FlyInterface(seed=seed)

    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }

    for game_number in range(
        1,
        games + 1,
    ):

        # Alternate who goes first.
        fly_player = (
            1
            if game_number % 2 == 1
            else -1
        )

        # Exploration decreases over time.
        epsilon = max(
            0.05,
            0.30
            * (
                1
                - game_number / games
            ),
        )

        result, history = play_game(
            fly,
            epsilon,
            fly_player,
        )

        if result == 1:
            results["wins"] += 1

        elif result == -1:
            results["losses"] += 1

        else:
            results["draws"] += 1

        # -----------------------------------------
        # Reward the FlyNet's previous decisions
        # -----------------------------------------

        if result != 0:

            # Later decisions get more credit.
            gamma = 0.90

            for step, (action, eligibility, neuron_trace) in enumerate(
                reversed(history)
            ):

                reward = (
                    result
                    * (gamma ** step)
                )

                fly.brain.learn(
                    eligibility,
                    reward,
                    action,
                    neuron_trace=neuron_trace,
                )
        if game_number % 50 == 0:

            total = (
                results["wins"]
                + results["losses"]
                + results["draws"]
            )

            print(
                f"Game {game_number:4d} | "
                f"Win: "
                f"{results['wins'] / total:.1%} | "
                f"Loss: "
                f"{results['losses'] / total:.1%} | "
                f"Draw: "
                f"{results['draws'] / total:.1%} | "
                f"ε: {epsilon:.3f}"
            )

    print()
    print("=" * 60)
    print("🪰 Training finished")
    print("=" * 60)

    print(
        f"Wins : {results['wins']}"
    )

    print(
        f"Losses: {results['losses']}"
    )

    print(
        f"Draws : {results['draws']}"
    )

    fly.brain.save_weights(
        "data/flynet/flynet_learned_weights.npy"
    )


if __name__ == "__main__":
    train(500)
