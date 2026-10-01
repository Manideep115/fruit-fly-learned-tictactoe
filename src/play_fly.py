from fly_interface import FlyInterface
from tictactoe import TicTacToe


def play_game(fly, starting_player):

    game = TicTacToe()

    player = starting_player

    while not game.done:

        move = fly.choose_move(
            game.board,
            player,
        )

        if move is None:
            raise RuntimeError(
                "Fly could not find a legal move."
            )

        _, reward, done = game.make_move(
            move,
            player,
        )

        if reward == -1:
            raise RuntimeError(
                f"Illegal move selected: {move}"
            )

        if done:

            if reward == 1:
                return player

            return 0

        player *= -1


def main():

    print("🪰 FlyNet self-play test")
    print("=" * 40)

    fly = FlyInterface()

    results = {
        1: 0,
        -1: 0,
        0: 0,
    }

    games = 10

    for game_number in range(1, games + 1):

        # Alternate starting player
        starting_player = (
            1 if game_number % 2 == 1
            else -1
        )

        winner = play_game(
            fly,
            starting_player,
        )

        results[winner] += 1

        if winner == 1:
            result = "X wins"
        elif winner == -1:
            result = "O wins"
        else:
            result = "Draw"

        print(
            f"Game {game_number:2d}: "
            f"{result} "
            f"(started as "
            f"{'X' if starting_player == 1 else 'O'})"
        )

    print("\nResults")
    print("-" * 40)
    print(f"X wins : {results[1]}")
    print(f"O wins : {results[-1]}")
    print(f"Draws  : {results[0]}")


if __name__ == "__main__":
    main()