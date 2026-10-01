class TicTacToe:

    def __init__(self):
        self.reset()

    def reset(self):
        # 0 = empty, 1 = X, -1 = O
        self.board = [0] * 9
        self.done = False
        return self.board.copy()

    def legal_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, position, player):
        if self.done:
            return self.board.copy(), 0, True

        if position not in self.legal_moves():
            return self.board.copy(), -1, False

        self.board[position] = player

        if self._has_won(player):
            self.done = True
            return self.board.copy(), 1, True

        if not self.legal_moves():
            self.done = True
            return self.board.copy(), 0, True

        return self.board.copy(), 0, False

    def _has_won(self, player):
        wins = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),
            (0, 4, 8),
            (2, 4, 6),
        ]

        return any(
            self.board[a] == player
            and self.board[b] == player
            and self.board[c] == player
            for a, b, c in wins
        )

    def display(self):
        symbols = {
            1: "X",
            -1: "O",
            0: " "
        }

        for row in range(3):
            cells = self.board[row * 3:(row + 1) * 3]
            print(
                f" {symbols[cells[0]]} |"
                f" {symbols[cells[1]]} |"
                f" {symbols[cells[2]]} "
            )

            if row < 2:
                print("---+---+---")


if __name__ == "__main__":

    game = TicTacToe()

    print("🎮 Tic-Tac-Toe")
    print("Positions:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")

    player = 1

    while not game.done:

        print()
        game.display()

        move = int(input(f"\nPlayer {'X' if player == 1 else 'O'}, move: "))

        _, reward, done = game.make_move(move, player)

        if reward == -1:
            print("❌ Invalid move!")
            continue

        if done:
            print()
            game.display()

            if reward == 1:
                print(f"\n🎉 Player {'X' if player == 1 else 'O'} wins!")
            else:
                print("\n🤝 Draw!")

            break

        player *= -1