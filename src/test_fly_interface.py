from fly_interface import FlyInterface


fly = FlyInterface()

board = [
    1,  0, -1,
    0,  1,  0,
   -1,  0,  0,
]

player = 1  # X

print("Board:")
print(board)

move = fly.choose_move(
    board,
    player,
)

print()
print("🪰 Fly chose position:", move)