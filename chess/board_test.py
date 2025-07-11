import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess.board import Board
from chess.position import Position

def test_board():
    board = Board()
    board.setup_initial_position()
    print(board)

    move = board.move_piece(Position('e', 2), Position('e', 4))
    print(move)
    print(board)

    move = board.move_piece(Position('e', 8), Position('e', 5))
    print(move)
    print(board)
    
    #move = board.move_piece(Position('b', 8), Position('e', 5))
    #print(move)
    #print(board)

    whites = board.get_pieces_by_color(board.get_piece_at(Position('e', 4)).color)
    blacks = board.get_pieces_by_color(board.get_piece_at(Position('e', 5)).color)
    print(len(whites))
    print(len(blacks))

if __name__ == "__main__":
    test_board() 