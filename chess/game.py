import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chess.board import Board
from chess.position import Position
from chess.color import Color
from chess.game_state import GameState
from chess.move import Move

from chess.pieces.piece import Piece
from chess.pieces.rook import Rook
from chess.pieces.knight import Knight
from chess.pieces.bishop import Bishop
from chess.pieces.queen import Queen
from chess.pieces.king import King
from chess.pieces.pawn import Pawn

board = Board()
board.setup_initial_position()


current_player = Color(Color.WHITE)
game_state = GameState(board=board, current_player=current_player)

algebraic_notation = None
error = None

def is_valid_move(move_input):
    if len(move_input) != 4:
        return False

    start_col = move_input[0]
    start_row = move_input[1]
    end_col = move_input[2]
    end_row = move_input[3]

    if start_col not in 'abcdefgh':
        return False
    if end_col not in 'abcdefgh':
        return False
    if not (start_row.isdigit() and end_row.isdigit()):
        return False
    if not (1 <= int(start_row) <= 8 and 1 <= int(end_row) <= 8):
        return False

    return True


while game_state.game_result == None:
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"Current player is {game_state.current_player}")
    if error:
        print(error)
    else:
        print(f"Last move is ({game_state.current_player.opposite()}): {algebraic_notation}")
    if game_state.is_draw():
        print("Draw")
        break
    elif game_state.is_checkmate(color=game_state.current_player.opposite()):
        print(f"Checkmate! {game_state.current_player.opposite()} wins!")
        break
    elif game_state.is_check(color=game_state.current_player):
        print("Check")
    
    print(board)
    
    move_input = input("")
    if not is_valid_move(move_input):
        error = "Invalid input"
        continue

    try: 
        from_position = Position(move_input[0], int(move_input[1]))
        to_position = Position(move_input[2], int(move_input[3]))
        piece_to_play = game_state.board.get_piece_at(from_position)
    except Exception as e:
        error = e

    for i in (piece_to_play.get_possible_moves(board=board)):
        print(i, end=' ')
    if isinstance(piece_to_play, Pawn):
        if piece_to_play.can_en_passant(target_position=to_position, board=board):
            move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, is_en_passant_move=True)
            algebraic_notation = move.to_algebraic_notation()

        elif piece_to_play.can_promotion(target_position=to_position, board=board):
            promotion_piece_type = input("Promotion: ")
            if promotion_piece_type not in ['Q', 'R', 'B', 'N']:
                promotion_piece_type = 'Q'  
            move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, promotion_piece_type=promotion_piece_type)
            algebraic_notation = move.to_algebraic_notation()

        else:
            move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play)
            algebraic_notation = move.to_algebraic_notation()


    elif isinstance(piece_to_play, King) and piece_to_play.can_castle(board=board):
        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, is_castling_move=True)
        algebraic_notation = move.to_algebraic_notation()

    else:
        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play)
        algebraic_notation = move.to_algebraic_notation()


    try:
        game_state.make_move(move=move)
        error = None
    except Exception as e:
        error = e

