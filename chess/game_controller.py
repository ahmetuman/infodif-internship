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


class GameController:
    def __init__(self):
        self.board = Board()
        self.board.setup_initial_position()
        
        self.current_player = Color(Color.WHITE)
        self.game_state = GameState(board=self.board, current_player=self.current_player)
        
        self.algebraic_notation = None
        self.error = None

    def is_valid_move(self, move_input):
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

    def run_game(self):
        while self.game_state.game_result == None:
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"Current player is {self.game_state.current_player}")
            if self.error:
                print(self.error)
            else:
                print(f"Last move is ({self.game_state.current_player.opposite()}): {self.algebraic_notation}")

            if self.game_state.is_check(color=self.game_state.current_player):
                print("Check")
            
            print(self.board)
            
            move_input = input("")
            if not self.is_valid_move(move_input):
                self.error = "Invalid input"
                continue

            try: 
                from_position = Position(move_input[0], int(move_input[1]))
                to_position = Position(move_input[2], int(move_input[3]))
                piece_to_play = self.game_state.board.get_piece_at(from_position)
                
                captured_piece = self.game_state.board.get_piece_at(to_position)

                #for i in (piece_to_play.get_possible_moves(board=board)):
                #    print(i, end=' ')

                if isinstance(piece_to_play, Pawn):
                    if piece_to_play.can_en_passant(target_position=to_position, board=self.board):
                        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, is_en_passant_move=True)

                    elif piece_to_play.can_promotion(target_position=to_position, board=self.board):
                        promotion_piece_type = input("Promotion: ")
                        if promotion_piece_type not in ['Q', 'R', 'B', 'N']:
                            promotion_piece_type = 'Q'  
                            move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, captured_piece=captured_piece, promotion_piece_type=promotion_piece_type)

                    else:
                        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, captured_piece=captured_piece)

                elif isinstance(piece_to_play, King):
                    file_difference = abs(ord(to_position.file) - ord(from_position.file))
                
                    if file_difference > 1 and piece_to_play.can_castle(board=self.board):
                        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, is_castling_move=True)
                    else:
                        move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, captured_piece=captured_piece)

                else:
                    move = Move(from_position=from_position, to_position=to_position, piece=piece_to_play, captured_piece=captured_piece)

            except Exception as e:
                self.error = e


            try:
                self.algebraic_notation = move.to_algebraic_notation()
                self.game_state.make_move(move=move)
                self.error = None
            except Exception as e:
                self.error = e
            
            if self.game_state.is_draw():
                print("Draw")
                print(self.board)
                break
            elif self.game_state.is_checkmate(color=self.game_state.current_player):
                print(f"Checkmate! {self.game_state.current_player.opposite()} wins!")
                print(self.board)
                break
