import sys
import os
import subprocess  
import time  
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


class StockfishBot:
    def __init__(self, stockfish_path, depth=15):
        self.depth = depth
        self.engine = subprocess.Popen(
            stockfish_path,
            universal_newlines=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1,
        )
        self._init_engine()

    def _init_engine(self):
        self._send("uci")
        self._read_until("uciok")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, command):
        self.engine.stdin.write(command + "\n")
        self.engine.stdin.flush()

    def _read_line(self):
        return self.engine.stdout.readline().strip()

    def _read_until(self, keyword):
        while True:
            line = self._read_line()
            if keyword in line:
                return line

    def get_best_move(self, moves_uci: list[str]):
        # Build position string
        move_str = " ".join(moves_uci)
        self._send(f"position startpos moves {move_str}")
        self._send(f"go depth {self.depth}")

        while True:
            line = self._read_line()
            if line.startswith("bestmove"):
                return line.split()[1]  

    def close(self):
        self._send("quit")
        self.engine.terminate()


class GameController:
    def __init__(self, stockfish_path, use_stockfish_for_black=False):
        self.board = Board()
        self.board.setup_initial_position()
        
        self.current_player = Color(Color.WHITE)
        self.game_state = GameState(board=self.board, current_player=self.current_player)
        
        self.algebraic_notation = None
        self.error = None
        
        self.use_stockfish_for_black = use_stockfish_for_black
        self.stockfish_bot = None
        if use_stockfish_for_black:
            try:
                self.stockfish_bot = StockfishBot(stockfish_path)
                print("Stockfish initialized")
            except Exception as e:
                print(f"Failed to initialize Stockfish: {e}")
                self.use_stockfish_for_black = False

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

    def get_stockfish_move(self):
        if not self.stockfish_bot:
            return None
        
        try:
            best_move_uci = self.stockfish_bot.get_best_move(self.game_state.move_history)
            return self.position_to_move(best_move_uci)
        except Exception as e:
            print(f"Error getting Stockfish move: {e}")
            return None

    def position_to_move(self, input_move):
        if len(input_move) < 4:
            return None
        
        from_position = Position(input_move[0], int(input_move[1]))
        to_position = Position(input_move[2], int(input_move[3]))
        
        piece = self.board.get_piece_at(from_position)
        if not piece:
            return None
        
        captured_piece = self.board.get_piece_at(to_position)
        
        if isinstance(piece, Pawn):
            if piece.can_en_passant(target_position=to_position, board=self.board):
                return Move(from_position=from_position, to_position=to_position, piece=piece, is_en_passant_move=True)
            
            if piece.can_promotion(target_position=to_position, board=self.board):
                promotion_piece = 'Q'  # Default Queen
                if len(input_move) == 5:
                    promotion_piece = input_move[4].upper()
                return Move(from_position=from_position, to_position=to_position, piece=piece, 
                          captured_piece=captured_piece, promotion_piece_type=promotion_piece)
        
        elif isinstance(piece, King):
            file_difference = abs(ord(to_position.file) - ord(from_position.file))
            if file_difference > 1 and piece.can_castle(board=self.board):
                return Move(from_position=from_position, to_position=to_position, piece=piece, is_castling_move=True)
        
        return Move(from_position=from_position, to_position=to_position, piece=piece, captured_piece=captured_piece)

    def execute_stockfish_move(self):
        if not self.use_stockfish_for_black or self.game_state.current_player != Color.BLACK:
            return False
        
        print("Stockfish is thinking...")
        stockfish_move = self.get_stockfish_move()
        
        if stockfish_move:
            try:
                self.algebraic_notation = stockfish_move.to_algebraic_notation()
                self.game_state.make_move(stockfish_move)
                print(f"Stockfish played: {self.algebraic_notation}")
                
                time.sleep(2)
                
                return True
            except Exception as e:
                print(f"Error executing Stockfish move: {e}")
                return False
        else:
            print("Stockfish could not find a move")
            return False

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
            
            # Check if Stockfish should play for black
            if self.use_stockfish_for_black and self.game_state.current_player == Color.BLACK:
                if self.execute_stockfish_move():
                    self.error = None
                else:
                    self.error = "Stockfish failed to make a move"
                continue
            
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
        
        # Clean up Stockfish engine if it was used
        if self.stockfish_bot:
            self.stockfish_bot.close()

    def close_stockfish(self):
        if self.stockfish_bot:
            self.stockfish_bot.close()
            self.stockfish_bot = None
