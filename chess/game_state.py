from typing import Optional, List, Dict
from .position import Position
from .color import Color
from .board import Board
from .move import Move
from .pieces.piece import Piece
from .pieces.rook import Rook
from .pieces.knight import Knight
from .pieces.bishop import Bishop
from .pieces.queen import Queen
from .pieces.king import King
from .pieces.pawn import Pawn


class GameState:
    def __init__(self, board: Board, current_player: Color):
        self.board = board
        self.current_player = current_player
        self.move_history: List[Move] = []
        self.captured_pieces_history: List[Optional[Piece]] = []
        self.half_move_clock = 0  # For 50 move rule
        self.full_move_number = 1  # Increments after black's move
        self.position_history: List[str] = []  # For threefold repetition
        self.game_result: Optional[str] = None 
        
        # Store initial position for threefold repetition
        self._store_current_position()
    
    def is_check(self, color: Color) -> bool:
        king_position = self._find_king(color)
        if king_position is None:
            raise LookupError(f"There is no king :D")
        
        if color == Color.WHITE:
            opponent_color = Color.BLACK
        else:
            opponent_color = Color.WHITE

        return self.board.is_square_attacked(king_position, opponent_color)
    
    def _find_king(self, color: Color) -> Optional[Position]:
        pieces = self.board.get_pieces_by_color(color)
        for piece in pieces:
            if isinstance(piece, King):
                return piece.position
        raise LookupError(f"There is no king :D")
    
    def get_legal_moves(self, color: Color) -> List[Move]:
        legal_moves = []
        pieces = self.board.get_pieces_by_color(color)
        
        for piece in pieces:
            if piece.position is None:
                raise LookupError(f"{piece} has no position.")
                
            possible_moves = piece.get_possible_moves(self.board)
            
            for target_position in possible_moves:
                captured_piece = self.board.get_piece_at(target_position)
                move = Move(from_position=piece.position, to_position=target_position, piece=piece, captured_piece=captured_piece)
                
                if not self._is_king_under_check_after_this_move(move, color):
                    legal_moves.append(move)
        
        return legal_moves
    
    def _is_king_under_check_after_this_move(self, move: Move, color: Color) -> bool:
        board_copy = self.board.copy()
        self._execute_move_on_board(move, board_copy)
        
        king_position = None
        pieces = board_copy.get_pieces_by_color(color)
        for piece in pieces:
            if isinstance(piece, King):
                king_position = piece.position
                break
        
        if king_position is None:
            raise LookupError(f"King not found after move")
        
 

        opponent_color = color.opposite()     
        
        return board_copy.is_square_attacked(king_position, opponent_color)
    
    def _execute_move_on_board(self, move: Move, board: Board):
        if move.is_castling():
            self._handle_castling_on_board(move, board)
        elif move.is_en_passant():
            self._handle_en_passant_on_board(move, board)
        else:
            board.move_piece(move.from_position, move.to_position)
            
            if move.is_promotion():
                self._handle_promotion_on_board(move, board)
    
    def _handle_castling_on_board(self, move: Move, board: Board):
        board.move_piece(move.from_position, move.to_position)
        
        if move.to_position.file == 'g':  # Kingside castling
            rook_from = Position('h', move.from_position.rank)
            rook_to = Position('f', move.from_position.rank)
        else:  # Queenside castling
            rook_from = Position('a', move.from_position.rank)
            rook_to = Position('d', move.from_position.rank)
        
        board.move_piece(rook_from, rook_to)
    
    def _handle_en_passant_on_board(self, move: Move, board: Board):
        board.move_piece(move.from_position, move.to_position)
        
        captured_pawn_rank = move.from_position.rank  # Same rank as attacking pawn
        captured_pawn_position = Position(move.to_position.file, captured_pawn_rank)
        board.remove_piece(captured_pawn_position)
    
    def _handle_promotion_on_board(self, move: Move, board: Board):
        board.remove_piece(move.to_position)
        
        piece_classes = {'Q': Queen, 'R': Rook, 'B': Bishop, 'N': Knight}

        piece_class = piece_classes.get(move.promotion_piece_type, Queen)
        promoted_piece = piece_class(move.piece.color, None)
        
        board.place_piece(promoted_piece, move.to_position)

    def is_checkmate(self, color: Color) -> bool:
        if not self.is_check(color):
            return False
        
        legal_moves = self.get_legal_moves(color)
        return len(legal_moves) == 0
    
    def is_stalemate(self, color: Color) -> bool:
        if self.is_check(color):
            return False
        
        legal_moves = self.get_legal_moves(color)
        return len(legal_moves) == 0
    
    def is_draw(self) -> bool:
        return self._is_fifty_move_rule() or self._is_threefold_repetition()
    
    def _is_fifty_move_rule(self) -> bool:
        return self.half_move_clock >= 100 
    
    def _is_threefold_repetition(self) -> bool:
        current_position = self._get_position_string()
        count = self.position_history.count(current_position)
        return count >= 3
    
    def _get_position_string(self) -> str:
        board_str = str(self.board)
        return f"{board_str}|{self.current_player.name}"
    
    def _store_current_position(self):
        position_str = self._get_position_string()
        self.position_history.append(position_str)
    
    def make_move(self, move: Move) -> bool:
        if move.piece.color != self.current_player:
            raise ValueError(f"Current player is {self.current_player}, not {move.piece.color}!")
        
        legal_moves = self.get_legal_moves(self.current_player)
        if move not in legal_moves:
            raise ValueError(f"Illegal move!") 
        
        # TODO: Captured pieces or move history may be unnecessary, recheck if not used 
        captured_piece = None
        if move.is_capture():
            if move.is_en_passant():
                captured_piece = self._get_en_passant_captured_piece(move)
            else:
                captured_piece = self.board.get_piece_at(move.to_position)
        
        self._execute_move_on_board(move, self.board)
        
        self.move_history.append(move)
        self.captured_pieces_history.append(captured_piece)
        
        if move.piece.__class__.__name__ == 'Pawn' or move.is_capture():
            self.half_move_clock = 0  # Reset on pawn move or capture
        else:
            self.half_move_clock += 1
        
        if self.current_player == Color.BLACK:
            self.full_move_number += 1 # TODO: This is probably unnecessary too
        
        if self.current_player == Color.WHITE: # Use the opposite function from color class, i never used it for now
            self.current_player = Color.BLACK
        else: 
            self.current_player = Color.WHITE
        
        self._store_current_position()
        
        self.check_game_end() # This one may be added to the game engine (main function)
        
        return True
    
    def check_game_end(self):
        if self.is_checkmate(self.current_player):
            if self.current_player == Color.BLACK:
                winner = "White" 
            else:
                winner = "Black"
            self.game_result = f"Checkmate: {winner} wins"
        elif self.is_stalemate(self.current_player):
            self.game_result = "Stalemate: Draw"
        elif self.is_draw():
            if self._is_fifty_move_rule():
                self.game_result = "Draw: 50 move rule"
            elif self._is_threefold_repetition():
                self.game_result = "Draw: Threefold repetition"
    
    def _get_en_passant_captured_piece(self, move: Move) -> Optional[Piece]:
        captured_pawn_rank = move.from_position.rank
        captured_pawn_pos = Position(move.to_position.file, captured_pawn_rank)
        return self.board.get_piece_at(captured_pawn_pos)
 