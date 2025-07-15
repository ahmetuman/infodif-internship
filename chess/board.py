from typing import Optional, List, Dict
from .position import Position
from .color import Color
from .pieces.piece import Piece
from .pieces.rook import Rook
from .pieces.knight import Knight
from .pieces.bishop import Bishop
from .pieces.queen import Queen
from .pieces.king import King
from .pieces.pawn import Pawn

class Board:
    def __init__(self):
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.piece_positions: Dict[Position, Piece] = {}

    def _position_to_indices(self, position: Position) -> tuple[int, int]: # Same with position_to_tuple at position.py
        file_index = ord(position.file.lower()) - ord('a')
        rank_index = position.rank - 1 
        return rank_index, file_index

    def get_piece_at(self, position: Position) -> Optional[Piece]:
        rank_idx, file_idx = self._position_to_indices(position)
        return self.board[rank_idx][file_idx]

    def place_piece(self, piece: Piece, position: Position):
        rank_idx, file_idx = self._position_to_indices(position)
        
        # If the square is full, remove existing one first then place the new one.
        if piece.position is not None:
            self.remove_piece(piece.position)
        
        self.board[rank_idx][file_idx] = piece
        self.piece_positions[position] = piece
        piece.position = position

    def remove_piece(self, position: Position) -> Optional[Piece]:
        rank_idx, file_idx = self._position_to_indices(position)
        piece = self.board[rank_idx][file_idx]
        
        if piece is not None:
            self.board[rank_idx][file_idx] = None
            if position in self.piece_positions:
                del self.piece_positions[position]
            piece.position = None
        else:
            raise ValueError(f"The square is already empty.")
        
        return piece

    def move_piece(self, from_position: Position, to_position: Position) -> bool:
        # possible moves ekle
        piece = self.get_piece_at(from_position)
        if piece is None:
            raise ValueError(f"There is no piece at specificed square.")
        
        self.remove_piece(from_position)
        piece.has_moved = True  
        self.place_piece(piece, to_position)
        return True

    def setup_initial_position(self):
        # Clear the board first in case of reinitialization
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.piece_positions = {}

        for file in 'abcdefgh':
            self.place_piece(Pawn(Color.WHITE, None), Position(file, 2))
        
        for file in 'abcdefgh': 
            self.place_piece(Pawn(Color.BLACK, None), Position(file, 7))
        
        self.place_piece(Rook(Color.WHITE, None), Position('a', 1))
        self.place_piece(Knight(Color.WHITE, None), Position('b', 1))
        self.place_piece(Bishop(Color.WHITE, None), Position('c', 1))
        self.place_piece(Queen(Color.WHITE, None), Position('d', 1))
        self.place_piece(King(Color.WHITE, None), Position('e', 1))
        self.place_piece(Bishop(Color.WHITE, None), Position('f', 1))
        self.place_piece(Knight(Color.WHITE, None), Position('g', 1))
        self.place_piece(Rook(Color.WHITE, None), Position('h', 1))
        
        self.place_piece(Rook(Color.BLACK, None), Position('a', 8))
        self.place_piece(Knight(Color.BLACK, None), Position('b', 8))
        self.place_piece(Bishop(Color.BLACK, None), Position('c', 8))
        self.place_piece(Queen(Color.BLACK, None), Position('d', 8))
        self.place_piece(King(Color.BLACK, None), Position('e', 8))
        self.place_piece(Bishop(Color.BLACK, None), Position('f', 8))
        self.place_piece(Knight(Color.BLACK, None), Position('g', 8))
        self.place_piece(Rook(Color.BLACK, None), Position('h', 8))

    def get_pieces_by_color(self, color: Color) -> List[Piece]: # for game logic script
        pieces = []
        for rank in range(8):
            for file in range(8):
                piece = self.board[rank][file]
                if piece is not None and piece.color == color:
                    pieces.append(piece)

        return pieces

    def is_square_attacked(self, position: Position, by_color: Color) -> bool:
        attacking_pieces = self.get_pieces_by_color(by_color)
        
        for piece in attacking_pieces:
            if piece.position is not None:
                if piece.__class__.__name__ == 'King':
                    possible_moves = piece.get_possible_moves(self, include_castling=False)
                else:
                    possible_moves = piece.get_possible_moves(self)
                if position in possible_moves:
                    return True
        
        return False

    def copy(self) -> 'Board':
        new_board = Board()
        
        for rank in range(8):
            for file in range(8):
                piece = self.board[rank][file]
                if piece is not None:
                    copied_piece = piece.copy()
                    file_char = chr(ord('a') + file)
                    rank_num = rank + 1
                    position = Position(file_char, rank_num)
                    
                    # WARNING: Directly set the board positions instead of using place_piece()
                    new_board.board[rank][file] = copied_piece
                    new_board.piece_positions[position] = copied_piece
                    copied_piece.position = position
        
        return new_board

    def __str__(self) -> str:
        lines = []
        
        for rank in range(7, -1, -1):
            rank_num = rank + 1
            line = f"{rank_num}  "
            
            for file in range(8):
                piece = self.board[rank][file]
                if piece is not None:
                    line += piece.get_symbol() + " "
                else:
                    line += ". "
            
            lines.append(line)
        
        lines.append("   a b c d e f g h")
        
        return "\n".join(lines)
    
    def setup_test_position(self):
        # Clear the board first in case of reinitialization
        self.board = [[None for _ in range(8)] for _ in range(8)]
        self.piece_positions = {}

        self.place_piece(Pawn(Color.WHITE, None), Position('b', 7))
        self.place_piece(Pawn(Color.WHITE, None), Position('g', 5))
        self.place_piece(King(Color.WHITE, None), Position('e', 1))
        self.place_piece(Rook(Color.WHITE, None), Position('a', 1))
        self.place_piece(Rook(Color.WHITE, None), Position('h', 1))


        self.place_piece(King(Color.BLACK, None), Position('c', 6))
        self.place_piece(Bishop(Color.BLACK, None), Position('c', 8))
        self.place_piece(Rook(Color.BLACK, None), Position('d', 8))
        self.place_piece(Pawn(Color.BLACK, None), Position('f', 7))

