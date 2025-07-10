from typing import Optional
from position import Position
from pieces.piece import Piece


class Move:
    def __init__(
        self, 
        from_position: Position, 
        to_position: Position, 
        piece: Piece, 
        captured_piece: Optional[Piece] = None,
        is_castling_move: bool = False,
        is_en_passant_move: bool = False,
        promotion_piece_type: Optional[str] = None
    ):
        self.from_position = from_position
        self.to_position = to_position
        self.piece = piece
        self.captured_piece = captured_piece
        self.is_castling_move = is_castling_move
        self.is_en_passant_move = is_en_passant_move
        self.promotion_piece_type = promotion_piece_type
        
        self._validate_move()
    
    def _validate_move(self):
        if self.from_position == self.to_position:
            raise ValueError("Same position")
        
        if self.is_castling_move and self.piece.__class__.__name__ != 'King':
            raise ValueError("Only kings can perform castling")
        
        if self.is_en_passant_move and self.piece.__class__.__name__ != 'Pawn':
            raise ValueError("Only pawns can perform en passant")
        
        if self.promotion_piece_type and self.piece.__class__.__name__ != 'Pawn':
            raise ValueError("Only pawns can be promoted")
    
    def is_capture(self) -> bool:
        return self.captured_piece is not None or self.is_en_passant_move
    
    def is_castling(self) -> bool:
        return self.is_castling_move
    
    def is_en_passant(self) -> bool:
        return self.is_en_passant_move
    
    def is_promotion(self) -> bool:
        return self.promotion_piece_type is not None
    
    def to_algebraic_notation(self) -> str:
        notation = ""
        
        if self.is_castling():
            return self._get_castling_notation() # and break
        
        # Piece prefix (except for pawns)
        if self.piece.__class__.__name__ != 'Pawn':
            notation += self._get_piece_symbol()
        
        # Capture notation (optional)
        if self.is_capture():
            if self.piece.__class__.__name__ == 'Pawn':
                notation += self.from_position.file
            notation += "x"
        
        # Destination square
        notation += str(self.to_position)
        
        # Promotion (only for pawns)
        if self.is_promotion():
            notation += f"={self.promotion_piece_type}"
        
        # En passant (only for pawns)
        if self.is_en_passant():
            notation += " e.p."
        
        return notation
    
    def _get_piece_symbol(self) -> str:
        piece_symbols = {
            'King': 'K',
            'Queen': 'Q', 
            'Rook': 'R',
            'Bishop': 'B',
            'Knight': 'N', # K de olabilir.
            'Pawn': ''
        }
        return piece_symbols.get(self.piece.__class__.__name__, '')
    
    def _get_castling_notation(self) -> str:
        if self.to_position.file == 'g':
            return "O-O"  # Short castling
        elif self.to_position.file == 'c':
            return "O-O-O"  # Long castling
    
    def __str__(self) -> str:
        return self.to_algebraic_notation()
    
    def __repr__(self) -> str:
        return f"Move({self.from_position} -> {self.to_position}, {self.piece.__class__.__name__})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Move):
            return False
        return (
            self.from_position == other.from_position and
            self.to_position == other.to_position and
            self.piece == other.piece
        ) 