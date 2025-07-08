from typing import List
from .piece import Piece
from ..position import Position
from ..color import Color


class Pawn(Piece):
    def get_possible_moves(self, board) -> List[Position]:
        moves = []
        current_x, current_y = self.position.position_to_tuple()
        
        # white moves up (+1), black moves down (-1)
        direction = 1 if self.color == Color.WHITE else -1
        starting_rank = 1 if self.color == Color.WHITE else 6
        
        new_y = current_y + direction
        if 0 <= new_y <= 7:
            forward_position = Position(chr(ord('a') + current_x), new_y + 1)
            if board.get_piece_at(forward_position) is None:
                moves.append(forward_position)
                
                # Double move from starting position
                if current_y == starting_rank:
                    double_move_y = current_y + (2 * direction)
                    if 0 <= double_move_y <= 7:
                        double_position = Position(chr(ord('a') + current_x), double_move_y + 1)
                        if board.get_piece_at(double_position) is None:
                            moves.append(double_position)
        
        # Diagonal eating
        for capture_x in [current_x - 1, current_x + 1]:
            if 0 <= capture_x <= 7:
                capture_y = current_y + direction
                if 0 <= capture_y <= 7:
                    capture_position = Position(chr(ord('a') + capture_x), capture_y + 1)
                    piece_at_capture = board.get_piece_at(capture_position)
                    
                    if piece_at_capture is not None and piece_at_capture.color != self.color:
                        moves.append(capture_position)
        
        return moves
    # buna da bak
    def can_en_passant(self, target_position: Position, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        target_x, target_y = target_position.position_to_tuple()
        
        expected_rank = 4 if self.color == Color.WHITE else 3
        if current_y != expected_rank:
            return False
            
        if abs(target_x - current_x) != 1 or target_y != current_y + (1 if self.color == Color.WHITE else -1):
            return False
            
        # Check if there's an enemy pawn next to us that can be captured en passant
        adjacent_position = Position(chr(ord('a') + target_x), current_y + 1)
        adjacent_piece = board.get_piece_at(adjacent_position)
        
        if (adjacent_piece is not None and 
            adjacent_piece.__class__.__name__ == 'Pawn' and 
            adjacent_piece.color != self.color):
            return True
            
        return False
    
    def get_symbol(self) -> str:
        return "♟" if self.color == Color.BLACK else "♙" 