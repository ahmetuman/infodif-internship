from typing import List
from .piece import Piece
from ..position import Position
from ..color import Color


class Pawn(Piece):
    def get_possible_moves(self, board) -> List[Position]:
        moves = []
        current_x, current_y = self.position.position_to_tuple()
        
        # white moves +1, black moves down -1
        if self.color == Color.WHITE:
            direction = 1
            starting_rank = 1
        else:
            direction = -1
            starting_rank = 6
                
        new_y = current_y + direction
        if 0 <= new_y <= 7:
            forward_square = Position(chr(ord('a') + current_x), new_y + 1)
            if board.get_piece_at(forward_square) is None:
                moves.append(forward_square)
                
                # Double move at starting
                if current_y == starting_rank:
                    double_move_y = current_y + (2 * direction)
                    double_forward_square = Position(chr(ord('a') + current_x), double_move_y + 1)
                    if board.get_piece_at(double_forward_square) is None:
                        moves.append(double_forward_square)
        
        # Diagonal capture
        for possible_capture_x in [current_x - 1, current_x + 1]:
            if 0 <= possible_capture_x <= 7:
                possible_capture_y = current_y + direction
                if 0 <= possible_capture_y <= 7:
                    capture_square = Position(chr(ord('a') + possible_capture_x), possible_capture_y + 1)
                    piece_at_capture = board.get_piece_at(capture_square)
                    
                    if piece_at_capture is not None and piece_at_capture.color != self.color:
                        moves.append(capture_square)
        
        return moves
    
    def can_en_passant(self, target_position: Position, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        target_x, target_y = target_position.position_to_tuple()
        
        if self.color == Color.WHITE:
            expected_rank = 4
        else:
            expected_rank = 3

        if current_y != expected_rank:
            return False
            
        if abs(target_x - current_x) != 1 or target_y != current_y + (1 if self.color == Color.WHITE else -1):
            return False
            
        adjacent_position = Position(chr(ord('a') + target_x), current_y + 1)
        adjacent_piece = board.get_piece_at(adjacent_position)
        
        if (adjacent_piece is not None and adjacent_piece.__class__.__name__ == 'Pawn' and adjacent_piece.color != self.color):
            return True
            
        return False
    
    def get_symbol(self) -> str:
        if self.color == Color.BLACK:
            return "♟"
        else:
            return "♙"