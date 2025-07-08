from typing import List
from .piece import Piece
from ..position import Position
from ..color import Color


class Rook(Piece):
    def get_possible_moves(self, board) -> List[Position]:
        moves = []
        directions = [
            (0, 1),   # Up
            (0, -1),  # Down
            (1, 0),   # Right
            (-1, 0)   # Left
        ]
        
        current_x, current_y = self.position.position_to_tuple()
        
        for dx, dy in directions:
            for distance in range(1, 8):
                new_x = current_x + (dx * distance)
                new_y = current_y + (dy * distance)
                
                if not (0 <= new_x <= 7 and 0 <= new_y <= 7):
                    break
                
                target_position = Position(chr(ord('a') + new_x), new_y + 1)
                piece_at_target = board.get_piece_at(target_position)
                
                if piece_at_target is None:
                    moves.append(target_position)
                elif piece_at_target.color != self.color:
                    moves.append(target_position)
                    break
                else:
                    break
        
        return moves
    
    def get_symbol(self) -> str:
        return "♜" if self.color == Color.BLACK else "♖" 