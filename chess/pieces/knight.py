from typing import List
from .piece import Piece
from ..position import Position
from ..color import Color


class Knight(Piece):
    def get_possible_moves(self, board) -> List[Position]:
        moves = []
        knight_moves = [
            (2, 1),   # Right 2, Up 1
            (2, -1),  # Right 2, Down 1
            (-2, 1),  # Left 2, Up 1
            (-2, -1), # Left 2, Down 1
            (1, 2),   # Right 1, Up 2
            (1, -2),  # Right 1, Down 2
            (-1, 2),  # Left 1, Up 2
            (-1, -2)  # Left 1, Down 2
        ]
        
        current_x, current_y = self.position.position_to_tuple()
        
        for dx, dy in knight_moves:
            new_x = current_x + dx
            new_y = current_y + dy
            
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                target_position = Position(chr(ord('a') + new_x), new_y + 1)
                piece_at_target = board.get_piece_at(target_position)
                
                if piece_at_target is None or piece_at_target.color != self.color:
                    moves.append(target_position)
        
        return moves
    
    def get_symbol(self) -> str:
        return "♞" if self.color == Color.BLACK else "♘" 