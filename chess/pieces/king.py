from typing import List
from .piece import Piece
from ..position import Position
from ..color import Color


class King(Piece):
    def get_possible_moves(self, board) -> List[Position]:
        moves = []
        directions = [
            (0, 1),   # Up
            (0, -1),  # Down
            (1, 0),   # Right
            (-1, 0),  # Left
            (1, 1),   # Up-right
            (1, -1),  # Down-right
            (-1, 1),  # Up-left
            (-1, -1)  # Down-left
        ]
        
        current_x, current_y = self.position.position_to_tuple()
        
        # Normal king moves (1 square in any direction)
        for dx, dy in directions:
            new_x = current_x + dx
            new_y = current_y + dy
            
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                target_position = Position(chr(ord('a') + new_x), new_y + 1)
                piece_at_target = board.get_piece_at(target_position)
                
                if piece_at_target is None or piece_at_target.color != self.color:
                    moves.append(target_position)
        
        # Add castling moves if possible
        castling_moves = self._get_castling_moves(board)
        moves.extend(castling_moves)
        
        return moves
    
    # asagidakilere bir daha bak
    def _get_castling_moves(self, board) -> List[Position]:
        moves = []
        
        # Can only castle if king hasn't moved
        if self.has_moved:
            return moves
            
        current_x, current_y = self.position.position_to_tuple()
        
        # King must be on starting rank
        expected_rank = 0 if self.color == Color.WHITE else 7
        if current_y != expected_rank:
            return moves
        
        # Check kingside castling (O-O)
        if self._can_castle_kingside(board):
            castle_position = Position('g', current_y + 1)
            moves.append(castle_position)
            
        # Check queenside castling (O-O-O)
        if self._can_castle_queenside(board):
            castle_position = Position('c', current_y + 1)
            moves.append(castle_position)
            
        return moves
    
    def _can_castle_kingside(self, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        
        # Check if rook is in correct position and hasn't moved
        rook_position = Position('h', current_y + 1)
        rook = board.get_piece_at(rook_position)
        if (rook is None or 
            rook.__class__.__name__ != 'Rook' or 
            rook.color != self.color or 
            rook.has_moved):
            return False
            
        # Check if squares between king and rook are empty
        for file in ['f', 'g']:
            check_position = Position(file, current_y + 1)
            if board.get_piece_at(check_position) is not None:
                return False
                
        # TODO: Check if king is in check or passes through check
        return True
    
    def _can_castle_queenside(self, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        
        # Check if rook is in correct position and hasn't moved
        rook_position = Position('a', current_y + 1)
        rook = board.get_piece_at(rook_position)
        if (rook is None or 
            rook.__class__.__name__ != 'Rook' or 
            rook.color != self.color or 
            rook.has_moved):
            return False
            
        # Check if squares between king and rook are empty
        for file in ['b', 'c', 'd']:
            check_position = Position(file, current_y + 1)
            if board.get_piece_at(check_position) is not None:
                return False
                
        # TODO: Check if king is in check or passes through check
        return True
    
    def can_castle(self, board) -> tuple[bool, bool]:
        """Returns (can_castle_kingside, can_castle_queenside)"""
        return (self._can_castle_kingside(board), self._can_castle_queenside(board))
    
    def get_symbol(self) -> str:
        return "♚" if self.color == Color.BLACK else "♔" 