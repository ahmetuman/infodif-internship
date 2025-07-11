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
        
        for i, j in directions:
            new_x = current_x + i
            new_y = current_y + j
            
            if 0 <= new_x <= 7 and 0 <= new_y <= 7:
                target_position = Position(chr(ord('a') + new_x), new_y + 1)
                piece_at_target = board.get_piece_at(target_position)
                
                if piece_at_target is None or piece_at_target.color != self.color:
                    moves.append(target_position)
        
        castling_moves = self._get_castling_moves(board)
        moves.extend(castling_moves)
        
        return moves
    
    def _get_castling_moves(self, board) -> List[Position]:
        moves = []
        
        if self.has_moved:
            return moves
            
        current_x, current_y = self.position.position_to_tuple()
        
        if self.color == Color.WHITE:
            expected_rank = 0
        else: 
            expected_rank = 7

        if current_y != expected_rank:
            return moves
        
        # O-O 
        if self._can_castle_kingside(board):
            castle_position = Position('g', current_y + 1)
            moves.append(castle_position)
            
        # O-O-O
        if self._can_castle_queenside(board):
            castle_position = Position('c', current_y + 1)
            moves.append(castle_position)
            
        return moves
    
    def _can_castle_kingside(self, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        
        rook_position = Position('h', current_y + 1)
        rook = board.get_piece_at(rook_position)
        if (rook is None or rook.__class__.__name__ != 'Rook' or rook.color != self.color or rook.has_moved):
            return False
            
        # Check between king and rook are empty
        for file in ['f', 'g']:
            check_position = Position(file, current_y + 1)
            if board.get_piece_at(check_position) is not None:
                return False
                
        # TODO: Check if king is in check or passes through check for impossible move
        return True
    
    def _can_castle_queenside(self, board) -> bool:
        current_x, current_y = self.position.position_to_tuple()
        
        rook_position = Position('a', current_y + 1)
        rook = board.get_piece_at(rook_position)
        if (rook is None or rook.__class__.__name__ != 'Rook' or rook.color != self.color or rook.has_moved):
            return False
        
        # Check between king and rook are empty
        for file in ['b', 'c', 'd']:
            check_position = Position(file, current_y + 1)
            if board.get_piece_at(check_position) is not None:
                return False
                
        # TODO: Check if king is in check or passes through check for impossible move
        return True
    
    def can_castle(self, board) -> tuple[bool, bool]:
        return (self._can_castle_kingside(board), self._can_castle_queenside(board))
    
    def get_symbol(self) -> str:
        if self.color == Color.BLACK:
            return "♚"
        else:
            return "♔"