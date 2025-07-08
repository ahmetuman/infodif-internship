from abc import ABC, abstractmethod
from typing import List
from copy import deepcopy
from ..position import Position
from ..color import Color


class Piece(ABC):
    def __init__(self, color: Color, position: Position):
        self.color = color
        self.position = position
        self.has_moved = False
    
    @abstractmethod
    def get_possible_moves(self, board) -> List[Position]:
        pass
    
    @abstractmethod
    def get_symbol(self) -> str:
        pass
    
    def is_valid_move(self, target: Position, board) -> bool:
        possible_moves = self.get_possible_moves(board)
        return target in possible_moves
    
    def copy(self):
        copied_piece = deepcopy(self)
        return copied_piece
    
    def move_to(self, new_position: Position):
        self.position = new_position
        self.has_moved = True 