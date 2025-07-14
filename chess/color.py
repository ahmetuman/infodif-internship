from enum import Enum

class Color(Enum):
    WHITE = "white"
    BLACK = "black"
    
    def opposite(self):
        return Color.BLACK if self == Color.WHITE else Color.WHITE 
    
    def __str__(self):
        return "White" if self == Color.WHITE else "Black" 

        