class Position:
    def __init__(self, file: str, rank: int):
        self.file = file.lower()
        self.rank = rank
        if not self.is_valid():
            raise ValueError(f"Invalid position: {file}{rank}")
    
    def is_valid(self) -> bool:
        if isinstance(self.file, str) and len(self.file) == 1 and 'a' <= self.file <= 'h':
            if isinstance(self.rank, int) and 1 <= self.rank <= 8:
                return True
        return False
    
    def position_to_tuple(self) -> tuple[int, int]: # Chess notation to matrix coordinates
        return (ord(self.file) - ord('a'), self.rank - 1)
    
    @classmethod
    def from_notation(cls, notation: str):
        if len(notation) != 2:
            raise ValueError(f"Invalid notation: {notation}")
        file = notation[0]
        rank = int(notation[1])
        return cls(file, rank)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Position):
            return False
        return self.file == other.file and self.rank == other.rank
    
    def __hash__(self) -> int: # To use it as a key in a dictionary
        return hash((self.file, self.rank))
    
    def __str__(self) -> str:
        return f"{self.file}{self.rank}"
    
#a = Position("A",3)
#print(a.position_to_tuple())
#print(a.from_notation(f"{a.file}{a.rank}"))