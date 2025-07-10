from chess import Position, Color, King, Queen, Rook, Bishop, Knight, Pawn, Move


def main():
    king = King(Color.BLACK, Position.from_notation("e1"))
    move = Move(Position.from_notation("e1"), Position.from_notation("e8"), king)
    print(move)
    print(king.color.value)
    print(king.get_symbol())
    
    queen = Queen(Color.WHITE, Position.from_notation("d1"))
    move = Move(Position.from_notation("d1"), Position.from_notation("d4"), queen)
    print(move)
    
    pawn = Pawn(Color.WHITE, Position.from_notation("e2"))
    move = Move(Position.from_notation("e2"), Position.from_notation("e4"), pawn)
    print(move)
    
    captured_piece = Pawn(Color.BLACK, Position.from_notation("d7"))
    move = Move(Position.from_notation("d1"), Position.from_notation("d7"), queen, captured_piece)
    print(move)    

    captured_pawn = Pawn(Color.BLACK, Position.from_notation("f6"))
    move = Move(Position.from_notation("e5"), Position.from_notation("f6"), pawn, captured_pawn)
    print(move)
    print(move.is_capture())
    
    move = Move(Position.from_notation("e1"), Position.from_notation("g1"), king, is_castling_move=False) # False ise rok notasyonu basmiyor.
    print(move)
    
    move = Move(Position.from_notation("e1"), Position.from_notation("c1"), king, is_castling_move=True)
    print(move)
    
    move = Move(Position.from_notation("e5"), Position.from_notation("f6"), pawn, is_en_passant_move=True)
    print(move)
    
    promoting_pawn = Pawn(Color.WHITE, Position.from_notation("e7"))
    move = Move(Position.from_notation("e7"), Position.from_notation("e8"), promoting_pawn, promotion_piece_type="Q")
    print(move)
    
    rook = Rook(Color.BLACK, Position.from_notation("f8"))
    move = Move(Position.from_notation("e7"), Position.from_notation("f8"), promoting_pawn, rook, promotion_piece_type="N")
    print(move)
    

if __name__ == "__main__":
    main() 