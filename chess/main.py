from game_controller import GameController

if __name__ == "__main__":
    use_stockfish = input("Enter 1 for Player vs Player\nEnter 2 for Player vs Stockfish\n").strip()
    if use_stockfish == "1":
        use_stockfish = False  # No Stockfish
    elif use_stockfish == "2":
        use_stockfish = True   # Use Stockfish for Black
    stockfish_path = None
    
    if use_stockfish:
        stockfish_path = "stockfish/stockfish-macos-m1-apple-silicon"
        print("\nHuman vs Stockfish")
    else:
        print("\nHuman vs Human game")
    
    print()
    
    try:
        game_controller = GameController(
            use_stockfish_for_black=use_stockfish,
            stockfish_path=stockfish_path
        )
        game_controller.run_game()
    except Exception as e:
        print(e)
