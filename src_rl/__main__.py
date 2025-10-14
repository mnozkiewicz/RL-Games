from games.gui import GameGui


def main() -> None:
    gui = GameGui(pixel_size=600, board_size=20, frame_rate=120)
    gui.run_game()


if __name__ == "__main__":
    main()
