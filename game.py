"""Pygame front-end – glue board.py, ui.py, and ai.py together.

Features
---------
* Sidebar buttons: **Undo**, **Restart**, **Quit**
* Scroll-wheel blocked
* Winner detection with on-screen message
* Graceful message when there is nothing to undo
* AI opponent (Black) with depth‐10 minimax + patterns
"""

from __future__ import annotations

import time
import pygame as pg

from . import board
from .ui import Button, load_image, load_sound
from .ai import find_best_move

import tkinter as tk
from tkinter import simpledialog

# ---------------------------------------------------------------------------
# Constants & configuration
# ---------------------------------------------------------------------------
TILE        = 40   # px between grid lines
MARGIN      = 27   # px from edge of bg.png to first line

# Who plays what
HUMAN_COLOUR = board.WHITE
AI_COLOUR    = board.BLACK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def grid_to_px(r: int, c: int) -> tuple[int, int]:
    """Convert board coords → centre pixel."""
    return MARGIN + c * TILE, MARGIN + r * TILE

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    pg.init()
    pg.mixer.init()

    root = tk.Tk()
    root.withdraw()  # hide the main tkinter window

    mode = simpledialog.askstring("Game Mode", "Enter game mode:\n'1' for Human vs AI\n'2' for Human vs Human")
    if mode not in {"1", "2"}:
        print("Invalid mode selected. Defaulting to Human vs AI.")
        mode = "1"

    vs_ai = mode == "1"

    # dummy surface required before convert_alpha()
    pg.display.set_mode((1, 1), flags=pg.HIDDEN)

    # ------------------------------------------------------------------
    # Load assets
    # ------------------------------------------------------------------
    bg          = load_image("bg.png")
    stone_black = load_image("stone_black.png")
    stone_white = load_image("stone_white.png")
    click_snd   = load_sound("click.mp3")
    restart_snd = load_sound("restart.mp3")

    # ------------------------------------------------------------------
    # Window + sidebar
    # ------------------------------------------------------------------
    W, H = bg.get_size()
    screen = pg.display.set_mode((W + 200, H))      # extra sidebar
    pg.display.set_caption("Connect Five")

    btn_undo    = Button("Undo",    (W + 30, 120))
    btn_restart = Button("Restart", (W + 30, 200))
    btn_quit    = Button("Quit",    (W + 30, 280))

    # ------------------------------------------------------------------
    # Game state
    # ------------------------------------------------------------------
    grid: list[list[str]] = board.blank_grid()
    turn: str = AI_COLOUR                    # AI (Black) goes first
    history: list[tuple[int, int, str]] = []  # (row, col, colour)

    font = pg.font.SysFont("Times New Roman", 24)
    msg_text: str | None = None    # transient sidebar message
    msg_until: float      = 0.0    # display deadline (epoch secs)

    clock = pg.time.Clock()
    running = True
    while running:
        # --------------------------- events ---------------------------
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False

            # only handle left-clicks on human's turn
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                if vs_ai and turn != HUMAN_COLOUR:
                    continue

                mx, my = ev.pos

                # ----- sidebar actions -----
                if btn_undo.clicked(ev.pos):
                    if vs_ai:
                        if len(history) >= 2:
                            # Undo AI + Human moves
                            for _ in range(2):
                                r, c, prev = history.pop()
                                grid[r][c] = board.EMPTY
                            turn = HUMAN_COLOUR
                        else:
                            msg_text = "Nothing to undo"
                            msg_until = time.time() + 1.0
                    else:
                        if history:
                            r, c, prev = history.pop()
                            grid[r][c] = board.EMPTY
                            turn = prev
                        else:
                            msg_text = "Nothing to undo"
                            msg_until = time.time() + 1.0
                    continue

                if btn_restart.clicked(ev.pos):
                    restart_snd.play()
                    grid      = board.blank_grid()
                    history.clear()
                    turn      = AI_COLOUR
                    msg_text  = None
                    continue

                if btn_quit.clicked(ev.pos):
                    running = False
                    continue

                # ----- board clicks -----
                if board.winner(grid) is not None:
                    continue  # game over

                col = round((mx - MARGIN) / TILE)
                row = round((my - MARGIN) / TILE)
                if board.apply_move(grid, (row, col), turn):
                    click_snd.play()
                    history.append((row, col, turn))
                    if vs_ai:
                        turn = AI_COLOUR
                    else:
                        turn = board.BLACK if turn == board.WHITE else board.WHITE

        # --------------------------- draw -----------------------------
        screen.fill((30, 30, 30))
        screen.blit(bg, (0, 0))
        btn_undo.draw(screen)
        btn_restart.draw(screen)
        btn_quit.draw(screen)

        # stones
        for r in range(board.SIZE):
            for c in range(board.SIZE):
                clr = grid[r][c]
                if clr == board.EMPTY:
                    continue
                img = stone_black if clr == board.BLACK else stone_white
                screen.blit(img, img.get_rect(center=grid_to_px(r, c)))

        # status / winner
        winner = board.winner(grid)
        if winner is None:
            turn_msg = 'Black' if turn == board.BLACK else 'White'
            status = f"{turn_msg} to move"
        else:
            status = f"{'Black' if winner == board.BLACK else 'White'} wins!"
        txt = font.render(status, True, (255, 255, 255))
        screen.blit(txt, (W + 20, 40))

        # transient message
        if msg_text and time.time() < msg_until:
            hint = font.render(msg_text, True, (255, 255, 0))
            screen.blit(hint, (W + 20, 80))
        else:
            msg_text = None

        # --------------------------- AI move -------------------------
        if vs_ai and turn == AI_COLOUR and board.winner(grid) is None:
            if turn == AI_COLOUR and board.winner(grid) is None:
                # give AI more thinking time & deeper search
                move = find_best_move(
                    grid,
                    AI_COLOUR,
                    max_depth=2,    # search 10 plies deep
                    time_limit=3.0   # up to 3 seconds per move
                )
                if board.apply_move(grid, move, AI_COLOUR):
                    click_snd.play()
                    history.append((move[0], move[1], AI_COLOUR))
                    turn = HUMAN_COLOUR

        # --------------------------- finalize -----------------------
        pg.display.flip()
        clock.tick(60)  # cap FPS

    pg.quit()


# allow “python -m connect_five.game” entry point
if __name__ == "__main__":
    main()
