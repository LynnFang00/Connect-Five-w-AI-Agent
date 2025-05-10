"""Board‐level logic for a 15 × 15 “connect-five” game.

Everything here is deterministic, side-effect free, and easily unit-testable.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Iterable

EMPTY = " "
BLACK = "O"
WHITE = "X"
SIZE  = 15
Move  = Tuple[int, int]           # (row, col) 0-based


# ---------------------------------------------------------------------------

def blank_grid() -> List[List[str]]:
    """Return a fresh **SIZE×SIZE** grid filled with `EMPTY`."""
    return [[EMPTY] * SIZE for _ in range(SIZE)]


def inside(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE


DIRECTIONS: tuple[tuple[int, int], ...] = (
    (1, 0),   # vertical
    (0, 1),   # horizontal
    (1, 1),   # ↘ diagonal
    (1, -1),  # ↙ diagonal
)


def winner(grid: List[List[str]]) -> Optional[str]:
    """Return **BLACK** or **WHITE** if either has five in a row, else None."""
    for r in range(SIZE):
        for c in range(SIZE):
            colour = grid[r][c]
            if colour == EMPTY:
                continue

            for dr, dc in DIRECTIONS:
                if all(
                    inside(r + i * dr, c + i * dc)
                    and grid[r + i * dr][c + i * dc] == colour
                    for i in range(5)
                ):
                    return colour
    return None


def legal_moves(grid: List[List[str]]) -> Iterable[Move]:
    """Yield every empty square as *(row, col)*."""
    for r in range(SIZE):
        for c in range(SIZE):
            if grid[r][c] == EMPTY:
                yield (r, c)


def apply_move(grid: List[List[str]], move: Move, colour: str) -> bool:
    """Try to place *colour* at *move*.

    Returns ``True`` on success, ``False`` if that square is occupied.
    """
    r, c = move
    if not inside(r, c) or grid[r][c] != EMPTY:
        return False
    grid[r][c] = colour
    return True


def full(grid: List[List[str]]) -> bool:
    """True if no empties remain."""
    return all(cell != EMPTY for row in grid for cell in row)


# ---------------------------------------------------------------------------
# Minimal evaluation stub (placeholder for future AI)
# ---------------------------------------------------------------------------

def evaluate(grid: List[List[str]], colour: str) -> int:
    """Tiny heuristic: +∞ if *colour* wins, −∞ if opponent wins, else 0.

    (Replace with something smarter later.)
    """
    opp = BLACK if colour == WHITE else WHITE
    if winner(grid) == colour:
        return 10_000
    if winner(grid) == opp:
        return -10_000
    return 0
