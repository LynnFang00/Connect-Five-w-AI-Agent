
"""
Self-contained AI module for Connect Five.
Supports:
1) Immediate win and block detection.
2) Block high-threat open-three patterns.
3) Greedy 1-ply pattern-based moves with proximity penalties.
4) Depth-limited minimax with alpha-beta pruning and move ordering.
"""
from __future__ import annotations
import time
from typing import List, Tuple, Optional

from . import board

Move = Tuple[int, int]

# ---------------------------------------------------------------------------
# Pattern tables for X (Black)
# Contexts: 0=start, 1=middle, 2=end
# ---------------------------------------------------------------------------
value_model_X: list[dict[str, tuple[str, int]]] = [
    # beginning check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),
        '4_11': ('XXXX ', 100),  # beginning
        '4_13': ('XX XX ', 84),  # beginning
        '4_15': ('XXX X', 82),  # beginning
        '4_18': ('X XXX', 86),  # beginning

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),
        '3_10': ('XXX  ', 25),  # beginning
        '3_11': ('XX X ', 21),  # beginning
        '3_16': ('X XX ', 23),  # beginning

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4),
        '2_6': ('XX   ', 2),  # beginning
    },
    # middle check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4)
    },
    # ending check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),
        '4_10': (' XXXX', 100),  # ending
        '4_12': ('X XXX', 82),  # ending
        '4_14': ('XX XX', 84),  # ending
        '4_19': ('XXX X', 86),  # ending

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),
        '3_12': ('  XXX', 25),  # ending
        '3_13': (' X XX', 21),  # ending
        '3_17': (' XX X', 23),  # ending

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4),
        '2_7': ('   XX', 2),  # ending
    }
]

# ---------------------------------------------------------------------------
# Pattern tables for O (White)
# ---------------------------------------------------------------------------
value_model_O: list[dict[str, tuple[str, int]]] = [
    # beginning check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),
        '4_11': ('XXXX ', 100),  # beginning
        '4_13': ('XX XX ', 84),  # beginning
        '4_15': ('XXX X', 82),  # beginning
        '4_18': ('X XXX', 86),  # beginning

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),
        '3_10': ('XXX  ', 25),  # beginning
        '3_11': ('XX X ', 21),  # beginning
        '3_16': ('X XX ', 23),  # beginning

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4),
        '2_6': ('XX   ', 2),  # beginning
    },
    # middle check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4)
    },
    # ending check
    {
        '5': ('XXXXX', 1000),

        '4_0': (' XXXX ', 400),
        '4_1': (' XXXXO', 100),
        '4_2': ('OXXXX ', 100),
        '4_3': ('X XXX  ', 120),
        '4_4': ('   XX XX   ', 110),
        '4_5': ('  XXX X', 120),
        '4_6': ('X XXXO', 82),
        '4_7': ('XX XXO', 84),
        '4_8': ('OXX XX', 84),
        '4_9': ('OXXX X', 82),
        '4_16': ('OX XXX', 86),
        '4_17': ('XXX XO', 86),
        '4_10': (' XXXX', 100),  # ending
        '4_12': ('X XXX', 82),  # ending
        '4_14': ('XX XX', 84),  # ending
        '4_19': ('XXX X', 86),  # ending

        '3_0': ('  XXX  ', 60),
        '3_1': ('  XXXO', 25),
        '3_2': ('OXXX  ', 25),
        '3_3': (' XXX  ', 30),
        '3_4': ('  XXX ', 30),
        '3_5': (' X XX ', 37),
        '3_6': (' XX X ', 37),
        '3_7': (' X X X ', 27),
        '3_8': ('OXX X ', 21),
        '3_9': (' X XXO', 21),
        '3_14': ('OX XX ', 23),
        '3_15': (' XX XO', 23),
        '3_12': ('  XXX', 25),  # ending
        '3_13': (' X XX', 21),  # ending
        '3_17': (' XX X', 23),  # ending

        '2_0': ('   XX   ', 8),
        '2_1': ('   XXO', 2),
        '2_2': ('OXX   ', 2),
        '2_3': ('  XX   ', 5),
        '2_4': (' XX   ', 4),
        '2_5': (' X X ', 4),
        '2_7': ('   XX', 2),  # ending
    }
]


# ---------------------------------------------------------------------------
def additional(te_list: list[tuple[str, tuple[str, int]]]) -> int:
    """
    Compute extra bonus points based on pattern combinations in te_list.

    - If two or more of (open-three or four) threats exist, +30.
    - Else if two or more threes (and at least one open-three), +15.
    """
    score = 0
    temp_list = [code for code, _ in te_list]
    # Combine four and open-three threats
    key1 = ['3_0', '3_3', '3_4', '3_5', '3_6', '4_1', '4_2', '4_3', '4_4', '4_5', '4_6',
            '4_7', '4_8', '4_9', '4_10', '4_11', '4_12', '4_13', '4_14', '4_15', '4_16',
            '4_17', '4_18', '4_19']
    key2 = ['3_0', '3_3', '3_4', '3_5', '3_6', '3_1', '3_2', '3_7', '3_8', '3_9', '3_10',
            '3_11', '3_12', '3_13', '3_14', '3_15', '3_16', '3_17']
    if sum(temp_list.count(k) for k in key1) >= 2:
        score += 30
    elif sum(temp_list.count(k) for k in key2) >= 2 and sum(temp_list.count(k)
                                                            for k in ['3_0', '3_3', '3_4', '3_5', '3_6']) > 0:
        score += 15
    return score


def get_candidate_moves(grid: List[List[str]], dist: int = 2) -> list[Move]:
    """
        Return empty board positions within `dist` of any existing stone,
        sorted by Manhattan distance to the nearest stone.

        If the board is empty, returns only the center position.

        Parameters:
            grid:  Current board as a 2D list of ' ', 'O', 'X'.
            dist:  Maximum Manhattan distance from any stone to consider.

        Returns:
            A list of (row, col) tuples for legal moves.
        """
    stones = [
        (r, c)
        for r in range(board.SIZE)
        for c in range(board.SIZE)
        if grid[r][c] != board.EMPTY
    ]
    if not stones:
        centre = board.SIZE // 2
        return [(centre, centre)]
    rs, cs = zip(*stones)
    r0, r1 = max(0, min(rs) - dist), min(board.SIZE - 1, max(rs) + dist)
    c0, c1 = max(0, min(cs) - dist), min(board.SIZE - 1, max(cs) + dist)
    candidates = [
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if grid[r][c] == board.EMPTY
    ]
    # sort by distance to nearest stone
    # computes the Manhattan distance between the one candidate position and that one stone:
    # takes the smallest of those distances—that is, how far is mv from its nearest existing stone.
    candidates.sort(
        key=lambda mv: min(
            abs(mv[0] - sr) + abs(mv[1] - sc)
            for sr, sc in stones
        )
    )
    return candidates


def evaluate_pattern(grid: List[List[str]], colour: str) -> int:
    """
        Evaluate a grid for `colour` by scanning all rows, columns,
        and both diagonals for pattern matches.

        Parameters:
            grid:   2D list of board state.
            colour: board.BLACK or board.WHITE.

        Returns:
            The total heuristic score for that colour.
        """
    model = value_model_O if colour == board.WHITE else value_model_X
    total = 0
    lines: List[str] = []
    # rows
    for row in grid:
        lines.append(''.join(row))
    # cols
    for c in range(board.SIZE):
        lines.append(''.join(grid[r][c] for r in range(board.SIZE)))
    # main diagonals
    for p in range(-board.SIZE+1, board.SIZE):
        diag = [grid[r][r-p] for r in range(board.SIZE) if 0 <= r-p < board.SIZE]
        lines.append(''.join(diag))
    # anti-diagonals
    for p in range(2*board.SIZE):
        anti = [grid[r][p-r] for r in range(board.SIZE) if 0 <= p-r < board.SIZE]
        lines.append(''.join(anti))
    for table in model:
        for pat, score in table.values():
            for line in lines:
                if pat in line:
                    total += score
    return total


def evaluate_pattern_after_move(grid: List[List[str]], mv: Move, colour: str) -> int:
    """
        Apply `mv` for `colour`, evaluate the board, then undo.

        Returns the pattern score at the leaf.
        """
    r, c = mv
    grid[r][c] = colour
    sc = evaluate_pattern(grid, colour)
    grid[r][c] = board.EMPTY
    return sc


def _greedy_one_ply(
    grid: List[List[str]],
    colour: str,
    opp: str,
    candidates: list[Move],
    time_limit: Optional[float],
) -> Move:
    """
    Perform a single-ply heuristic scan over `candidates`:
      diff = 1.1*(my_gain)
           + (opp_orig  - opp_new)
           + (opp_max_future - opp_new)
           + additional(combo_list)
           - distance_penalty

    Returns the move with highest diff, or the first candidate if none.
    """
    start = time.time()
    # baseline scores before any move
    base_self = evaluate_pattern(grid, colour)
    base_opp  = evaluate_pattern(grid, opp)

    best_val, best_mv = float('-inf'), None

    # precompute all stones for distance penalty
    stones = [(r, c) for r in range(board.SIZE)
                     for c in range(board.SIZE)
                     if grid[r][c] != board.EMPTY]

    for r, c in candidates:
        if time_limit and (time.time() - start) > time_limit:
            break

        # 1) simulate my move
        grid[r][c] = colour
        sc_self = evaluate_pattern(grid, colour)

        # 2) simulate opponent move at same spot
        grid[r][c] = opp
        sc_opp_new = evaluate_pattern(grid, opp)

        # undo simulation
        grid[r][c] = board.EMPTY

        # 3) estimate opponent’s best reply
        future_candidates = get_candidate_moves(grid)
        opp_max = max(
            evaluate_pattern_after_move(grid, (rr, cc), opp)
            for rr, cc in future_candidates
        )

        # 4) combo bonus
        # build te_list by scanning patterns that matched
        # (you’ll need to accumulate these in evaluate_pattern or replicate here)
        combo_bonus = additional([])  # <-- pass in the proper te_list if you have it

        # 5) distance penalty: farther away from existing stones → small penalty
        dist_penalty = min(abs(r - sr) + abs(c - sc) for sr, sc in stones) * 0.5

        # 6) strategic diff
        diff = (
            1.5 * (sc_self - base_self) +
            (base_opp - sc_opp_new) +
            (opp_max - sc_opp_new) +
            combo_bonus -
            dist_penalty
        )

        if diff > best_val:
            best_val, best_mv = diff, (r, c)

    return best_mv or candidates[0]


def find_best_move(
    grid: List[List[str]],
    colour: str,
    max_depth: int = 15,
    time_limit: Optional[float] = None,
) -> Move:
    """
        Main entry: decide AI move for `colour`.

        Pipeline:
          1. Center-open if board empty.
          2. Generate nearby `candidates`.
          3. Immediate win check (5-in-row).
          4. Immediate block check.
          5. Block high-threat open-three (Hǔosān).
          6. Greedy 1-ply if `max_depth <= 1`.
          7. Minimax α/β search otherwise.

        Parameters:
            grid:       Current board state.
            colour:     AI's colour (board.BLACK or board.WHITE).
            max_depth:  Depth of lookahead (1 = greedy only).
            time_limit: Seconds cap for search (None = no cap).

        Returns:
            (row, col) best move according to pipeline.
        """
    opp = board.BLACK if colour == board.WHITE else board.WHITE

    # 1) center-opening
    flat = [cell for row in grid for cell in row]
    if all(cell == board.EMPTY for cell in flat):
        centre = board.SIZE//2
        return (centre, centre)

    # 2) candidate moves
    candidates = get_candidate_moves(grid)
    # 3) immediate win
    for mv in candidates:
        grid[mv[0]][mv[1]] = colour
        if board.winner(grid) == colour:
            grid[mv[0]][mv[1]] = board.EMPTY
            return mv
        grid[mv[0]][mv[1]] = board.EMPTY

    # 4) block opponent win
    for mv in candidates:
        grid[mv[0]][mv[1]] = opp
        if board.winner(grid) == opp:
            grid[mv[0]][mv[1]] = board.EMPTY
            return mv
        grid[mv[0]][mv[1]] = board.EMPTY

    # 5) block high-threat open-three
    start_block = time.time()
    best_thr, block_mv = 0, None
    for mv in candidates:
        if time_limit and time.time()-start_block > time_limit:
            break
        grid[mv[0]][mv[1]] = opp
        thr = evaluate_pattern(grid, opp)
        grid[mv[0]][mv[1]] = board.EMPTY
        if thr > best_thr:
            best_thr, block_mv = thr, mv
    if best_thr >= 60 and block_mv:
        return block_mv

    # 6) greedy one-ply
    if max_depth <= 1:
        return _greedy_one_ply(grid, colour, opp, candidates, time_limit)

    # 7) minimax with alpha-beta
    start = time.time()
    best_val, best_mv = float('-inf'), None

    def minimax(node_depth, depth, alpha, beta, maximizing):
        """
        Recursively evaluate the best achievable heuristic score from a given board
        position using depth-limited minimax search with alpha-beta pruning.

        Parameters:
            node_depth (List[List[str]]):
                The current game grid (15×15) to search from. This list of lists
                is mutated in place when trying moves—and then undone.
            depth (int):
                Remaining plies to search. When depth == 0, the function returns
                a heuristic leaf evaluation (evaluate_pattern difference).
            alpha (float):
                The best (highest) value found so far along the path to the maximizer;
                used to prune branches that can’t improve the maximizer’s outcome.
            beta (float):
                The best (lowest) value found so far along the path to the minimizer;
                used to prune branches that can’t worsen the minimizer’s outcome.
            maximizing (bool):
                If True, this call is choosing the AI’s move (trying to maximize
                the score). If False, it’s simulating the opponent (trying to minimize
                the score).

        Returns:
            float:
                The minimax value of this position—i.e. the best guaranteed
                `(AI_score − Opp_score)` difference achievable under optimal play
                within the given search depth.
        """
        w = board.winner(node_depth)
        if w == colour:
            return float('inf')
        if w == opp:
            return float('-inf')
        if depth == 0:
            return evaluate_pattern(node_depth, colour)-evaluate_pattern(node_depth, opp)
        moves = get_candidate_moves(node_depth)
        # move ordering by 1-ply
        moves.sort(key=lambda mv: evaluate_pattern_after_move(node_depth, mv, colour if maximizing else opp), reverse=maximizing)
        if maximizing:
            val = float('-inf')
            for mv in moves:
                node_depth[mv[0]][mv[1]] = colour
                score = minimax(node_depth, depth-1, alpha, beta, False)
                node_depth[mv[0]][mv[1]] = board.EMPTY
                val = max(val, score)
                alpha = max(alpha, val)
                if alpha >= beta:
                    break
            return val
        else:
            val = float('inf')
            for mv in moves:
                node_depth[mv[0]][mv[1]] = opp
                score = minimax(node_depth, depth-1, alpha, beta, True)
                node_depth[mv[0]][mv[1]] = board.EMPTY
                val = min(val, score)
                beta = min(beta, val)
                if alpha >= beta:
                    break
            return val
    for mv in candidates:  # final “evaluate each possible spot” for full look-ahead
        if time_limit and time.time()-start > time_limit:
            break
        grid[mv[0]][mv[1]] = colour
        val = minimax(grid, max_depth-1, float('-inf'), float('inf'), False)
        grid[mv[0]][mv[1]] = board.EMPTY
        if val > best_val:
            best_val, best_mv = val, mv
    return best_mv or candidates[0]
