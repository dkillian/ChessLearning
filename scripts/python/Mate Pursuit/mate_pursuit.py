"""Mate-pursuit engine, version 1.

Uses only the rules of chess and the objective of checkmate. Each move:

1. Play an immediate checkmate if one exists.
2. Otherwise score each legal move by the fraction of opponent replies after
   which the engine still has a mate in 1 (a mate-in-2 threat). A score of 1.0
   is a forced mate in 2.
3. Play a highest-scoring threat, breaking ties at random. If no move creates
   a threat, play the move the evaluation function scores highest (or a random
   legal move if no evaluation function is supplied).

The search is rebuilt every move, so a pursuit lasts only while the threat
survives the opponent's replies.
"""

import random
from typing import Callable, Optional

import chess


def has_mate_in_1(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        mated = board.is_checkmate()
        board.pop()
        if mated:
            return True
    return False


def threat_score(board: chess.Board, move: chess.Move) -> float:
    """Fraction of opponent replies after which the mover still has mate in 1."""
    board.push(move)
    if board.is_game_over(claim_draw=True):
        board.pop()
        return 0.0

    replies = list(board.legal_moves)
    survived = 0
    for reply in replies:
        board.push(reply)
        if not board.is_game_over(claim_draw=True) and has_mate_in_1(board):
            survived += 1
        board.pop()

    board.pop()
    return survived / len(replies)


def choose_move(
    board: chess.Board,
    rng: random.Random,
    evaluate_fn: Optional[Callable[[chess.Board], float]] = None,
) -> chess.Move:
    """Pick a move; evaluate_fn scores positions for the side to move.

    When no threat exists, the fallback is the move whose resulting position
    scores best for the mover (evaluate_fn is called on the child position,
    so its score is negated). Without evaluate_fn, the fallback is random.
    """
    legal = list(board.legal_moves)

    for move in legal:
        board.push(move)
        mated = board.is_checkmate()
        board.pop()
        if mated:
            return move

    scored = [(threat_score(board, move), move) for move in legal]
    best = max(score for score, _ in scored)
    if best > 0:
        return rng.choice([move for score, move in scored if score == best])

    return greedy_move(board, rng, evaluate_fn)


def greedy_move(
    board: chess.Board,
    rng: random.Random,
    evaluate_fn: Optional[Callable[[chess.Board], float]] = None,
) -> chess.Move:
    """Play the move whose resulting position scores best for the mover.

    Without evaluate_fn, the move is chosen at random.
    """
    legal = list(board.legal_moves)
    if evaluate_fn is None:
        return rng.choice(legal)

    scored = []
    for move in legal:
        board.push(move)
        scored.append((-evaluate_fn(board), move))
        board.pop()
    best = max(value for value, _ in scored)
    return rng.choice([move for value, move in scored if value == best])


class _BudgetExceeded(Exception):
    pass


def _mate_line(board, mover, plies, counter, budget):
    """Optimistic search: every node, including the opponent's, takes any move that
    keeps a mate for the mover available within the remaining plies.

    Returns the list of UCI moves to the mate (mover's and opponent's moves), or None.
    """
    counter[0] += 1
    if counter[0] > budget:
        raise _BudgetExceeded()
    if board.is_checkmate():
        return [] if board.turn != mover else None
    if plies == 0 or board.is_game_over(claim_draw=True):
        return None

    moves = list(board.legal_moves)
    # Checks first: a cheap ordering that finds mates earlier
    moves.sort(key=lambda m: not board.gives_check(m))
    for move in moves:
        board.push(move)
        sub = _mate_line(board, mover, plies - 1, counter, budget)
        board.pop()
        if sub is not None:
            return [move.uci()] + sub
    return None


def find_mate(board, max_plies=7, node_budget=200000):
    """Shortest optimistic mate for the side to move, searched by iterative deepening.

    Returns (line, nodes) where line is the UCI move list or None, and nodes is the
    number of positions visited. Returns (None, nodes) when the budget runs out.
    """
    mover = board.turn
    work = board.copy()
    counter = [0]
    try:
        for plies in range(1, max_plies + 1, 2):
            line = _mate_line(work, mover, plies, counter, node_budget)
            if line is not None:
                return line, counter[0]
    except _BudgetExceeded:
        return None, counter[0]
    return None, counter[0]


def pursuit_choice(
    board: chess.Board,
    rng: random.Random,
    max_plies: int = 7,
    node_budget: int = 200000,
):
    """Return (move, mate_plies). mate_plies is 0 when the move is random.

    The search runs again every move. If Black's reply keeps the mate, the search
    still finds it at the shorter remaining depth, so the pursuit continues. If the
    reply stops it, the search finds a different mate or the engine plays randomly.
    """
    line, _ = find_mate(board, max_plies, node_budget)
    if line:
        return chess.Move.from_uci(line[0]), len(line)
    return rng.choice(list(board.legal_moves)), 0


def _forced(board, mover, plies, counter, budget):
    """Forced mate search: the mover's moves are OR nodes, Black's replies AND nodes."""
    counter[0] += 1
    if counter[0] > budget:
        raise _BudgetExceeded()
    if board.is_checkmate():
        return board.turn != mover
    if plies == 0 or board.is_game_over(claim_draw=True):
        return False

    if board.turn == mover:
        moves = list(board.legal_moves)
        moves.sort(key=lambda m: not board.gives_check(m))
        for move in moves:
            board.push(move)
            ok = _forced(board, mover, plies - 1, counter, budget)
            board.pop()
            if ok:
                return True
        return False

    for move in list(board.legal_moves):
        board.push(move)
        ok = _forced(board, mover, plies - 1, counter, budget)
        board.pop()
        if not ok:
            return False
    return True


def forced_mate(board: chess.Board, plies: int, node_budget: int = 5000) -> bool:
    """True if the side to move has a forced mate within plies (odd). A budget
    overrun counts as no mate found."""
    try:
        return _forced(board.copy(), board.turn, plies, [0], node_budget)
    except _BudgetExceeded:
        return False


def robust_choice(
    board: chess.Board,
    rng: random.Random,
    sub_plies: int = 3,
    sub_budget: int = 5000,
):
    """Return (move, score). score is the share of Black's replies after which the
    mover still has a forced mate within sub_plies; 0.0 means the move is random.

    A mate in 1 scores 1.0 at once. Ties are broken at random. The sub-searches use
    a node budget, so a reply that exhausts the budget counts as no mate found.
    """
    legal = list(board.legal_moves)
    legal.sort(key=lambda m: not board.gives_check(m))

    scored = []
    for move in legal:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move, 1.0
        if board.is_game_over(claim_draw=True):
            board.pop()
            continue

        replies = list(board.legal_moves)
        kept = 0
        for reply in replies:
            board.push(reply)
            if not board.is_game_over(claim_draw=True):
                if forced_mate(board, sub_plies, sub_budget):
                    kept += 1
            board.pop()
        board.pop()

        if replies:
            scored.append((kept / len(replies), move))

    if not scored:
        return rng.choice(legal), 0.0
    best = max(score for score, _ in scored)
    if best == 0:
        return rng.choice(legal), 0.0
    return rng.choice([move for score, move in scored if score == best]), best


def pursuit_move(
    board: chess.Board,
    rng: random.Random,
    max_plies: int = 7,
    node_budget: int = 200000,
) -> chess.Move:
    """Move-only wrapper around pursuit_choice."""
    move, _ = pursuit_choice(board, rng, max_plies, node_budget)
    return move


def root_mate_lines(board, max_plies=7, node_budget=200000):
    """Every shortest optimistic mate for the side to move, as UCI move lists.

    Each line starts with the mover's move and includes the opponent's replies.
    Returns [] if no mate is found within max_plies or the budget runs out.
    """
    mover = board.turn
    work = board.copy()
    for plies in range(1, max_plies + 1, 2):
        found = []
        for move in list(work.legal_moves):
            work.push(move)
            if work.is_checkmate():
                found.append([move.uci()])
            elif plies > 1 and not work.is_game_over(claim_draw=True):
                # Each first move gets its own budget, so one large subtree
                # cannot starve the others
                # A sub-search that runs out of budget leaves its board mid-search,
                # so it gets a copy and the root board stays intact
                try:
                    sub = _mate_line(work.copy(), mover, plies - 1, [0], node_budget)
                except _BudgetExceeded:
                    sub = None
                if sub is not None:
                    found.append([move.uci()] + sub)
            work.pop()
        if found:
            shortest = min(len(line) for line in found)
            return [line for line in found if len(line) == shortest]
    return []


def choose_pursuit(board, rng, state, max_plies=7, node_budget=200000):
    """Return (move, new_state, kind).

    state is the remaining known mate line, starting with the mover's move. If the
    next move in it is legal, play it. Otherwise search for the shortest mate lines,
    choose one at random, and commit to it. If no mate is found, play randomly.
    """
    if state:
        first = chess.Move.from_uci(state[0])
        if first in board.legal_moves:
            return first, list(state[1:]), f"pursuing known mate line, {len(state)} moves left"

    lines = root_mate_lines(board, max_plies, node_budget)
    if lines:
        line = rng.choice(lines)
        return (chess.Move.from_uci(line[0]), line[1:],
                f"new mate line, {len(line)} moves")
    return rng.choice(list(board.legal_moves)), [], "random, no mate found"


def advance_state(state, opponent_uci):
    """Update the known line after the opponent's move: keep it only if the
    opponent followed it."""
    if state and state[0] == opponent_uci:
        return list(state[1:])
    return []


def play_vs_random(
    engine_color: chess.Color = chess.WHITE,
    seed: int = 0,
    max_plies: int = 300,
) -> dict:
    """Play one game: the mate-pursuit engine against a uniformly random opponent."""
    rng = random.Random(seed)
    board = chess.Board()
    plies = 0

    while not board.is_game_over(claim_draw=True) and plies < max_plies:
        if board.turn == engine_color:
            move = choose_move(board, rng)
        else:
            move = rng.choice(list(board.legal_moves))
        board.push(move)
        plies += 1

    if board.is_checkmate():
        outcome = "engine" if board.turn != engine_color else "opponent"
    elif plies >= max_plies:
        outcome = "ply_cap"
    else:
        outcome = "draw"

    return {"outcome": outcome, "plies": plies}
