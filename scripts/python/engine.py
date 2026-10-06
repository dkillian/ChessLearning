"""
engine.py -- Stage 01: Chess engine core components

Components:
    - Feature extraction  (board -> named feature vector)
    - Linear evaluator    (feature vector -> position score)
    - Negamax search      (alpha-beta pruning -> best move)

All features are computed from the perspective of the side to move
(positive = good for side to move). This lets the negamax search
simply negate scores at each ply.
"""

import chess
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Pawn structure
    "passed_pawn",
    "doubled_pawn",     # penalty: positive weight = bad doubled pawns hurt
    "isolated_pawn",    # same convention
    # King safety
    "king_safety",      # safe squares adjacent to king
    # Activity
    "center_control",
    "rook_open_file",
    "connected_rooks",
    # Piece coordination & development
    "bishop_pair",
    "rook_seventh",
    "piece_development",
    "mobility",
    "pawn_advancement",
]

# Piece values hard-coded as prior knowledge — not learned.
# Scaled to the TD target range [-1, +1]: traditional values (1, 3, 3, 5, 9) ÷ 10.
PIECE_VALUES = {
    chess.PAWN:   0.1,
    chess.KNIGHT: 0.3,
    chess.BISHOP: 0.3,
    chess.ROOK:   0.5,
    chess.QUEEN:  0.9,
}


def default_weights(init: str = "zero") -> dict:
    """
    Return a weight dict for all learnable (positional) features.
    Material is no longer a learnable feature — it is hard-coded in evaluate().

    init='zero'      -- all weights start at 0
    init='random'    -- small random values to break symmetry
    init='canonical' -- alias for 'zero' (positional priors are zero)
    """
    if init in ("zero", "canonical"):
        return {name: 0.0 for name in FEATURE_NAMES}
    elif init == "random":
        return {name: random.uniform(-0.1, 0.1) for name in FEATURE_NAMES}
    raise ValueError(f"Unknown init mode: {init!r}")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _count_passed_pawns(board: chess.Board, color: chess.Color) -> int:
    """Pawns with no opposing pawns on the same or adjacent files ahead of them."""
    our_pawns = board.pieces(chess.PAWN, color)
    opp_pawns = [
        (chess.square_file(sq), chess.square_rank(sq))
        for sq in board.pieces(chess.PAWN, not color)
    ]
    count = 0
    for sq in our_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        adj = [f for f in (file - 1, file, file + 1) if 0 <= f <= 7]
        passed = True
        for opp_f, opp_r in opp_pawns:
            if opp_f in adj:
                if color == chess.WHITE and opp_r > rank:
                    passed = False
                    break
                elif color == chess.BLACK and opp_r < rank:
                    passed = False
                    break
        if passed:
            count += 1
    return count


def _count_doubled_pawns(board: chess.Board, color: chess.Color) -> int:
    """Extra pawns beyond one on any file (e.g. two pawns on e-file = 1 doubled)."""
    file_counts: dict[int, int] = {}
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        file_counts[f] = file_counts.get(f, 0) + 1
    return sum(max(0, n - 1) for n in file_counts.values())


def _count_isolated_pawns(board: chess.Board, color: chess.Color) -> int:
    """Pawns with no friendly pawns on adjacent files."""
    our_pawns = board.pieces(chess.PAWN, color)
    pawn_files = {chess.square_file(sq) for sq in our_pawns}
    count = 0
    for sq in our_pawns:
        f = chess.square_file(sq)
        if not ({f - 1, f + 1} & pawn_files):
            count += 1
    return count


def _king_safety_score(board: chess.Board, color: chess.Color) -> int:
    """Number of squares adjacent to king NOT attacked by the opponent."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    opp = not color
    return sum(
        1 for sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
        if not board.is_attacked_by(opp, sq)
    )


def _center_control_score(board: chess.Board, color: chess.Color) -> int:
    """Number of central squares (d4, d5, e4, e5) attacked by color."""
    center = (chess.D4, chess.D5, chess.E4, chess.E5)
    return sum(1 for sq in center if board.is_attacked_by(color, sq))


def _count_rooks_open_file(board: chess.Board, color: chess.Color) -> int:
    """Rooks on files containing no pawns of either color."""
    all_pawn_files = {
        chess.square_file(sq)
        for sq in board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    }
    return sum(
        1 for sq in board.pieces(chess.ROOK, color)
        if chess.square_file(sq) not in all_pawn_files
    )


def _are_rooks_connected(board: chess.Board, color: chess.Color) -> bool:
    """
    True if both rooks are on the same rank or file
    with no pieces standing between them.
    """
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return False
    r1, r2 = rooks[0], rooks[1]
    f1, rank1 = chess.square_file(r1), chess.square_rank(r1)
    f2, rank2 = chess.square_file(r2), chess.square_rank(r2)
    if f1 == f2:
        lo, hi = sorted((rank1, rank2))
        return all(board.piece_at(chess.square(f1, r)) is None for r in range(lo + 1, hi))
    if rank1 == rank2:
        lo, hi = sorted((f1, f2))
        return all(board.piece_at(chess.square(f, rank1)) is None for f in range(lo + 1, hi))
    return False


_WHITE_MINOR_STARTS = frozenset({chess.B1, chess.G1, chess.C1, chess.F1})
_BLACK_MINOR_STARTS = frozenset({chess.B8, chess.G8, chess.C8, chess.F8})


def _piece_development(board: chess.Board, color: chess.Color) -> int:
    """Minor pieces (knights and bishops) that have left their starting squares."""
    starts = _WHITE_MINOR_STARTS if color == chess.WHITE else _BLACK_MINOR_STARTS
    minors = board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color)
    return sum(1 for sq in minors if sq not in starts)


def _count_rooks_seventh(board: chess.Board, color: chess.Color) -> int:
    """Rooks on the 7th rank (rank index 6 for white, rank index 1 for black)."""
    target_rank = 6 if color == chess.WHITE else 1
    return sum(1 for sq in board.pieces(chess.ROOK, color) if chess.square_rank(sq) == target_rank)


def _mobility(board: chess.Board, color: chess.Color) -> int:
    """Number of squares attacked by all non-king pieces."""
    return sum(
        len(board.attacks(sq))
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for sq in board.pieces(pt, color)
    )


def _pawn_advancement(board: chess.Board, color: chess.Color) -> int:
    """Total rank advancement of all pawns from their starting rank."""
    total = 0
    for sq in board.pieces(chess.PAWN, color):
        rank = chess.square_rank(sq)
        total += (rank - 1) if color == chess.WHITE else (6 - rank)
    return total


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(board: chess.Board) -> dict:
    """
    Return a dict mapping feature name -> value, from the perspective
    of the side to move. Positive values are favorable for the side to move.

    Penalty features (doubled_pawn, isolated_pawn) are negated so that
    a positive weight always means 'this is good for the side to move.'
    """
    us = board.turn
    them = not us

    features = {
        # Pawn structure (passed = good; doubled/isolated = bad, so negate)
        "passed_pawn":   _count_passed_pawns(board, us)   - _count_passed_pawns(board, them),
        "doubled_pawn":  -(_count_doubled_pawns(board, us) - _count_doubled_pawns(board, them)),
        "isolated_pawn": -(_count_isolated_pawns(board, us)- _count_isolated_pawns(board, them)),

        # King safety and activity
        "king_safety":    _king_safety_score(board, us)    - _king_safety_score(board, them),
        "center_control": _center_control_score(board, us) - _center_control_score(board, them),
        "rook_open_file": _count_rooks_open_file(board, us)- _count_rooks_open_file(board, them),
        "connected_rooks":   int(_are_rooks_connected(board, us)) - int(_are_rooks_connected(board, them)),

        # Piece coordination & development
        "bishop_pair":       int(len(board.pieces(chess.BISHOP, us)) >= 2) - int(len(board.pieces(chess.BISHOP, them)) >= 2),
        "rook_seventh":      _count_rooks_seventh(board, us)  - _count_rooks_seventh(board, them),
        "piece_development": _piece_development(board, us)    - _piece_development(board, them),
        # Normalized: raw differences can reach ±30 and ±40 respectively, which creates
        # prediction errors far outside the [-1, +1] TD target range and causes weight explosion.
        # Dividing brings typical values to ±1, matching the scale of material features.
        "mobility":         (_mobility(board, us)          - _mobility(board, them))         / 30.0,
        "pawn_advancement": (_pawn_advancement(board, us)  - _pawn_advancement(board, them)) / 10.0,
    }

    return features


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board, weights: dict) -> float:
    """
    Evaluate a position from the perspective of the side to move.
    Returns a scalar score (positive = good for side to move).
    Terminal states return fixed values:
        checkmate: -10,000 (side to move has been mated)
        draw:       0
    """
    if board.is_checkmate():
        return -10_000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    us   = board.turn
    them = not us

    # Hard-coded material score (not learned)
    material = sum(
        value * (len(board.pieces(pt, us)) - len(board.pieces(pt, them)))
        for pt, value in PIECE_VALUES.items()
    )

    features = extract_features(board)
    positional = sum(weights.get(name, 0.0) * val for name, val in features.items())

    return material + positional


# ---------------------------------------------------------------------------
# Search: negamax with alpha-beta pruning
# ---------------------------------------------------------------------------

def negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    weights: dict,
) -> float:
    """
    Negamax search with alpha-beta pruning.
    Returns the score from the perspective of the side to move.
    """
    if board.is_game_over():
        return -10_000.0 if board.is_checkmate() else 0.0

    if depth == 0:
        return evaluate(board, weights)

    max_score = float("-inf")
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, weights)
        board.pop()

        if score > max_score:
            max_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  # Beta cutoff

    return max_score


def get_best_move(
    board: chess.Board,
    weights: dict,
    depth: int = 2,
    randomize: bool = True,
) -> tuple[Optional[chess.Move], float]:
    """
    Return (best_move, score) for the current side to move.
    
    randomize=True shuffles the move list before searching, so that
    ties are broken randomly rather than by move order. This prevents
    the engine from repeating the same games when weights are near zero.
    """
    moves = list(board.legal_moves)
    if not moves:
        return None, 0.0
    if randomize:
        random.shuffle(moves)

    best_move = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta = float("inf")

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, weights)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

    return best_move, best_score
