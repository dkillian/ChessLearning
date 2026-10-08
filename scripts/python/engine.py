"""
engine.py -- Stage 01: Chess engine core components

Components:
    - Feature extraction  (board -> named feature vector)
    - Linear evaluator    (feature vector -> position score)
    - Negamax search      (alpha-beta pruning -> best move)

All features are computed from the perspective of the side to move
(positive = good for side to move). This lets the negamax search
simply negate scores at each ply.

Feature count: 14 learnable positional features (material is hard-coded).
Phase-scaled features: mobility (midgame), king_safety (midgame),
                       pawn_advancement (endgame).
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
    "doubled_pawn",     # penalty — negated so positive weight = good
    "isolated_pawn",    # penalty — same convention
    "backward_pawn",    # penalty — same convention
    # King safety
    "king_safety",      # safe squares adjacent to king; phase-scaled (midgame)
    # Activity
    "center_control",   # attacks + occupation of d4/d5/e4/e5
    "rook_open_file",
    "connected_rooks",
    # Piece coordination & development
    "knight_pst",       # piece-square table score for knights
    "bishop_pair",
    "rook_seventh",
    "piece_development",
    "mobility",         # squares attacked by non-king pieces; phase-scaled (midgame)
    "pawn_advancement", # most advanced pawn rank; phase-scaled (endgame)
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

# Knight piece-square table (White's perspective; rank 0 = White's back rank).
# Values in pawn fractions: rim squares ≈ −0.50, central outposts ≈ +0.20.
# Mirrored vertically for Black.
_KNIGHT_PST = [
    -0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50,  # rank 0
    -0.40, -0.20,  0.00,  0.00,  0.00,  0.00, -0.20, -0.40,  # rank 1
    -0.30,  0.00,  0.10,  0.15,  0.15,  0.10,  0.00, -0.30,  # rank 2
    -0.30,  0.05,  0.15,  0.20,  0.20,  0.15,  0.05, -0.30,  # rank 3
    -0.30,  0.00,  0.15,  0.20,  0.20,  0.15,  0.00, -0.30,  # rank 4
    -0.30,  0.05,  0.10,  0.15,  0.15,  0.10,  0.05, -0.30,  # rank 5
    -0.40, -0.20,  0.00,  0.05,  0.05,  0.00, -0.20, -0.40,  # rank 6
    -0.50, -0.40, -0.30, -0.30, -0.30, -0.30, -0.40, -0.50,  # rank 7
]


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


def _count_backward_pawns(board: chess.Board, color: chess.Color) -> int:
    """
    Pawns that cannot advance safely and have no friendly pawn support.
    A pawn is backward if:
      1. The square immediately in front is attacked by an opponent pawn.
      2. No friendly pawn on an adjacent file is at the same rank or behind.
    """
    our_pawns = board.pieces(chess.PAWN, color)
    opp = not color
    pawn_positions = [
        (chess.square_file(sq), chess.square_rank(sq)) for sq in our_pawns
    ]
    count = 0
    for sq in our_pawns:
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        front_rank = rank + 1 if color == chess.WHITE else rank - 1
        if not (0 <= front_rank <= 7):
            continue
        front_sq = chess.square(file, front_rank)
        if not board.is_attacked_by(opp, front_sq):
            continue  # square ahead is safe — not backward
        supported = False
        for adj_file in (file - 1, file + 1):
            if not (0 <= adj_file <= 7):
                continue
            for f, r in pawn_positions:
                if f != adj_file:
                    continue
                if color == chess.WHITE and r <= rank:
                    supported = True
                    break
                if color == chess.BLACK and r >= rank:
                    supported = True
                    break
            if supported:
                break
        if not supported:
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
    """
    Attacks on d4/d5/e4/e5 by any piece, plus occupation of those squares
    by friendly pawns. Max = 8 (4 attacks + 4 pawn occupations).
    """
    center = (chess.D4, chess.D5, chess.E4, chess.E5)
    attacks = sum(1 for sq in center if board.is_attacked_by(color, sq))
    occupation = sum(
        1 for sq in center
        if (p := board.piece_at(sq)) is not None
        and p.piece_type == chess.PAWN
        and p.color == color
    )
    return attacks + occupation


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


def _knight_pst_score(board: chess.Board, color: chess.Color) -> float:
    """
    Piece-square table score summed over all knights of the given color.
    Uses _KNIGHT_PST indexed from rank 0 (White's back rank); mirrored
    vertically for Black so both sides use the same positional incentives.
    """
    total = 0.0
    for sq in board.pieces(chess.KNIGHT, color):
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        idx = (rank * 8 + file) if color == chess.WHITE else ((7 - rank) * 8 + file)
        total += _KNIGHT_PST[idx]
    return total


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
    return sum(
        1 for sq in board.pieces(chess.ROOK, color)
        if chess.square_rank(sq) == target_rank
    )


def _mobility(board: chess.Board, color: chess.Color) -> int:
    """Number of squares attacked by all non-king pieces."""
    return sum(
        len(board.attacks(sq))
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for sq in board.pieces(pt, color)
    )


def _pawn_advancement(board: chess.Board, color: chess.Color) -> int:
    """
    Rank advancement of the single most advanced pawn from its starting rank.
    White starting rank = 1; Black starting rank = 6.
    Returns 0 if no pawns remain. Max = 5 (one step before promotion square).
    """
    pawns = board.pieces(chess.PAWN, color)
    if not pawns:
        return 0
    if color == chess.WHITE:
        return max(chess.square_rank(sq) - 1 for sq in pawns)
    else:
        return max(6 - chess.square_rank(sq) for sq in pawns)


def _game_phase(board: chess.Board) -> float:
    """
    Midgame fraction ∈ [0.0, 1.0].
    1.0 = full midgame (all pieces on board); 0.0 = pure endgame.
    Weighted piece count: Q=4, R=2, N=1, B=1. Full midgame baseline = 24.
    """
    n = (
        (len(board.pieces(chess.QUEEN,  chess.WHITE)) +
         len(board.pieces(chess.QUEEN,  chess.BLACK))) * 4 +
        (len(board.pieces(chess.ROOK,   chess.WHITE)) +
         len(board.pieces(chess.ROOK,   chess.BLACK))) * 2 +
        (len(board.pieces(chess.KNIGHT, chess.WHITE)) +
         len(board.pieces(chess.KNIGHT, chess.BLACK))) +
        (len(board.pieces(chess.BISHOP, chess.WHITE)) +
         len(board.pieces(chess.BISHOP, chess.BLACK)))
    )
    return min(n / 24.0, 1.0)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(board: chess.Board) -> dict:
    """
    Return a dict mapping feature name -> value, from the perspective
    of the side to move. Positive values are favorable for the side to move.

    Conventions:
      - Penalty features (doubled_pawn, isolated_pawn, backward_pawn) are
        negated so a positive weight always means 'this is good for side to move'.
      - center_control normalized by /4 (raw max = 8 per side).
      - knight_pst normalized by /2 (raw max ≈ 0.4 per side with two knights).
      - mobility normalized by /30 (raw max ≈ 30 per side).
      - pawn_advancement normalized by /5 (raw max = 5 per side).

    Phase scaling:
      - king_safety   × phase          (diminishes as pieces leave the board)
      - mobility      × (0.5 + 0.5×phase) (midgame-weighted; ≥50% always)
      - pawn_advancement × (0.5 + 0.5×end) (endgame-weighted; ≥50% always)
    """
    us    = board.turn
    them  = not us
    phase = _game_phase(board)
    end   = 1.0 - phase

    features = {
        # Pawn structure
        "passed_pawn":    _count_passed_pawns(board, us)    - _count_passed_pawns(board, them),
        "doubled_pawn":  -(_count_doubled_pawns(board, us)  - _count_doubled_pawns(board, them)),
        "isolated_pawn": -(_count_isolated_pawns(board, us) - _count_isolated_pawns(board, them)),
        "backward_pawn": -(_count_backward_pawns(board, us) - _count_backward_pawns(board, them)),

        # King safety — phase-scaled: attacking pieces must be present to matter
        "king_safety":    (_king_safety_score(board, us) - _king_safety_score(board, them)) * phase,

        # Activity
        "center_control": (_center_control_score(board, us) - _center_control_score(board, them)) / 4.0,
        "rook_open_file": _count_rooks_open_file(board, us) - _count_rooks_open_file(board, them),
        "connected_rooks": int(_are_rooks_connected(board, us)) - int(_are_rooks_connected(board, them)),

        # Piece coordination & development
        "knight_pst":     (_knight_pst_score(board, us)  - _knight_pst_score(board, them))  / 2.0,
        "bishop_pair":    int(len(board.pieces(chess.BISHOP, us)) >= 2) - int(len(board.pieces(chess.BISHOP, them)) >= 2),
        "rook_seventh":   _count_rooks_seventh(board, us)   - _count_rooks_seventh(board, them),
        "piece_development": _piece_development(board, us)  - _piece_development(board, them),

        # Mobility — midgame-weighted: piece activity matters more with more pieces
        "mobility":       (_mobility(board, us) - _mobility(board, them)) / 30.0 * (0.5 + 0.5 * phase),

        # Pawn advancement — most advanced pawn; endgame-weighted
        "pawn_advancement": (_pawn_advancement(board, us) - _pawn_advancement(board, them)) / 5.0 * (0.5 + 0.5 * end),
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

    features   = extract_features(board)
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
            break  # beta cutoff

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

    best_move  = None
    best_score = float("-inf")
    alpha      = float("-inf")
    beta       = float("inf")

    for move in moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, weights)
        board.pop()

        if score > best_score:
            best_score = score
            best_move  = move
        if score > alpha:
            alpha = score

    return best_move, best_score
