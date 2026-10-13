"""
engine_learnable_material.py -- Experimental condition: learnable material

Condition: LEARNABLE_MATERIAL
    Material evaluation is NOT hard-coded. Instead, piece count differences
    are included as learnable features alongside the 14 positional features,
    giving 19 learnable features in total. All weights are initialized at zero;
    the engine must discover piece values entirely from self-play experience.

    Research question: Can the engine learn correct piece values autonomously,
    and how does that affect the acquisition of positional concepts?

    First used in run_id=20.

Components:
    - Feature extraction  (board -> 19-element feature vector)
    - Linear evaluator    (fully learned: material + positional weights)
    - Negamax search      (alpha-beta pruning -> best move)

All features are computed from the perspective of the side to move
(positive = good for side to move). This lets the negamax search
simply negate scores at each ply.

Feature count: 19 learnable features (5 material + 14 positional).
Phase-scaled features: mobility (midgame), king_safety (midgame),
                       pawn_advancement (endgame).

Sign convention: all 19 features are oriented so that higher = better for
the side to move. Material features are (my_count - their_count), so a
positive weight is always correct. Penalty positional features
(doubled_pawn, isolated_pawn, backward_pawn) are negated in extract_features()
so positive weights are correct for those too.

Expected learned material weights (for reference):
    material_queen  ~ 0.9
    material_rook   ~ 0.5
    material_bishop ~ 0.3
    material_knight ~ 0.3
    material_pawn   ~ 0.1
"""

import chess
import random
from typing import Optional

# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Material (learnable — not hard-coded)
    "material_queen",
    "material_rook",
    "material_bishop",
    "material_knight",
    "material_pawn",
    # Pawn structure
    "passed_pawn",
    "doubled_pawn",     # penalty — negated so positive weight = good
    "isolated_pawn",    # penalty — same convention
    "backward_pawn",    # penalty — same convention
    # King safety
    "king_safety",      # pawn shelter − attacker count; phase-scaled (midgame)
    # Activity
    "center_control",   # attacks + occupation of d4/d5/e4/e5
    "rook_open_file",
    "connected_rooks",
    # Piece coordination & development
    "knight_outpost",   # safe knights in opponent's half not attacked by opponent pawn
    "bishop_pair",
    "rook_seventh",
    "piece_development",
    "mobility",         # squares attacked by non-king pieces; phase-scaled (midgame)
    "pawn_advancement", # total advancement of all pawns; phase-scaled (endgame)
]




def default_weights(init: str = "zero") -> dict:
    """
    Return a weight dict for all 19 learnable features (5 material + 14 positional).
    All weights start at zero — the engine must discover piece values from scratch.

    init='zero'   -- all weights start at 0 (recommended for this condition)
    init='random' -- small random values to break symmetry
    """
    if init == "zero":
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
        if not any(board.piece_type_at(s) == chess.PAWN
                   for s in board.attackers(opp, front_sq)):
            continue
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


def _king_safety_score(board: chess.Board, color: chess.Color) -> float:
    """
    King safety composite: pawn shelter minus attacker count.

    Pawn shelter: count of friendly pawns on the same or adjacent files
    within one or two ranks directly in front of the king.

    Attacker count: number of distinct opponent pieces attacking any square
    in the king zone (the 8 squares adjacent to the king).

    Returns shelter − attackers. Positive = safer; negative = exposed.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0
    opp       = not color
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    shelter = 0
    for sq in board.pieces(chess.PAWN, color):
        pf = chess.square_file(sq)
        pr = chess.square_rank(sq)
        if abs(pf - king_file) <= 1:
            if color == chess.WHITE and king_rank < pr <= king_rank + 2:
                shelter += 1
            elif color == chess.BLACK and king_rank - 2 <= pr < king_rank:
                shelter += 1

    king_zone: chess.SquareSet = chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])
    attacking_pieces = chess.SquareSet()
    for sq in king_zone:
        attacking_pieces |= board.attackers(opp, sq)
    attackers = len(attacking_pieces)

    return float(shelter - attackers)


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


def _safe_knight_outpost(board: chess.Board, color: chess.Color) -> int:
    """
    Count knights that are in the opponent's half of the board AND not attacked
    by any opponent pawn. Safe outpost knights are stable, not immediately
    dislodgeable — unlike PST-rewarded central knights that are frequently
    captured at depth=3 self-play, generating spurious negative TD signals.
    """
    opponent = not color

    opp_pawn_attacks = chess.BB_EMPTY
    for sq in board.pieces(chess.PAWN, opponent):
        opp_pawn_attacks |= chess.BB_PAWN_ATTACKS[opponent][sq]

    opp_half = (
        chess.BB_RANK_5 | chess.BB_RANK_6 | chess.BB_RANK_7 | chess.BB_RANK_8
        if color == chess.WHITE else
        chess.BB_RANK_1 | chess.BB_RANK_2 | chess.BB_RANK_3 | chess.BB_RANK_4
    )

    count = 0
    for sq in board.pieces(chess.KNIGHT, color):
        sq_bb = chess.BB_SQUARES[sq]
        if (sq_bb & opp_half) and not (sq_bb & opp_pawn_attacks):
            count += 1
    return count


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
    Total rank advancement of all pawns from their starting rank.
    Each pawn contributes its distance from its starting rank (0–5).
    White starting rank = 1; Black starting rank = 6.
    """
    pawns = board.pieces(chess.PAWN, color)
    if not pawns:
        return 0
    if color == chess.WHITE:
        return sum(chess.square_rank(sq) - 1 for sq in pawns)
    else:
        return sum(6 - chess.square_rank(sq) for sq in pawns)


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

    Material features: (my_count - their_count) for each piece type.
    A positive value means I have more of that piece — correct sign with a
    positive learned weight.

    Positional conventions:
      - Penalty features (doubled_pawn, isolated_pawn, backward_pawn) are
        negated so a positive weight always means 'this is good for side to move'.
      - center_control normalized by /4 (raw max = 8 per side).
      - knight_outpost normalized by /2 (max 2 knights per side).
      - mobility normalized by /30.
      - king_safety normalized by /3.
      - pawn_advancement normalized by /20.

    Phase scaling:
      - king_safety      × phase            (less relevant in endgame)
      - mobility         × (0.5 + 0.5×phase) (midgame-weighted)
      - pawn_advancement × (0.5 + 0.5×end)   (endgame-weighted)
    """
    us    = board.turn
    them  = not us
    phase = _game_phase(board)
    end   = 1.0 - phase

    features = {
        # Material — learnable; (my_count - their_count); positive = I have more
        "material_queen":  len(board.pieces(chess.QUEEN,  us)) - len(board.pieces(chess.QUEEN,  them)),
        "material_rook":   len(board.pieces(chess.ROOK,   us)) - len(board.pieces(chess.ROOK,   them)),
        "material_bishop": len(board.pieces(chess.BISHOP, us)) - len(board.pieces(chess.BISHOP, them)),
        "material_knight": len(board.pieces(chess.KNIGHT, us)) - len(board.pieces(chess.KNIGHT, them)),
        "material_pawn":   len(board.pieces(chess.PAWN,   us)) - len(board.pieces(chess.PAWN,   them)),

        # Pawn structure
        "passed_pawn":    _count_passed_pawns(board, us)    - _count_passed_pawns(board, them),
        "doubled_pawn":  -(_count_doubled_pawns(board, us)  - _count_doubled_pawns(board, them)),
        "isolated_pawn": -(_count_isolated_pawns(board, us) - _count_isolated_pawns(board, them)),
        "backward_pawn": -(_count_backward_pawns(board, us) - _count_backward_pawns(board, them)),

        # King safety — pawn shelter minus attacker count; phase-scaled
        "king_safety":    (_king_safety_score(board, us) - _king_safety_score(board, them)) / 3.0 * phase,

        # Activity
        "center_control": (_center_control_score(board, us) - _center_control_score(board, them)) / 4.0,
        "rook_open_file": _count_rooks_open_file(board, us) - _count_rooks_open_file(board, them),
        "connected_rooks": int(_are_rooks_connected(board, us)) - int(_are_rooks_connected(board, them)),

        # Piece coordination & development
        "knight_outpost": (_safe_knight_outpost(board, us) - _safe_knight_outpost(board, them)) / 2.0,
        "bishop_pair":    int(len(board.pieces(chess.BISHOP, us)) >= 2) - int(len(board.pieces(chess.BISHOP, them)) >= 2),
        "rook_seventh":   _count_rooks_seventh(board, us)   - _count_rooks_seventh(board, them),
        "piece_development": _piece_development(board, us)  - _piece_development(board, them),

        # Mobility — midgame-weighted
        "mobility":       (_mobility(board, us) - _mobility(board, them)) / 30.0 * (0.5 + 0.5 * phase),

        # Pawn advancement — endgame-weighted
        "pawn_advancement": (_pawn_advancement(board, us) - _pawn_advancement(board, them)) / 20.0 * (0.5 + 0.5 * end),
    }

    return features


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board, weights: dict) -> float:
    """
    Evaluate a position from the perspective of the side to move.
    Returns a scalar score (positive = good for side to move).

    Unlike engine_hardcoded_material.py, there is no hard-coded material
    term here. Material is evaluated entirely through the learned weights
    for material_queen, material_rook, etc.

    Terminal states return fixed values:
        checkmate: -10,000 (side to move has been mated)
        draw:       0
    """
    if board.is_checkmate():
        return -10_000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    features = extract_features(board)
    return sum(weights.get(name, 0.0) * val for name, val in features.items())


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
