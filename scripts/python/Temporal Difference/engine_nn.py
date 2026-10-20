"""
engine_nn.py -- Neural net condition (Phase 1)

Condition: NEURAL_NET
    Same 19-feature input as engine_learnable_material.py (5 material + 14 positional).
    Evaluator is a single-hidden-layer MLP with tanh activations, trained via
    TD(0) + Adam optimizer instead of manual gradient descent.

    Research question: Does a nonlinear evaluator learn chess concepts faster,
    in a different sequence, or to a higher level than the linear evaluator?

    First used in run_id=22.

Architecture:
    19 features  →  Linear(32) + tanh  →  Linear(1) + tanh  →  scalar ∈ (−1, 1)

    Output is tanh-constrained to (−1, 1), matching the TD target range.
    Terminal positions (checkmate/draw) bypass the network and return ±10,000 / 0
    directly from evaluate(), exactly as in the linear conditions.

Attribution weights (for DB logging):
    The MLP has ~672 parameters, not one per feature. To enable comparison with
    linear conditions using the same DB schema and R scripts, we log dV/dx_i at
    the starting position — the local sensitivity of the output to each input
    feature. For a linear model this equals the weight itself; for the MLP it
    captures the same concept as a linearization at a reference point.

Components:
    - ChessNet            MLP class (19 → 32 → 1, tanh)
    - new_model()         Factory: freshly initialized ChessNet
    - save_model()        Serialize to .pt (state_dict + hidden_size)
    - load_model()        Deserialize from .pt
    - attribution_weights() dV/dx_i at starting position → {feature: float}
    - evaluate()          Scalar eval for negamax search (no grad)
    - negamax()           Alpha-beta search (same structure as linear engines)
    - get_best_move()     Root search with tie-breaking randomization
"""

import chess
import random
import os
import torch
import torch.nn as nn
from typing import Optional

# Feature extraction shared with the learnable material condition.
# FEATURE_NAMES defines the 19-element order; extract_features() returns
# a dict {feature_name: float} from the perspective of the side to move.
from engine_learnable_material import FEATURE_NAMES, extract_features

N_FEATURES = len(FEATURE_NAMES)  # 19


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ChessNet(nn.Module):
    """
    Single-hidden-layer MLP chess evaluator.

    Input:  N_FEATURES (19) values from extract_features()
    Hidden: hidden_size units with tanh activation
    Output: scalar ∈ (−1, 1) — constrained by tanh to match the TD target range

    Weight initialization:
        First layer: PyTorch default (Kaiming uniform)
        Output layer: near-zero uniform (±0.01) so the engine starts with
        near-zero evaluations, analogous to zero-init in the linear conditions.
    """
    def __init__(self, n_features: int = N_FEATURES, hidden_size: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Tanh(),
        )
        # Small output init → near-zero evaluations at game 0
        nn.init.uniform_(self.net[2].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def features_tensor(self, board: chess.Board) -> torch.Tensor:
        """Extract features and return as a float32 tensor (order = FEATURE_NAMES)."""
        f = extract_features(board)
        return torch.tensor([f[name] for name in FEATURE_NAMES], dtype=torch.float32)


def new_model(hidden_size: int = 32) -> ChessNet:
    """Create a freshly initialized ChessNet."""
    return ChessNet(hidden_size=hidden_size)


def save_model(model: ChessNet, path: str) -> None:
    """
    Save model to a .pt file. Stores state_dict + hidden_size so load_model()
    can reconstruct the architecture without requiring the caller to remember it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size}, path)


def load_model(path: str) -> ChessNet:
    """Load a ChessNet from a .pt file saved by save_model()."""
    data = torch.load(path, weights_only=False)
    model = ChessNet(hidden_size=data["hidden_size"])
    model.load_state_dict(data["state_dict"])
    return model


# ---------------------------------------------------------------------------
# Attribution weights  (analogous to linear weight vector, for DB logging)
# ---------------------------------------------------------------------------

_REFERENCE_BOARD = chess.Board()  # starting position — fixed reference for attribution


def attribution_weights(model: ChessNet, board: chess.Board = None) -> dict:
    """
    Compute dV/dx_i at a reference board position (default: starting position).

    Returns {feature_name: gradient_value} — the MLP's local sensitivity to
    each input feature. For a linear model, dV/dx_i equals the weight w_i.
    For the MLP, this linearization at a fixed reference point tracks how the
    network's concept sensitivities evolve over training.

    Used in place of weight logging: stored in the `weights` DB table every
    checkpoint_every games, enabling the same R analysis scripts to work
    across both linear and neural net conditions.
    """
    if board is None:
        board = _REFERENCE_BOARD

    was_training = model.training
    model.eval()

    x = model.features_tensor(board).clone().requires_grad_(True)
    v = model(x)
    v.backward()
    grads = x.grad.detach()

    if was_training:
        model.train()

    return {name: float(grads[i]) for i, name in enumerate(FEATURE_NAMES)}


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

def evaluate(board: chess.Board, model: ChessNet) -> float:
    """
    Scalar evaluation for use in negamax search — no gradient tracking.
    Positive = good for side to move.

    Terminal states bypass the network:
        checkmate:          −10,000  (side to move has been mated)
        stalemate / insuf:       0
    """
    if board.is_checkmate():
        return -10_000.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0

    was_training = model.training
    model.eval()
    x = model.features_tensor(board)
    with torch.no_grad():
        score = model(x).item()
    if was_training:
        model.train()
    return score


# ---------------------------------------------------------------------------
# Search: negamax with alpha-beta pruning
# ---------------------------------------------------------------------------

def negamax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    model: ChessNet,
) -> float:
    """
    Negamax search with alpha-beta pruning.
    Returns the score from the perspective of the side to move.
    Same structure as engine.py / engine_learnable_material.py — only the
    evaluator function changes (dict → model).
    """
    if board.is_game_over():
        return -10_000.0 if board.is_checkmate() else 0.0

    if depth == 0:
        return evaluate(board, model)

    max_score = float("-inf")
    for move in board.legal_moves:
        board.push(move)
        score = -negamax(board, depth - 1, -beta, -alpha, model)
        board.pop()

        if score > max_score:
            max_score = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  # beta cutoff

    return max_score


# ---------------------------------------------------------------------------
# Training utilities  (used by selfplay_nn.ipynb and 07_human_play.R)
# ---------------------------------------------------------------------------

def outcome_for_color(outcome: str, color: chess.Color) -> float:
    """Convert PGN result string to scalar from the perspective of `color`."""
    if outcome == "1-0":
        return  1.0 if color == chess.WHITE else -1.0
    if outcome == "0-1":
        return -1.0 if color == chess.WHITE else  1.0
    return 0.0  # draw


# ---------------------------------------------------------------------------
# Cloud model serialization  (used by public_app/app.R via Supabase)
# ---------------------------------------------------------------------------

def model_to_b64(model: ChessNet) -> str:
    """
    Serialize model to a base64 string for storage in Supabase (TEXT column).
    Includes hidden_size so the architecture can be reconstructed on load.
    """
    import io, base64
    buf = io.BytesIO()
    torch.save({"state_dict": model.state_dict(), "hidden_size": model.hidden_size}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def model_from_b64(b64: str) -> ChessNet:
    """Reconstruct a ChessNet from a base64 string produced by model_to_b64()."""
    import io, base64
    buf  = io.BytesIO(base64.b64decode(b64))
    data = torch.load(buf, weights_only=False)
    m    = ChessNet(hidden_size=data["hidden_size"])
    m.load_state_dict(data["state_dict"])
    return m


def compute_baseline_norm(path_a: str, path_b: str, n_games: int = 10) -> float:
    """
    Average per-game parameter change norm between two checkpoint files.
    Used in the human play app to compare human games to the self-play baseline.
    path_a / path_b : .pt files for consecutive checkpoints (e.g. game_40, game_50)
    n_games         : number of self-play games between the two checkpoints
    """
    m_a = load_model(path_a)
    m_b = load_model(path_b)
    total = sum(
        (pa - pb).norm().item()
        for pa, pb in zip(m_a.parameters(), m_b.parameters())
    )
    return total / n_games


def make_optimizer(model: ChessNet, lr: float = 0.001):
    """Create a fresh Adam optimizer for `model`. Call once per session."""
    return torch.optim.Adam(model.parameters(), lr=lr)


def nn_td_update(
    model: ChessNet,
    optimizer,
    feature_dicts: list,
    colors: list,
    outcome: str,
    draw_lr_scale: float = 0.1,
) -> dict:
    """
    TD(0) update for the neural net evaluator. R-friendly interface: feature
    dicts and colors are passed as separate lists rather than zipped tuples,
    which avoids reticulate tuple-construction overhead.

    feature_dicts : list of dicts from extract_features() — one per half-move
    colors        : list of chess.WHITE / chess.BLACK — one per half-move
    outcome       : PGN result string ("1-0", "0-1", "1/2-1/2")
    draw_lr_scale : global loss scale during draw games

    Returns a dict:
        td_error     : mean absolute TD error over positions
        deltas       : {feature_name: attribution_after - attribution_before}
        attr_weights : {feature_name: dV/dx_i after update} — for DB logging
    """
    positions = list(zip(feature_dicts, colors))
    attr_before = attribution_weights(model)

    is_draw = (outcome == "1/2-1/2")
    scale   = draw_lr_scale if is_draw else 1.0

    tensors = [
        torch.tensor([feat[name] for name in FEATURE_NAMES], dtype=torch.float32)
        for feat, _ in positions
    ]

    with torch.no_grad():
        next_values = [model(x) for x in tensors[1:]]

    # Snapshot parameters before any updates (for Frobenius norm)
    params_before = [p.detach().clone() for p in model.parameters()]

    total_error    = 0.0
    position_errors = []

    for t, (feat, color) in enumerate(positions):
        if t < len(positions) - 1:
            target = -next_values[t]
        else:
            target = torch.tensor(outcome_for_color(outcome, color))

        optimizer.zero_grad()
        v    = model(tensors[t])
        loss = scale * (target - v) ** 2
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            err = abs((target - v).item())
            total_error += err
            position_errors.append(err)

    # Frobenius norm of total parameter change this game
    param_norm = sum(
        (p - p0).norm().item()
        for p, p0 in zip(model.parameters(), params_before)
    )

    attr_after = attribution_weights(model)
    return {
        "td_error":        total_error / len(positions) if positions else 0.0,
        "deltas":          {name: attr_after[name] - attr_before[name] for name in FEATURE_NAMES},
        "attr_weights":    attr_after,
        "param_norm":      param_norm,
        "position_errors": position_errors,
    }


def get_best_move(
    board: chess.Board,
    model: ChessNet,
    depth: int = 2,
    randomize: bool = True,
) -> tuple[Optional[chess.Move], float]:
    """
    Return (best_move, score) for the current side to move.
    randomize=True shuffles the move list to break evaluation ties randomly.
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
        score = -negamax(board, depth - 1, -beta, -alpha, model)
        board.pop()

        if score > best_score:
            best_score = score
            best_move  = move
        if score > alpha:
            alpha = score

    return best_move, best_score
