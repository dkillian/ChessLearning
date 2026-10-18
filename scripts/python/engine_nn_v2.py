"""
engine_nn_v2.py — Chess engine for AlphaZero-style training (Stages 2–5).

Architecture progression
────────────────────────
  Stage 2: ValueNet(input_dim=19)  + MCTS  — 19 hand-crafted features; MCTS replaces negamax
  Stage 3: ValueNet(input_dim=768) + MCTS  — bitboard input; net discovers patterns
  Stage 4: DualNet(input_dim=768)  + MCTS  — policy + value heads; cross-entropy on visit counts
  Stage 5: ResNet(input_dim=768)   + MCTS  — shared residual trunk (AlphaZero architecture)

Shared interface
────────────────
  All network classes expose:
    .policy_value(x, legal_moves=None) -> (policy_probs | None, value_tensor)
    .forward(x, legal_moves=None)      -> value_tensor (ValueNet) or (policy, value) (Dual/Res)

  MCTS entry point:
    mcts_search(board, net, ..., board_to_input=None) -> (move, visit_counts_dict)
    get_best_move(board, net, ...)                    -> (move, visit_counts_dict)

  Board representation:
    board_to_tensor(board) -> FloatTensor(768,)  [always from White's perspective]
    Move encoding: from_sq * 64 + to_sq  (4096-dimensional policy space)

  Logging helpers:
    attribution_weights(board, net, board_to_input=None) -> dict[name -> float]
    save_model(net, path)
    load_model(path, net_class, **kwargs) -> net
    model_to_b64(net) -> str
    model_from_b64(b64_str, net_class, **kwargs) -> net
"""

import base64
import io
import math
import random

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Piece-plane ordering (White uppercase, Black lowercase) ────────────────────
PIECE_PLANES = [
    (chess.PAWN,   chess.WHITE),  # 0  P
    (chess.KNIGHT, chess.WHITE),  # 1  N
    (chess.BISHOP, chess.WHITE),  # 2  B
    (chess.ROOK,   chess.WHITE),  # 3  R
    (chess.QUEEN,  chess.WHITE),  # 4  Q
    (chess.KING,   chess.WHITE),  # 5  K
    (chess.PAWN,   chess.BLACK),  # 6  p
    (chess.KNIGHT, chess.BLACK),  # 7  n
    (chess.BISHOP, chess.BLACK),  # 8  b
    (chess.ROOK,   chess.BLACK),  # 9  r
    (chess.QUEEN,  chess.BLACK),  # 10 q
    (chess.KING,   chess.BLACK),  # 11 k
]
PLANE_NAMES = ["P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"]


# ══════════════════════════════════════════════════════════════════════════════
# Board representation
# ══════════════════════════════════════════════════════════════════════════════

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Encode board as a (768,) float tensor: 12 piece planes × 64 squares.

    Always encoded from White's perspective: if it is Black's turn the board
    is mirrored vertically so that Black's pieces appear in the same ranks a
    White's pieces would in an equivalent White-to-move position.

    Plane order: P N B R Q K  p n b r q k  (White uppercase, Black lowercase).
    Square order within each plane: a1=0 … h8=63.
    """
    flip = (board.turn == chess.BLACK)
    tensor = torch.zeros(12 * 64, dtype=torch.float32)
    for plane_idx, (piece_type, color) in enumerate(PIECE_PLANES):
        for sq in board.pieces(piece_type, color):
            sq_idx = chess.square_mirror(sq) if flip else sq
            tensor[plane_idx * 64 + sq_idx] = 1.0
    return tensor


def _move_index(move: chess.Move, flip: bool = False) -> int:
    """Encode a move as from_sq * 64 + to_sq, with optional vertical flip."""
    from_sq = chess.square_mirror(move.from_square) if flip else move.from_square
    to_sq   = chess.square_mirror(move.to_square)   if flip else move.to_square
    return from_sq * 64 + to_sq


# ══════════════════════════════════════════════════════════════════════════════
# MCTS
# ══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """
    Single node in the MCTS tree.

    The node does not store the board position; the caller tracks board state
    as it traverses the tree.
    """

    __slots__ = ("parent", "children", "visit_count", "value_sum", "prior",
                 "is_expanded")

    def __init__(self, parent=None, prior: float = 0.0):
        self.parent:      "MCTSNode | None"          = parent
        self.children:    dict                        = {}   # move -> MCTSNode
        self.visit_count: int                         = 0
        self.value_sum:   float                       = 0.0
        self.prior:       float                       = prior
        self.is_expanded: bool                        = False

    @property
    def q_value(self) -> float:
        """Mean value seen through this node (0 if unvisited)."""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

    def ucb(self, c_puct: float = 1.4) -> float:
        """PUCT upper confidence bound (AlphaZero formula)."""
        parent_visits = self.parent.visit_count if self.parent is not None else 1
        return (
            self.q_value
            + c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )

    def select_child(self, c_puct: float = 1.4):
        """Return (move, child) with the highest UCB score."""
        return max(self.children.items(), key=lambda kv: kv[1].ucb(c_puct))


def _terminal_value(board: chess.Board) -> float:
    """
    Value of a terminal position from the perspective of the side to move.
    Checkmate → the side to move is mated → -1.
    Any draw → 0.
    """
    return -1.0 if board.is_checkmate() else 0.0


def mcts_search(
    board:          chess.Board,
    net:            nn.Module,
    n_simulations:  int   = 200,
    c_puct:         float = 1.4,
    temperature:    float = 1.0,
    board_to_input        = None,
) -> tuple:
    """
    Run MCTS from *board* and return the selected move plus visit counts.

    Parameters
    ──────────
    board         : current position (not modified).
    net           : network with .policy_value(x, legal_moves) interface.
    n_simulations : number of tree expansions per call.
    c_puct        : exploration constant in the PUCT formula.
    temperature   : 1.0 → sample proportional to visit counts (training);
                    0.0 → argmax (evaluation / greedy).
    board_to_input: function board -> tensor.  Defaults to board_to_tensor
                    (768-dim bitboard).  Pass a custom function for Stage 2
                    (19 hand-crafted features).

    Returns
    ───────
    (move, visit_counts_dict)
    visit_counts_dict maps each legal move to its root visit count.
    """
    if board_to_input is None:
        board_to_input = board_to_tensor

    root = MCTSNode(parent=None, prior=1.0)

    # Pre-expand root so all simulations descend through children
    if not board.is_game_over(claim_draw=True):
        legal_moves_root = list(board.legal_moves)
        x_root = board_to_input(board)
        if not isinstance(x_root, torch.Tensor):
            x_root = torch.tensor(list(x_root.values()), dtype=torch.float32)
        with torch.no_grad():
            policy_root, _ = net.policy_value(x_root, legal_moves_root)
        if policy_root is None:
            prior = 1.0 / len(legal_moves_root) if legal_moves_root else 1.0
            for move in legal_moves_root:
                root.children[move] = MCTSNode(parent=root, prior=prior)
        else:
            for move, p in zip(legal_moves_root, policy_root.tolist()):
                root.children[move] = MCTSNode(parent=root, prior=float(p))
        root.is_expanded = True

    for _ in range(n_simulations):
        node        = root
        sim_board   = board.copy()
        path        = [node]

        # ── Selection: descend using UCB until we reach an unexpanded node ──
        while node.is_expanded and not sim_board.is_game_over(claim_draw=True):
            move, node = node.select_child(c_puct)
            sim_board.push(move)
            path.append(node)

        # ── Evaluation ───────────────────────────────────────────────────────
        if sim_board.is_game_over(claim_draw=True):
            value = _terminal_value(sim_board)
        else:
            legal_moves = list(sim_board.legal_moves)
            x = board_to_input(sim_board)
            if not isinstance(x, torch.Tensor):
                # Support feature-dict input from extract_features()
                x = torch.tensor(list(x.values()), dtype=torch.float32)

            with torch.no_grad():
                policy, value_t = net.policy_value(x, legal_moves)
            value = value_t.item() if isinstance(value_t, torch.Tensor) else float(value_t)

            # ── Expansion: add child nodes ───────────────────────────────────
            if policy is None:
                # Uniform prior (ValueNet / Stage 2-3)
                prior = 1.0 / len(legal_moves) if legal_moves else 1.0
                for move in legal_moves:
                    node.children[move] = MCTSNode(parent=node, prior=prior)
            else:
                # Policy-based priors (DualNet / ResNet)
                # policy is already a probability vector over legal_moves
                for move, p in zip(legal_moves, policy.tolist()):
                    node.children[move] = MCTSNode(parent=node, prior=float(p))

            node.is_expanded = True

        # ── Backup: propagate value up the path (negate at each level) ──────
        for n in reversed(path):
            n.visit_count += 1
            n.value_sum   += value
            value          = -value

    # ── Move selection from root ─────────────────────────────────────────────
    if not root.children:
        # Fallback: return a random legal move (should not happen in practice)
        return random.choice(list(board.legal_moves)), {}

    visit_counts = {move: child.visit_count for move, child in root.children.items()}

    if temperature == 0.0:
        move = max(visit_counts, key=visit_counts.get)
    else:
        moves  = list(visit_counts.keys())
        counts = torch.tensor([visit_counts[m] for m in moves], dtype=torch.float32)
        probs  = torch.softmax(counts / temperature, dim=0).numpy()
        move   = moves[np.random.choice(len(moves), p=probs)]

    return move, visit_counts


def get_best_move(
    board:         chess.Board,
    net:           nn.Module,
    n_simulations: int   = 200,
    temperature:   float = 0.0,
    board_to_input        = None,
) -> tuple:
    """
    Thin wrapper around mcts_search for use from R (reticulate) and notebooks.
    temperature=0.0 → greedy best move (evaluation); 1.0 → training sampling.
    """
    return mcts_search(
        board,
        net,
        n_simulations=n_simulations,
        temperature=temperature,
        board_to_input=board_to_input,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Networks
# ══════════════════════════════════════════════════════════════════════════════

class ValueNet(nn.Module):
    """
    Stage 2 (input_dim=19) and Stage 3 (input_dim=768) value network.
    Outputs a scalar in [-1, 1].  Policy is uniform over legal moves.
    """

    def __init__(self, input_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.fc3 = nn.Linear(hidden // 2, 1)

    def forward(self, x: torch.Tensor, legal_moves=None) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x)).squeeze(-1)

    def policy_value(self, x: torch.Tensor, legal_moves=None):
        """
        Return (None, value).  Policy is uniform — handled in mcts_search.
        """
        value = self.forward(x)
        return None, value


class _ResBlock(nn.Module):
    def __init__(self, n_filters: int):
        super().__init__()
        self.fc1 = nn.Linear(n_filters, n_filters)
        self.fc2 = nn.Linear(n_filters, n_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.relu(x + residual)


class DualNet(nn.Module):
    """
    Stage 4: shared trunk with separate value and policy heads.
    Policy outputs a probability distribution over *legal* moves only
    (masked softmax using from_sq * 64 + to_sq indexing).
    """

    def __init__(self, input_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.shared_fc1 = nn.Linear(input_dim, hidden)
        self.shared_fc2 = nn.Linear(hidden, hidden // 2)
        self.value_head  = nn.Linear(hidden // 2, 1)
        self.policy_head = nn.Linear(hidden // 2, 4096)

    def forward(self, x: torch.Tensor, legal_moves=None):
        h = F.relu(self.shared_fc1(x))
        h = F.relu(self.shared_fc2(h))
        value = torch.tanh(self.value_head(h)).squeeze(-1)
        if legal_moves is None:
            return None, value
        flip         = False  # caller responsible for correct orientation
        indices      = torch.tensor(
            [_move_index(m) for m in legal_moves], dtype=torch.long
        )
        policy_full  = self.policy_head(h)
        policy_legal = policy_full[indices]
        policy_probs = torch.softmax(policy_legal, dim=-1)
        return policy_probs, value

    def policy_value(self, x: torch.Tensor, legal_moves=None):
        return self.forward(x, legal_moves)


class ResNet(nn.Module):
    """
    Stage 5: shared residual trunk with value and policy heads.
    Mirrors the AlphaZero architecture at small scale
    (5 residual blocks, 128 filters, linear layers throughout).
    """

    def __init__(self, input_dim: int = 768, n_filters: int = 128, n_blocks: int = 5):
        super().__init__()
        self.input_layer     = nn.Linear(input_dim, n_filters)
        self.residual_blocks = nn.ModuleList(
            [_ResBlock(n_filters) for _ in range(n_blocks)]
        )
        # Value head
        self.value_fc  = nn.Linear(n_filters, 64)
        self.value_out = nn.Linear(64, 1)
        # Policy head
        self.policy_out = nn.Linear(n_filters, 4096)

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.input_layer(x))
        for block in self.residual_blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor, legal_moves=None):
        h     = self._trunk(x)
        value = torch.tanh(self.value_out(F.relu(self.value_fc(h)))).squeeze(-1)
        if legal_moves is None:
            return None, value
        indices      = torch.tensor(
            [_move_index(m) for m in legal_moves], dtype=torch.long
        )
        policy_full  = self.policy_out(h)
        policy_legal = policy_full[indices]
        policy_probs = torch.softmax(policy_legal, dim=-1)
        return policy_probs, value

    def policy_value(self, x: torch.Tensor, legal_moves=None):
        return self.forward(x, legal_moves)


# ══════════════════════════════════════════════════════════════════════════════
# Attribution
# ══════════════════════════════════════════════════════════════════════════════

def attribution_weights(
    board:         chess.Board,
    net:           nn.Module,
    board_to_input = None,
    feature_names  = None,
) -> dict:
    """
    Compute attribution weights: gradient of value output w.r.t. input features.

    For Stage 2 (19 features): pass board_to_input=<feature_fn> and feature_names.
    For Stages 3-5 (768-dim):  gradients are aggregated per piece plane (12 values).

    Returns a dict mapping feature/plane name to attribution float.
    Also returns 'policy_top5' key for DualNet/ResNet with a JSON-serialisable
    list of {move, prob} dicts for the 5 highest-probability policy moves.
    """
    legal_moves = list(board.legal_moves)
    if board_to_input is None:
        board_to_input = board_to_tensor

    x = board_to_input(board)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(list(x.values()), dtype=torch.float32)

    x = x.detach().clone().requires_grad_(True)

    net.eval()
    _, value = net.policy_value(x, legal_moves)
    value.backward()

    grad = x.grad.detach()

    if feature_names is not None:
        # Stage 2: named features
        attr = {name: grad[i].item() for i, name in enumerate(feature_names)}
    else:
        # Stages 3-5: aggregate gradient magnitude per piece plane (12 planes × 64 sq)
        grad_planes = grad.view(12, 64).abs().sum(dim=1)
        attr = {name: grad_planes[i].item() for i, name in enumerate(PLANE_NAMES)}

    # Policy top-5 for DualNet / ResNet
    if isinstance(net, (DualNet, ResNet)) and legal_moves:
        x2 = x.detach().clone()
        with torch.no_grad():
            policy_probs, _ = net.policy_value(x2, legal_moves)
        top_k = min(5, len(legal_moves))
        top_indices = torch.topk(policy_probs, top_k).indices.tolist()
        attr["policy_top5"] = [
            {"move": legal_moves[i].uci(), "prob": round(policy_probs[i].item(), 4)}
            for i in top_indices
        ]

    return attr


# ══════════════════════════════════════════════════════════════════════════════
# Model persistence
# ══════════════════════════════════════════════════════════════════════════════

def new_model(net_class=ValueNet, **kwargs) -> nn.Module:
    """Create a fresh, randomly-initialised network."""
    return net_class(**kwargs)


def save_model(net: nn.Module, path: str) -> None:
    """Save network weights to a .pt checkpoint file."""
    torch.save({"state_dict": net.state_dict(), "class": net.__class__.__name__}, path)


def load_model(path: str, net_class=ValueNet, **kwargs) -> nn.Module:
    """Load a network from a .pt checkpoint file."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    net = net_class(**kwargs)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net


def model_to_b64(net: nn.Module) -> str:
    """Serialise network weights to a base-64 string (Supabase-compatible)."""
    buf = io.BytesIO()
    torch.save({"state_dict": net.state_dict(), "class": net.__class__.__name__}, buf)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def model_from_b64(b64_str: str, net_class=ValueNet, **kwargs) -> nn.Module:
    """Deserialise a network from a base-64 string."""
    buf = io.BytesIO(base64.b64decode(b64_str))
    checkpoint = torch.load(buf, map_location="cpu", weights_only=False)
    net = net_class(**kwargs)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    return net
