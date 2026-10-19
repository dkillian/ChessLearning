# Chess Learning — Positron Assistant Conversations

> This file captures conversations with **Positron Assistant** (Posit, PBC).
> A parallel file `ChessLearningConversations.md` captures RStudio Assistant conversations.
> Both assistants share the same `AGENTS.md` project memory but do **not** share chat history.

---

## Session 1 — October 17, 2026

### Context at session start
- RUN_ID=28: `selfplay_nn_v2.ipynb` — pure self-play NN, 10 games, completed (1-0:1, 0-1:4, Draw:5)
- RUN_ID=29: `selfplay_sf_nn_v2.ipynb` — SF-teacher NN, 10 games, status unknown at time of logging

### Topics covered

**1. Notebook execution status**
- RUN_ID=28 finished: 10 games in 17.5 min, 50% decisive, TD err 0.0444
- Attribution weights at game 10 show rook (+0.051) and piece_development (+0.032) as top movers
- RUN_ID=29 was running in a separate session Positron Assistant couldn't see until manually attached

**2. Why Positron Assistant can't see all sessions**
- Only the active/focused session is automatically attached to chat context
- Other notebooks must be manually attached by the user

**3. Phone notifications when jobs finish**
- Positron Assistant cannot send phone notifications directly
- Recommended: `ntfy.sh` (free, no account needed) — subscribe on phone, POST from script
- Alternative: `win10toast` for Windows desktop toasts; `winsound.MessageBeep()` for audio

**4. Project goal clarification**
- Goal: replicate AlphaZero-style training from scratch as a **reinforcement learning learning exercise**
- Target: ELO 1800
- Current ELO: ~982–1285 depending on condition

**5. AlphaZero gap analysis**
Current setup vs AlphaZero:
| Dimension | Current | AlphaZero |
|---|---|---|
| Architecture | MLP 19→32→1 | Deep ResNet (policy + value heads) |
| Input | 19 hand-crafted features | Raw board (8×8×planes) |
| Search | Negamax + alpha-beta | MCTS |
| Training signal | TD(0) | MCTS visit counts + outcome |
| Scale | 10 games/session | Millions of games |

**6. RL learning roadmap proposed**

What has already been learned by building this project:
- State, value function, policy, TD(0), exploration, function approximation, replay/stability

Suggested progression:
1. **Stage 1 (current):** TD(0) + MLP + negamax — equivalent to TD-Gammon (Tesauro, 1992)
2. **Stage 2:** MCTS — generates better training targets; key AlphaZero insight
3. **Stage 3:** Raw board input (768-bit bitboard) — force net to discover patterns
4. **Stage 4:** Policy head — policy gradient, cross-entropy loss on MCTS visit counts
5. **Stage 5:** Shared ResNet trunk — actual AlphaZero architecture

Binding ELO constraints in current setup (in order of impact):
1. Search depth (depth=3 → ~3 moves ahead; 1800-level needs 6–8)
2. Self-play collapse without proper replay buffer
3. MLP architecture can't represent position-dependent patterns

**7. RStudio vs Positron Assistant**
- The two assistants do not share conversation history
- This file was created to track Positron-specific conversations separately
- User has ~1 week of prior work documented in `ChessLearningConversations.md` (RStudio)

---

## Session 2 — October 17, 2026

### Context at session start
- RUN_ID=28: `selfplay_nn_v2.ipynb` — pure self-play NN, 10 games, completed (1-0:1, 0-1:4, Draw:5)
- RUN_ID=29: `selfplay_sf_nn_v2.ipynb` — SF-teacher NN, 10 games, status unknown
- Continuing from Session 1 RL roadmap discussion

### Topics covered

**1. Stage 2–5 implementation plan agreed**

Four notebooks + one new engine file to implement the AlphaZero progression:

| Notebook | Stage | Key change |
|---|---|---|
| `selfplay_mcts_v1.ipynb` | Stage 2 | MCTS replaces negamax; visit-count proportions as value target |
| `selfplay_mcts_v2.ipynb` | Stage 3 | 768-bit bitboard input; new `engine_nn_v2.py` |
| `selfplay_mcts_v3.ipynb` | Stage 4 | Policy head; dual loss (MSE value + cross-entropy policy) |
| `selfplay_mcts_v4.ipynb` | Stage 5 | Shared ResNet trunk (5 residual blocks, 128 filters) |

Design decisions confirmed:
- All notebooks start **fresh** (no seeded weights from run_id=24)
- **Full MCTS self-play** throughout (past-checkpoint hook noted for future addition)
- **Legal-moves-only softmax** for policy head (not full 4,672-move AlphaZero space)
- Stage 2 value target: **MCTS visit-count proportions** (option a) — not TD bootstrap
- Separate notebooks per stage (not one monolithic notebook)

---

### Implementation Plan: `engine_nn_v2.py`

**Location:** `scripts/python/engine_nn_v2.py`  
**Purpose:** Engine for Stages 2–5. Replaces `engine_nn.py` for all MCTS notebooks.

#### `board_to_tensor(board)`
- Returns `torch.FloatTensor` of shape `(768,)`
- 12 piece planes × 64 squares: `[P, N, B, R, Q, K, p, n, b, r, q, k]` (uppercase = white)
- Each plane is 1.0 where that piece exists, 0.0 elsewhere
- Always from white's perspective — flip board if `board.turn == chess.BLACK`

#### `class MCTSNode`
```python
__init__(board, parent=None, prior=0.0)
  children: dict[move -> MCTSNode]
  visit_count: int = 0
  value_sum: float = 0.0
  prior: float
  is_expanded: bool = False

property q_value:  value_sum / visit_count  (0 if unvisited)
property ucb(c_puct=1.4):  q_value + c_puct * prior * sqrt(parent.visit_count) / (1 + visit_count)
```

#### `mcts_search(board, net, n_simulations=200, c_puct=1.4, temperature=1.0)`
- Runs `n_simulations` from root
- Each simulation: select (UCB) → expand (net forward pass) → backup
- **Expand:** call `net.policy_value(board_to_tensor(board))` → `(policy_logits, value)`; for Stage 2/3 (value-only net), policy is uniform over legal moves
- **Backup:** propagate `-value` up the tree (negamax convention)
- Returns `(move, visit_counts_dict)` where `visit_counts_dict` maps legal moves to visit counts
- `temperature=1.0` during training (sample proportional to visits); `0.0` for eval (argmax)

#### `class ValueNet(nn.Module)` — Stage 2/3
```python
__init__(input_dim=768, hidden=256)
  fc1: Linear(768, 256), ReLU
  fc2: Linear(256, 128), ReLU
  fc3: Linear(128, 1), Tanh
forward(x) -> scalar value in [-1, 1]
policy_value(x) -> (None, value)   # uniform policy, value from net
```

#### `class DualNet(nn.Module)` — Stage 4
```python
__init__(input_dim=768, hidden=256)
  shared: Linear(768,256), ReLU, Linear(256,128), ReLU
  value_head: Linear(128, 1), Tanh
  policy_head: Linear(128, 4096)   # masked softmax over legal moves only
forward(x, legal_moves) -> (policy_probs over legal_moves, value)
policy_value(board_tensor, legal_moves) -> (policy_probs, value)
```
- Move indexing: `from_sq * 64 + to_sq` (4096-dim; legal_moves subset masked)

#### `class ResNet(nn.Module)` — Stage 5
```python
__init__(input_dim=768, n_filters=128, n_blocks=5)
  input_layer: Linear(768, n_filters)
  residual_blocks: n_blocks × ResBlock(n_filters)
    each ResBlock: Linear(n_filters, n_filters), ReLU, Linear(n_filters, n_filters), residual add, ReLU
  value_head: Linear(n_filters, 64), ReLU, Linear(64, 1), Tanh
  policy_head: Linear(n_filters, 4096) -> masked softmax over legal moves
forward(x, legal_moves) -> (policy_probs, value)
```

#### Other functions
- `get_best_move(board, net, n_simulations=200, temperature=0.0)` — thin wrapper around `mcts_search`
- `attribution_weights(board, net)` — gradient of value w.r.t. input; aggregated by piece type for DB logging; Stages 4/5 also return top-5 policy moves by probability
- `model_to_b64(net)` / `model_from_b64(b64_str, net_class)` — same as `engine_nn.py`; Supabase-compatible serialization

---

### Implementation Plan: Notebooks

#### `selfplay_mcts_v1.ipynb` — Stage 2
- Import `engine_nn_v2`; use `ValueNet(input_dim=19)` (still 19 features — only search changes)
- Replace negamax with `mcts_search(board, net, n_simulations=200)`
- **Training target:** `root.q_value` after MCTS (not TD bootstrap)
- Loss: `MSE(net(features), mcts_value_target)`
- Replay buffer: `deque(maxlen=10000)`; sample 64 per update
- DB logging: same schema as run_id=24; `run_type='mcts_v1'`
- `temperature=1.0` during training; `0.0` for ELO eval

#### `selfplay_mcts_v2.ipynb` — Stage 3
- Switch to `ValueNet(input_dim=768)` and `board_to_tensor()`
- Remove all `extract_features()` calls
- Attribution: 768-dim gradient → aggregate by piece type (12 values) for DB logging
- Everything else identical to Stage 2

#### `selfplay_mcts_v3.ipynb` — Stage 4
- Switch to `DualNet`
- **Dual loss:** `loss = MSE(value_out, outcome) + CrossEntropy(policy_logits, visit_count_probs)`
- DB schema addition: `ALTER TABLE game_records ADD COLUMN policy_loss REAL` (wrapped in try/except)
- MCTS uses policy head priors (not uniform) during node expansion
- Logging: top-5 policy moves by probability logged in attribution field (JSON string)

#### `selfplay_mcts_v4.ipynb` — Stage 5
- Switch to `ResNet(n_filters=128, n_blocks=5)`
- Everything else identical to Stage 4
- Header note: this is the actual AlphaZero architecture at small scale

---

### DB / compatibility notes
- Stages 2–3: no schema changes needed
- Stage 4: `policy_loss` column added (safe to re-run via try/except)
- `05_stockfish_benchmark.R` needs `ENGINE_MODULE = "engine_nn_v2"` for Stage 3+ runs
- `get_best_move(board, net)` signature is compatible with existing R reticulate call pattern

### Status at session end
- Plan documented; `engine_nn_v2.py` written by RStudio Assistant — all 13 unit tests passing
- Four MCTS notebooks (`selfplay_mcts_v1` through `v4`) not yet created
