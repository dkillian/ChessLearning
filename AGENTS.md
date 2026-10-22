| `README.md` | Public-facing GitHub readme |
| `ChessLearningConversations.md` | Running transcript of RStudio Assistant conversations |
| `PositronAssistantConversations.md` | Running transcript of Positron Assistant conversations |

--- Overview

This is the **Chess Learning** workspace, located at:
`C:/Users/dkill/OneDrive/Documents/Chess Learning/`

GitHub repo: https://github.com/dkillian/ChessLearning

The workspace is a research project that trains a simple chess engine via self-play
and documents the progression of chess concept acquisition over training time.

---

## Workspace Conventions

See `ActivitySetupGuide.md` for the full guide on initializing and managing activities.

Project structure (flat — all at repo root):

```
Chess Learning/
├── AGENTS.md
├── ActivitySetupGuide.md
├── README.md
├── ChessLearningDocumentation.md
├── ChessLearningLog.md
├── ChessLearningConversations.md
├── scripts/
│   ├── python/   # Chess engine, self-play, TD learning
│   └── r/        # Visualization and cross-run analysis
├── data/         # SQLite database
├── viz/          # Plots and figures
└── models/       # Saved weight checkpoints
```

Script naming convention:

| Prefix | Stage | Purpose |
|--------|-------|---------|
| `01_*` | Engine / Setup | Chess rules, feature extraction, evaluator, search |
| `02_*` | Self-Play / Learning | Self-play loop, TD learning, database logging |
| `03_*` | Analysis / Reporting | R visualization, cross-run comparison |

Each activity maintains three documentation files at its root:
- `[ActivityName]Documentation.md` — internal reference (overview, methods, architecture, file table)
- `[ActivityName]Log.md` — session-by-session log (append each session)
- `[ActivityName]Conversations.md` — running transcript of assistant conversations

To resume an activity, ask the assistant to read `ChessLearningLog.md` before proceeding.

---

## Workspace-Level Files

| File | Description |
|------|-------------|
| `AGENTS.md` | This file — project memory for Posit Assistant |
| `ActivitySetupGuide.md` | Guide for initializing and managing activities |
| `README.md` | Public-facing GitHub readme |

---

## Project Purpose

Train a chess engine via self-play using a linear evaluation function with explicit
chess concept feature weights. Document and visualize the progression of concept
acquisition — and investigate whether the sequence of learning is reproducible
across independent training runs.

To start a session:
1. Ask the assistant to read `ChessLearningLog.md` before proceeding
2. Confirm the current development stage before writing code

---

## Tech Stack

- **Python**: `python-chess` for rules; custom engine for feature extraction, minimax search, TD learning
- **R**: `tidyverse`, `DBI`, `RSQLite` for visualization and analysis
- **Stockfish 19**: installed; **lc0 v0.32.1** (NVIDIA GPU/CUDA 12) installed to `C:\lc0\`; bundled network `791556.pb.gz`; Maia weights available at github.com/CSSLab/maia-chess (NOT recommended as training teacher — nodes=1 design)
- **Key reference**: McGrath et al. PNAS 2022 — "Acquisition of Chess Knowledge in AlphaZero" — central scientific reference for concept acquisition sequence research
- **Storage**: SQLite database at `data/chess_learning.db` — 10 tables (runs, games, game_records, weights, weight_deltas, feature_stats, position_logs, baseline_evals, stockfish_evals, stockfish_games); `stockfish_games` created by `05_stockfish_benchmark.R`, not by `setup_database()`. **Session 21:** `games` and `stockfish_games` gained `seed_run_id INTEGER` and `seed_game_number INTEGER` columns to track which checkpoint each game was played against.
- **Stockfish 19**: installed via winget; used for ELO benchmarking and live game play; minimum `UCI_Elo` = 1320

---

## Current Architecture (as of Session 25)

### Engine (`scripts/python/Temporal Difference/engine.py`) — v2
- **14 learnable positional features** (material features removed entirely)
- **Material evaluation hard-coded** in `evaluate()` via `PIECE_VALUES` constant
  (queen=0.9, rook=0.5, bishop=0.3, knight=0.3, pawn=0.1 — scaled to TD target range)
- Search: negamax with alpha-beta pruning

### Learnable Features (14 total)
Pawn structure: `passed_pawn`, `doubled_pawn`, `isolated_pawn`, `backward_pawn` (pawn sentry definition), `pawn_advancement`
King safety: `king_safety` (pawn shelter − attacker count; ÷ 3; phase-scaled × phase)
Activity: `center_control` (attacks + occupation of d4/d5/e4/e5; ÷ 4), `rook_open_file`, `connected_rooks`
Coordination: `knight_outpost`, `bishop_pair`, `rook_seventh`, `piece_development`, `mobility` (÷ 30; phase-scaled × (0.5+0.5×phase))
Pawn advancement: `pawn_advancement` (total advancement of **all** pawns from starting rank; ÷ 20; endgame-scaled × (0.5+0.5×end))

**Sign convention (v2 engine):** All 14 features are oriented so that a higher value is better for the side to move. Penalty features (`doubled_pawn`, `isolated_pawn`, `backward_pawn`) are negated in `extract_features()` — stored as (opponent_count − my_count) — so the **correct learned weight is positive for all 14 features**. When assessing sign correctness for v2 runs (18+), any negative weight is wrong. (v1 engine did not negate penalty features; correct signs for v1 runs 1–17 were negative for those three.)

**`knight_outpost` (session 16):** Replaced `knight_pst` in `engine_learnable_material.py` only (`engine.py` unchanged). Counts knights in opponent's half not attacked by any opponent pawn — safe outpost knights. PST approach generated 75.8% wrong-direction TD updates because centralized knights were frequently captured at depth=3.

### Game Phase Weighting
`_game_phase(board)` returns midgame fraction [0,1] from weighted piece count / 24 (Q=4, R=2, N/B=1).
Applied to: `king_safety` (× phase), `mobility` (× (0.5+0.5×phase)), `pawn_advancement` (× (0.5+0.5×end)).

### Training Script (`scripts/python/Temporal Difference/selfplay_td0.ipynb`)
- **Temporal difference (0) bootstrapping**: target for position t = `-V(s_{t+1})`; terminal uses actual outcome
- **Per-feature draw LR scales**: `draw_lr_scales = {'king_safety': 0.05, 'center_control': 0.2}` — king_safety gets 5% LR during draws; center_control 20% (added run_id=14: center_control became top draw-gradient feature in run_id=13)
- **`BASELINE_EVERY = None`** — random mover baseline disabled; engine wins 100% at all checkpoints (ceiling effect). Replace with Stockfish-at-low-ELO in future runs.
- `selfplay_mc.ipynb` preserved as Monte Carlo archive
- `selfplay_softmax.ipynb` — softmax selection + style modifiers variant (run_id=13+)

### Key Run History
| run_id | Games | Depth | Key change | Result |
|--------|-------|-------|------------|--------|
| 1–4 | various | 2 | MC; various fixes | Diagnosed draw equilibrium |
| 5–6 | 300 | 2 | MC + draw_lr_scale | 18–26% decisive; continued erosion |
| 7 | 600 | 2 | TD(0) | Erosion reduced; slow linear drift |
| 10 | 50 | 2 | TD(0) + frozen material | Zero erosion confirmed |
| 11 | 850 | 2 | Hard-coded material; 12 positional features | **78% decisive; all 8 significant signs correct** |
| 12 | 1200 | 3 | Depth=3; king_safety draw_lr=0.05 | **84% decisive at game 600; declining to 80.4% at game 1200; center_control crossed zero** |
| 13 | 212 | 3 | Softmax selection (τ=0.05, top-10); `selfplay_softmax.ipynb` | 69.4% decisive (regression); 2:1 black-wins asymmetry; pawn_advancement correct sign first time |
| 14 | 400 | 3 | **14-feature engine**; epsilon-greedy; draw_lr king_safety=0.05 center_control=0.2 | **81.2% decisive; 11/14 correct signs; benchmark 3W/1L/6D vs SF1320 (ELO ~1428)** |
| 15 | 60+ | 3 | 14-feature engine; softmax τ=0.05 | 70% decisive; 11/14 correct signs; 2.8:1 black asymmetry persists |
| 16 | 560 | 3 | **Stockfish teacher** (SF ELO=1320); `selfplay_sf.ipynb`; move_limit=80 | ELO 1212→1267→1212 (peaked game 60, regressed by 560); 7/14 correct signs; not continued since — results were poor but approach not formally closed off |
| 17 | 5 | 3 | **Human play** (v1 engine); seeded from run_id=14 game 400; all white wins | 11/14 correct signs maintained; same 3 wrong-sign features persist |
| 18 | 450 | 3 | **v2 engine** (redesigned king_safety, pawn_advancement, backward_pawn); epsilon-greedy from zero | **84.2% decisive; 11/14 correct signs** (wrong: pawn_advancement, knight_pst, doubled_pawn); benchmark game 450: 2W/1L/7D vs SF1320 (~1355 ELO) |
| 19 | 1 | 3 | **Human play v2**; seeded from run_id=18 game_number=20; engine won (0-1, 102 half-moves) | **11/14 correct signs** (v2 convention; wrong: doubled_pawn, isolated_pawn, connected_rooks); noticeably stronger play reported |
| 20 | 510 | 3 | **Learnable material** (`engine_learnable_material.py`); 19 features (5 material + 14 positional); all weights init zero | **76.1% decisive; 16/19 correct signs**; material weights plateaued early (rook > queen persists); benchmark 2W/4L/14D vs SF1320 (~1285 ELO) |
| 21 | 1050+ | 3 | **`knight_outpost`** replaces `knight_pst`; learnable material; from zero | **17/19 correct signs** at game 1050; material ordering correct (rook > queen > bishop > knight > pawn); benchmark 3W/5L/12D vs SF1320 (~1285 ELO); black asymmetry 1.10:1 (best in learnable-material runs) |
| 22 | 3 | 3 | Human play; seeded run_id=21 game_200; learnable material | — |
| 23 | ? | 3 | Human play; seeded run_id=21 game_240; learnable material | — |
| 24 | 1005+ | 3 | **Neural net** (MLP 19→32→1); TD(0)+Adam; `selfplay_nn.ipynb`; past-checkpoint opponent added Session 20 | **17/19 correct signs at game 1005; ELO ~982 at game 900 (declining — self-play collapse); pawn_advancement top feature (+0.591); past-opponent mechanism corrected 3 stuck features in 10 games** |
| 25 | 1 | 3 | Human play (accidental); seeded from NN run_id=24 game_0 attribution weights; linear TD update (old code) | Near-zero-initialized linear run; not meaningful |
| 26 | 5 | 3 | **NN human play** (new NN-aware code); seeded from run_id=24 game_10 | Signs volatile: 14→11→15/19 across games 3–5; per-game seed not recorded in DB (gap now fixed) |
| 27 | 640 | 3 | **SF teacher** (`selfplay_sf_nn.ipynb`); TD(0)+Adam; fresh start | ELO ~1105 at games 110 & 390 (above run 24 peak); slipped to 1079 at game 440; material_queen +0.442; signs volatile 11–18/19 (peaked game 340) |
| 28 | 10 | 3 | **Self-play v2** (`selfplay_nn_v2.ipynb`); AdamW + LR schedule + resignation | 11/19 signs; 50% draw rate (expected, fresh model); no resignations |
| 29 | 10 | 3 | **SF teacher v2** (`selfplay_sf_nn_v2.ipynb`); AdamW + LR schedule + temperature + resignation | 12/19 signs at game 10 (vs run 27's 17/19); weaker start attributed to near-uniform temperature on fresh model |
| 30 | 81 | — | **MCTS v1** (`selfplay_mcts_v1.ipynb`); ValueNet(19); game-outcome replay buffer; N_SIMULATIONS=200 | 98.8% draws; 10/19 correct signs; weights random-walk; draw collapse confirmed |
| 31 | 80 | — | **MCTS v2** (`selfplay_mcts_v2.ipynb`); ValueNet(768); bitboard input | 100% draws; weights monotone → 0 (trivial solution); draw collapse architecture-independent |
| 32 | 80 | — | **MCTS v3** (`selfplay_mcts_v3.ipynb`); DualNet(768); policy + value heads | 100% draws; policy loss ~3.0 (near-uniform); weights decline then slightly stabilize |
| 33 | 80 | — | **MCTS v4** (`selfplay_mcts_v4.ipynb`); ResNet(768, 5 blocks, 128 filters) | 100% draws; 1.9 min/game (fastest); draw collapse confirmed across all 4 stages |
| 34 | 1323 | — | **MCTS v4 SF teacher** (`stockfish_mcts_v4.ipynb`); ResNet + MCTS vs SF1320; VALUE_LOSS_WEIGHT=2.0 (game 521+) | 0W/63D/1260L (4.8% draw rate); attribution peak ~0.9 games 80–100, collapsed by game 800; LR reset bug at game 521 |
| 35 | — | — | **Self-play v2** (`selfplay_mcts_v4.ipynb`); Dirichlet noise (α=0.3, ε=0.25); LR resume fix; TEMP_THRESHOLD=9999 | Not yet run |
| 36 | 100 | — | **SF teacher v2** (`stockfish_mcts_v4.ipynb`); same fixes as run 35 but with TEMP_THRESHOLD=30 | 0W/0D/100L; attribution near-zero throughout — temperature annealing identified as culprit |
| 37 | 100 | — | **SF teacher v3** (`stockfish_mcts_v4.ipynb`); Dirichlet noise + LR fix + TEMP_THRESHOLD=9999 (annealing disabled) | 0W/1D/99L; small non-zero attributions (max 0.101 game 10, ~0.001 game 100); no surge yet — extend to 300 games |

### engine_nn_v2.py — Stage 2–5 Engine (Session 21)

**Location:** `scripts/python/Monte Carlo Tree Search/engine_nn_v2.py`

| Component | Description |
|---|---|
| `board_to_tensor(board)` | 768-dim float tensor: 12 piece planes × 64 squares; always White's perspective (flips if Black to move) |
| `MCTSNode` | PUCT UCB: Q + c_puct × P × √N_parent / (1 + N); root pre-expanded before simulation loop |
| `mcts_search(board, net, ...)` | `board_to_input` param: defaults to `board_to_tensor`; pass custom fn for Stage 2 (19 features) |
| `ValueNet(input_dim, hidden=256)` | Stages 2–3; `input_dim=19` for Stage 2, `768` for Stage 3; policy=uniform |
| `DualNet(input_dim=768, hidden=256)` | Stage 4; shared trunk + value head + policy head (masked softmax over legal moves) |
| `ResNet(input_dim=768, n_filters=128, n_blocks=5)` | Stage 5; residual trunk + value + policy heads |
| `attribution_weights(board, net)` | Gradient of value w.r.t. input; 12 plane-aggregated values for 768-dim; `policy_top5` for Dual/ResNet |
| Move encoding | `from_sq * 64 + to_sq` (4096-dim policy space) |

All 13 unit tests passing as of Session 21.

### Current Weight State (run_id=24, game 100 — neural net, overnight run in progress)

**run_id=17 (human play, v1 engine, seeded from run_id=14 game 400, 5 games — v1 sign convention):**
Correct sign (11/14): passed_pawn (+0.120), mobility (+0.105), center_control (+0.052),
rook_open_file (+0.047), rook_seventh (+0.044), king_safety (+0.032), piece_development (+0.027),
bishop_pair (+0.018), connected_rooks (+0.010), isolated_pawn (−0.001), doubled_pawn (−0.019)
Wrong sign (3/14): backward_pawn (+0.023), knight_pst (−0.012), pawn_advancement (−0.023)

**run_id=18 (v2 engine, 380 games from zero — v2 sign convention: all positive correct):**
Correct sign (11/14): mobility (+0.112), passed_pawn (+0.112), center_control (+0.059),
rook_open_file (+0.049), king_safety (+0.038), rook_seventh (+0.037), isolated_pawn (+0.032),
piece_development (+0.028), bishop_pair (+0.023), backward_pawn (+0.006), connected_rooks (+0.005)
Wrong sign (3/14): pawn_advancement (−0.010), knight_pst (−0.013), doubled_pawn (−0.019)

**run_id=19 (v2 human play, seeded from run_id=18 game_number=20, 1 game — v2 sign convention):**
Correct sign (11/14): passed_pawn (+0.013), rook_open_file (+0.013), mobility (+0.010),
king_safety (+0.007), center_control (+0.005), piece_development (+0.005), rook_seventh (+0.003),
bishop_pair (+0.003), backward_pawn (+0.005), knight_pst (+0.000), pawn_advancement (+0.000)
Wrong sign (3/14): doubled_pawn (−0.002), isolated_pawn (−0.003), connected_rooks (−0.001)
Note: backward_pawn now correctly positive (was misclassified as wrong under v1 convention)

**run_id=20 (learnable material, engine_learnable_material.py, 510 games from zero — v2 sign convention: all positive correct):**
Correct sign (16/19): material_rook (+0.186), material_queen (+0.116), mobility (+0.074),
material_bishop (+0.074), passed_pawn (+0.049), material_pawn (+0.048), center_control (+0.043),
rook_seventh (+0.041), material_knight (+0.039), rook_open_file (+0.028), connected_rooks (+0.019),
piece_development (+0.013), king_safety (+0.010), isolated_pawn (+0.007), backward_pawn (+0.004),
bishop_pair (+0.004)
Wrong sign (3/19): doubled_pawn (−0.003), pawn_advancement (−0.011), knight_pst (−0.012)
Note: material weights plateaued early; rook (0.186) still exceeds queen (0.116) at game 510.
Benchmark: 2W/4L/14D vs SF1320 — estimated ELO ~1285 (~70 pts below hardcoded-material baseline).
`knight_pst` replaced by `knight_outpost` in engine_learnable_material.py; run_id=21 started from zero.

**run_id=24 (neural net MLP 19→32→1, TD(0)+Adam, 1005+ games — past-checkpoint opponent active from game 996; attribution weights = dV/dx_i at starting position; game 1005 checkpoint):**
Correct sign (17/19): pawn_advancement (+0.591), material_queen (+0.588), material_rook (+0.216), knight_outpost (+0.162), material_bishop (+0.124), mobility (+0.108), passed_pawn (+0.106), rook_seventh (+0.076), center_control (+0.059), rook_open_file (+0.058), material_knight (+0.054), material_pawn (+0.043), king_safety (+0.038), piece_development (+0.037), connected_rooks (+0.011), backward_pawn (+0.001), doubled_pawn (+0.001)
Wrong sign (2/19): isolated_pawn (−0.004), bishop_pair (−0.057)
Note: 17/19 is the best sign count for the NN in any run. Three features corrected in 10 games after past-opponent mechanism introduced (center_control, rook_open_file, connected_rooks — all had been stuck wrong since game 200). Stockfish ELO ~982 at game 900 (declining trend: 1079→1079→1051→982). ELO benchmark post-mechanism pending. Overnight tranche in progress.

**run_id=21 (learnable material, knight_outpost replaces knight_pst, 1050 games from zero — v2 sign convention: all positive correct):**
Correct sign (17/19): material_rook (+0.213), material_queen (+0.184), material_bishop (+0.112),
mobility (+0.074), material_knight (+0.060), center_control (+0.052), material_pawn (+0.045),
rook_seventh (+0.037), king_safety (+0.029), knight_outpost (+0.025), passed_pawn (+0.024),
connected_rooks (+0.010), piece_development (+0.008), isolated_pawn (+0.005), backward_pawn (+0.005),
rook_open_file (+0.002), bishop_pair (+0.001)
Wrong sign (2/19): doubled_pawn (−0.007), pawn_advancement (−0.031)
Note: material ordering fully correct (rook > queen > bishop > knight > pawn); knight_outpost
solidly positive throughout; pawn_advancement crossed zero at game 200 as in run_id=20 (structural
credit assignment issue); doubled_pawn volatile random-walk. Black asymmetry 1.10:1 — best in any
learnable-material run.
Benchmark: 3W/5L/12D vs SF1320 — estimated ELO ~1285 (60% games hit 80 half-move cap; next
benchmark will use no move limit — chess rules only via claim_draw=True).

### Stockfish Integration
- **Stockfish 19** installed via `winget install Stockfish.Stockfish`
- Auto-detected at: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Stockfish.Stockfish_*\stockfish\stockfish-windows-x86-64-universal.exe`
- Accessible via `chess.engine.SimpleEngine.popen_uci(SF_PATH)` in python-chess
- ELO limiting: `sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})` — minimum is 1320 for Stockfish 19
- Always pass `ponder=False` to `sf.play()` to prevent UCI communication hangs
- `stockfish_evals` DB table logs aggregate match results (source = `'benchmark'` or `'live'`)
- `stockfish_games` DB table logs individual game PGNs (created by `05_stockfish_benchmark.R`)
- **Session 12 benchmark (run_id=14, 400 games): 3W/1L/6D at ELO 1320 — estimated ELO ~1390**
- **Session 15 benchmark (run_id=18, 450 games): 2W/1L/7D at ELO 1320 — estimated ELO ~1355**
- **Session 16 benchmark (run_id=20, 510 games): 2W/4L/14D at ELO 1320 — estimated ELO ~1285** (learnable material; wrong engine import bug fixed before re-run)
- **Session 17 benchmark (run_id=21, 1040 games): 3W/5L/12D at ELO 1320 — estimated ELO ~1285** (learnable material + knight_outpost; 60% move-limit draws; next benchmark will use no move limit)
- `stockfish_evals` also receives ELO estimates from `selfplay_sf.ipynb` (source = `'sf_teacher'`)
- Windows asyncio fix required in Jupyter: `asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())` before `popen_uci()`

---

## Python Training Scripts

| File | Purpose |
|------|---------|
| `scripts/python/Temporal Difference/engine.py` | Importable module: 14-feature positional extractor (v2), linear evaluator with hard-coded material, negamax search with alpha-beta, `_game_phase()` helper |
| `scripts/python/Temporal Difference/engine_hardcoded_material.py` | Archive of hardcoded-material engine condition (runs 1–19): 14 positional features, material hard-coded |
| `scripts/python/Temporal Difference/engine_learnable_material.py` | Learnable-material engine condition (run 20+): 19 features — 5 material + 14 positional; no hard-coded piece values. **Session 16:** `knight_pst` replaced by `knight_outpost` (`_safe_knight_outpost()` — knights in opponent's half not attacked by opponent pawns) |
| `scripts/python/Temporal Difference/engine_nn.py` | Neural net engine (run 24+): ChessNet MLP (19→32→1, tanh); `evaluate()`, `negamax()`, `get_best_move()`, `attribution_weights()` (dV/dx_i at starting position), `nn_td_update()` (R-friendly interface for reticulate), `model_to_b64()` / `model_from_b64()` for Supabase storage |
| `scripts/python/Temporal Difference/selfplay_nn.ipynb` | TD(0)+Adam self-play training for NN condition; attribution weights logged; current runs: 24 |
| `scripts/python/Temporal Difference/selfplay_nn_v2.ipynb` | **New (Session 21)** — self-play + AdamW (weight_decay=1e-4) + MultiStepLR + resignation; run_id=28 |
| `scripts/python/Temporal Difference/selfplay_sf_nn.ipynb` | **New (Session 21)** — SF teacher (Adam); chess-rules-only termination; run_id=27 |
| `scripts/python/Temporal Difference/selfplay_sf_nn_v2.ipynb` | **New (Session 21)** — SF teacher + AdamW + MultiStepLR + temperature move selection + resignation; run_id=29 |
| `scripts/python/Monte Carlo Tree Search/selfplay_mcts_v1.ipynb` | Stage 2: MCTS + ValueNet(input_dim=19); game-outcome replay buffer; AdamW + MultiStepLR; N_SIMULATIONS=200; run_id=30. **Session 23:** resignation added (threshold=0.9, consecutive=4); td_error logging fixed |
| `scripts/python/Monte Carlo Tree Search/selfplay_mcts_v2.ipynb` | Stage 3: MCTS + ValueNet(input_dim=768); bitboard input; 12 plane-aggregated attribution; run_id=31. **Session 23:** resignation + td_error fix |
| `scripts/python/Monte Carlo Tree Search/selfplay_mcts_v3.ipynb` | Stage 4: MCTS + DualNet(768); dual loss MSE+CE; policy targets = normalized visit counts; run_id=32. **Session 23:** resignation added; fixed `_, val_out = net(...)` unpacking (DualNet returns policy first) |
| `scripts/python/Monte Carlo Tree Search/selfplay_mcts_v4.ipynb` | Stage 5: MCTS + ResNet(768, 5 blocks, 128 filters); AlphaZero architecture; same dual loss; run_id=33. **Session 23:** resignation added; same unpacking fix as v3 |
| `scripts/python/Monte Carlo Tree Search/stockfish_mcts_v4.ipynb` | **New (Session 23)** — MCTS v4 + Stockfish teacher; ResNet + MCTS vs SF1320; engine positions only in replay buffer; VALUE_LOSS_WEIGHT=2.0; run_id=34+ |
| `scripts/python/Monte Carlo Tree Search/make_sf_mcts_v4.py` | **New (Session 23)** — Generation script for stockfish_mcts_v4.ipynb (archived) |
| `scripts/python/Monte Carlo Tree Search/engine_nn_v2.py` | **New (Session 21)** — Stage 2–5 engine: `board_to_tensor` (768-dim bitboard), `MCTSNode`, `mcts_search`, `ValueNet` (Stages 2–3), `DualNet` (Stage 4), `ResNet` (Stage 5); all 13 tests passing |
| `scripts/python/Monte Carlo Tree Search/update_v2_notebooks.py` | Utility: patches v2 notebooks with AlphaZero improvements |
| `scripts/python/Temporal Difference/selfplay_td0.ipynb` | TD(0) self-play training; epsilon-greedy; v2 engine; current runs: 18 |
| `scripts/python/Temporal Difference/selfplay_learnable_material.ipynb` | TD(0) self-play training; learnable material condition; imports `engine_learnable_material`; current runs: 20 |
| `scripts/python/Temporal Difference/selfplay_softmax.ipynb` | TD(0) self-play; softmax move selection (τ=0.05, top-10); current runs: 15 |
| `scripts/python/Temporal Difference/selfplay_sf.ipynb` | **Stockfish teacher** (run_id=16): plays vs Stockfish at configurable ELO; run_id=16 showed ELO regression and only 7/14 correct signs — not continued since, but not closed off as a future approach |
| `scripts/python/Temporal Difference/selfplay_td0_v1_runs1-14.ipynb` | Archived v1 TD(0) notebook (runs 1–14) |
| `scripts/python/Temporal Difference/selfplay_softmax_v1_run15.ipynb` | Archived v1 softmax notebook (run 15) |
| `scripts/python/Temporal Difference/selfplay_sf_v1_run16.ipynb` | Archived v1 Stockfish-teacher notebook (run 16) |

## R Analysis Scripts

| File | Purpose |
|------|---------|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END`; shows engine perspective (Win/Loss/Draw) for SF-teacher runs |
| `scripts/r/03_visualize.qmd` | Full cross-run visualization. **Session 23:** default `show_runs` updated to include runs 30 and 31 |
| `scripts/r/04_game_viewer.R` | Shiny app — game viewer with run selector |
| `scripts/r/td_tutorial.qmd` | Tutorial: TD(0) learning via Scholar's Mate walkthrough |
| `scripts/r/negamax_tutorial.qmd` | Tutorial: negamax search with alpha-beta pruning; worked example using live weights |
| `scripts/r/05_move_scorer.R` | Parameterized script — score all legal moves from any FEN at configurable depth/run |
| `scripts/r/05_stockfish_benchmark.R` | Parameterized script — run ELO benchmark vs Stockfish; designed to run as a background job. Config includes `ENGINE_MODULE` parameter (`"engine"` for runs 1–19, `"engine_learnable_material"` for run 20+, `"engine_nn"` for runs 24–26, `"engine_nn_v2"` for Stage 2–5 runs). **Session 17:** 80 half-move limit removed. **Session 21:** logs `seed_run_id`/`seed_game_number`; tries `nn_sf_run_*` checkpoint naming before `nn_run_*`. **Session 23:** added `engine_nn_v2` branch with `N_SIMULATIONS` config; searches `mcts_v*_run_*.pt` and `mcts_sf_v*_run_*.pt` naming |
| `scripts/r/ch08_blackjack.R` | **New (Session 23)** — R code from Rodriguez & Mendes (2018) Ch. 8; dealer hand simulations for Tables 8.1, 8.2, 8.4 |
| `scripts/r/ch08_blackjack.qmd` | **New (Session 23)** — Quarto version of ch08_blackjack.R with explanatory prose, comparison table, and Exercise 13 |
| `scripts/r/seed_supabase.R` | **New (Session 18)** — one-time seeding script: creates `nn_models` table in Supabase; seeds run_id=2 (learnable material from local run_id=21) and run_id=3 (NN from local run_id=24); safe to re-run |
| `scripts/r/05_search_explorer.R` | Shiny app — interactive move scorer; navigate positions, compare move rankings by depth |
| `scripts/r/06_search_mechanics.R` | Step-by-step walkthrough: board evaluation feature-by-feature, negamax depth-by-depth |
| `scripts/r/06_stockfish_arena.R` | Shiny app — Live Game (engine vs Stockfish) and ELO Benchmark read-only viewer (polls DB every 30s); run benchmarks via `05_stockfish_benchmark.R` |
| `scripts/r/07_human_play.R` | Shiny app — Human vs engine with real TD(0) weight updates and post-game learning diagnostics; v2-aware: auto-detects active v2 human run or creates one seeded from `SEED_RUN_ID`; logs `engine_version` and `human_color`. **Session 18:** full NN support — engine_nn option in UI; separate `session_nn_model` / `session_nn_optimizer` reactives; `nn_td_update()` via reticulate; `.pt` checkpoint saved after each human game; Frobenius norm diagnostics (value box, norm vs self-play baseline, session norm history chart); NN-aware live feature table |
| `public_app/app.R` | Crowdsourced human-vs-engine Shiny app for shinyapps.io; Supabase PostgreSQL backend; two conditions (hardcoded material run_id=1, learnable material run_id=2); chessboard.js drag-and-drop board; all DB calls wrapped in tryCatch (game-start crash fixed Session 19). Deploy via `rsconnect::deployApp(appDir="public_app", appName="chess-learning", appId=18172172L, account="dkillian", server="shinyapps.io", forceUpdate=TRUE)` |
| `public_app/engine.py` | Copy of `engine.py` (v2, hardcoded material) for deployment — confirmed identical to `scripts/python/Temporal Difference/engine.py` |
| `public_app/engine_learnable_material.py` | Copy of `engine_learnable_material.py` for deployment — confirmed identical to `scripts/python/Temporal Difference/engine_learnable_material.py` |
| `public_app/engine.py` | Copy of `engine.py` (v2) for deployment alongside `public_app/app.R` |

---

## Planned Tutorials
1. Self-play Python script walkthrough

---

## Assistant Interaction Conventions

### Session-end procedure

Triggered by phrases like "let's end the session", "I'm going to bed", "I need to restart", "wrapping up", etc.
Execute the following steps **in order**, without waiting for individual confirmations:

1. **Close open DB connections** — run `dbDisconnect()` on all open SQLiteConnection objects visible in the R session (`con`, `con2`, `con3`, `con_retro`, `con_bm`, etc.)

2. **Note any running background jobs** — check for active self-play training or benchmark jobs; record their current status in the session log

3. **Update `ChessLearningLog.md`** — append the session entry:
   - Key findings and analytical results
   - New scripts or apps built
   - Bugs found and fixed
   - Overnight jobs scheduled
   - Steps for next session

4. **Update `ChessLearningDocumentation.md`** — refresh any stale sections:
   - Status header (session number, active run)
   - Current Weight State (latest run weights and sign count)
   - Chess Concept Features table (if features changed)
   - Folder Structure and File Descriptions (new/renamed files)
   - Open Questions (resolved or new)
   - Feature Improvement Candidates (status changes)
   - Key Decisions Made (new architectural decisions)

6. **Update `AGENTS.md`** — refresh any stale sections:
   - Current Architecture header (session number)
   - Key Run History table
   - Current Weight State
   - DB schema (table count)
   - R Analysis Scripts table (new files)

7. **Update `GitLog.md`** — append an entry if any git operations were performed this session (commands, outcomes, issues resolved)

8. **Git commit and push**:
   ```bash
   git add -A
   git commit -m "<concise summary of session work>"
   git push
   ```
   If push fails due to file size, follow the filter-branch procedure documented in `GitLog.md`.

### Code edits
- Do not make any edits without explicit instruction from the user
- Once instructed, make all changes without asking for mid-sequence confirmation
- After completing edits, provide a summary of all changes made so the user can revert any or all of them
- `.ipynb` files must be closed in Positron before editing — Positron overwrites external edits on save
- Use Python JSON editing (via bash) for `.ipynb` files; never use the text `edit` tool on notebooks

### Writing style
- Never abbreviate "temporal difference" in headings or at the start of a sentence
- "TD" is acceptable mid-sentence or in column/variable names in code

---

## Coding Conventions

- Use base R pipe `|>` (not magrittr `%>%`)
- R section headings: `# Section name ----` format (required for RStudio/Positron outline)
- No hard-coded file paths; use relative paths from project root
- Use `flextable` for all display tables (not `knitr::kable`)
- SQLite filters using R expressions (e.g., `names()`) must go after `collect()`, not inside `filter()` passed to `tbl()`
