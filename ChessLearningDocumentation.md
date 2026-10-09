# Learning Study — Documentation

**Activity**: Learning Study
**Workspace**: Chess Learning
**Initialized**: September 29, 2026
**Status**: Active — 14-feature engine; run_id=13 complete (212 games, 12-feature); run_id=14 overnight (14-feature, results pending)

---

## Overview

This activity trains a chess engine via self-play and documents the progression of
chess concept acquisition. The goal is not to build the strongest possible engine,
but to build a *transparent* one whose learning process can be measured and visualized.

The central research question is: **Do chess concepts emerge in a predictable sequence
during self-play training, and is that sequence reproducible across independent runs?**

---

## Research Questions

1. Do chess concepts emerge in a predictable sequence during self-play training?
2. Which concepts are learned early (structurally necessary) vs. late (contextually acquired)?
3. Across 30 independent 10-hour training runs, how consistent is the sequence of concept acquisition?
4. Is there variance in *when* a concept emerges, even if the *order* is stable?
5. Does the sequence of learning parallel how human chess players typically develop?
6. **Pawn U-curve hypothesis**: Does the pawn weight dip negative during the phase
   when major-piece weights (queen, rook) are rising sharply, before recovering?
   If this pattern is reproducible across the 30 runs in timing and direction, it suggests
   a structural artifact of the game rather than noise: the engine temporarily learns that
   "having more pawns while the opponent has an extra queen" predicts losing.
   Low cross-run variance in the dip = structural; high variance = noise.
   (Note: with material now hard-coded, this hypothesis applies if material features are
   reintroduced in a future experimental series.)

---

## Hypothesis

The game structure itself imposes a partial ordering on what gets learned:
- You cannot exploit a passed pawn if you are still blundering queens
- You cannot navigate an endgame if piece values are confused
- Positional subtleties only matter once tactics are mostly sound

Therefore, the sequence of concept acquisition should be broadly reproducible
across runs — with early concepts (piece values, basic tactics) showing low variance
in emergence time, and later concepts (pawn structure, coordination) showing higher variance.

Concepts with low cross-run variance are structurally necessary.
Concepts with high cross-run variance are contextually acquired.

---

## Methods

### Engine Architecture

A **linear evaluation function** with hard-coded material and 14 learnable positional features.

```
board position
      ↓
Feature Extractor  →  named positional feature vector (14 features)
      ↓
Evaluator: hard-coded material score + weighted dot product (positional)  →  position score
      ↓
Negamax Search (alpha-beta pruning, depth=3)  →  move selection
```

The positional feature weights *are* the knowledge. Their trajectory over training is the primary data.
Material evaluation is hard-coded via `PIECE_VALUES` (queen=0.9, rook=0.5, bishop=0.3,
knight=0.3, pawn=0.1, scaled to TD target range) and is not part of the learnable weight vector.

### Learning Algorithm

**Temporal Difference (TD(0)) learning** — after each move, the weight vector is updated
using one-step bootstrapping:

- Non-terminal target: `-V(s_{t+1})` (negated successor evaluation)
- Terminal target: actual game outcome (+1 win, 0 draw, −1 loss)

This is the same learning principle used in Tesauro's TD-Gammon (1992), applied to chess.
TD(0) structurally suppresses draw erosion: the draw signal (target=0) only propagates
one step at a time and does not flatten earlier weights.

A per-feature draw learning rate scale (`draw_lr_scales`) is applied to features with
unusually high draw-gradient magnitude. Currently: `king_safety` gets 5% of normal LR
during draws; `center_control` gets 20% (added in run_id=14 after it became the top
draw-gradient feature in run_id=13).

### Move Selection

**Epsilon-greedy** (ε=0.1): with probability 0.1, a random legal move is chosen;
otherwise, the engine plays the negamax-best move. An alternative variant (`selfplay_softmax.ipynb`)
uses softmax sampling over the top-N candidates with temperature τ=0.05.

### Self-Play Loop

The engine plays both sides. After each game, weights are updated and checkpointed
to the database. The cycle repeats continuously for the duration of a training run.

### Experimental Design

- Search depth: 3
- Number of independent runs: 30 (planned)
- Each run starts from zero positional weights
- Concept "emergence" is defined as: weight exceeds a meaningful threshold
  (to be calibrated during early runs)

---

## Chess Concept Features

Material evaluation is **hard-coded** in the evaluator and not part of the learnable weight vector.
The 14 learnable features are all positional:

| Feature Name | Description | Concept Tracked | Phase scaling |
|---|---|---|---|
| `passed_pawn` | Pawns with no opposing pawns on same/adjacent files | Endgame pawn play | — |
| `doubled_pawn` | Pawns on same file (penalty) | Pawn structure weakness | — |
| `isolated_pawn` | Pawns with no friendly pawns on adjacent files (penalty) | Pawn structure weakness | — |
| `backward_pawn` | Pawn whose front square is attacked by enemy pawn with no adjacent support (penalty) | Pawn structure weakness | — |
| `pawn_advancement` | Rank of the most advanced friendly pawn (÷ 5) | Pawn push tendency | × (0.5 + 0.5 × endgame) |
| `king_safety` | Count of attackers near king | King protection | × phase |
| `center_control` | Attacks on + occupation of e4/d4/e5/d5 (÷ 4) | Opening principles | — |
| `rook_open_file` | Rooks on files with no pawns | Rook activity | — |
| `connected_rooks` | Whether both rooks share a rank or file | Piece coordination | — |
| `knight_pst` | PST summed over all knights (rim=−0.50, central=+0.20) (÷ 2) | Knight placement | — |
| `bishop_pair` | Having both bishops vs. opponent | Static piece advantage | — |
| `rook_seventh` | Rooks on the 7th rank (2nd for black) | Rook infiltration | — |
| `piece_development` | Minor pieces off their starting squares | Opening principles | — |
| `mobility` | Squares attacked by all non-king pieces (÷ 30) | Overall piece activity | × (0.5 + 0.5 × phase) |

Features are computed from the perspective of the side to move (positive = good for side to move).
Game phase is a midgame fraction [0, 1] computed by `_game_phase(board)` from weighted piece count ÷ 24 (Q=4, R=2, N/B=1).

---

## Current Weight State (run_id=13, game 212 — last completed run)

Run_id=13 used the 12-feature engine (pre-redesign). Run_id=14 (14-feature engine) is in
progress overnight; weights not yet available.

| Feature | Weight | Status |
|---|---|---|
| `passed_pawn` | +0.087 | ✓ |
| `mobility` | +0.068 | ✓ |
| `rook_open_file` | +0.032 | ✓ |
| `piece_development` | +0.029 | ✓ |
| `rook_seventh` | +0.022 | ✓ |
| `king_safety` | +0.021 | ✓ |
| `center_control` | +0.013 | ✓ |
| `pawn_advancement` | +0.009 | ✓ **correct sign for first time across all runs** |
| `bishop_pair` | +0.007 | ✓ |
| `connected_rooks` | +0.004 | ✓ |
| `doubled_pawn` | −0.006 | ✗ |
| `isolated_pawn` | −0.010 | ✗ |

Correct signs: 10/12. `backward_pawn` and `knight_pst` not present in run_id=13 (added in
Session 11 engine redesign; will appear in run_id=14+).

---

## System Architecture

### Components

| Component | Language | Description |
|---|---|---|
| Rules engine | Python (`python-chess`) | Legal move generation, game state |
| Feature extractor | Python | Board → 12-element named positional feature vector |
| Evaluator | Python | Hard-coded material + weighted dot product over positional features |
| Search | Python | Negamax with alpha-beta pruning (depth=3) |
| Self-play loop | Python | Plays games, logs outcomes |
| TD learner | Python | TD(0) weight updates after each half-move |
| Database logger | Python (`sqlite3`) | Writes games and weights to SQLite |
| Visualization | R (`tidyverse`, `DBI`, `RSQLite`) | Weight trajectories, concept emergence, benchmarking |
| Stockfish interface | Python (`python-chess`) | ELO benchmarking and live play |

### Data Flow

```
Self-play game  →  game outcome + positions
                →  SQLite: games, game_records, feature_stats, position_logs
TD weight update  →  new weights + deltas
                →  SQLite: weights, weight_deltas (checkpointed every 10 games)
Stockfish match  →  game outcome + PGN
                →  SQLite: stockfish_evals, stockfish_games
R reads SQLite  →  weight trajectory plots
                →  concept emergence charts
                →  ELO benchmark viewer
```

### Database Schema

10 tables in `data/chess_learning.db`. The `stockfish_games` table is created by
`05_stockfish_benchmark.R`, not by `setup_database()` in the training notebook.

**`runs`** — one row per training run
| Column | Type | Description |
|---|---|---|
| run_id | INTEGER | Primary key |
| started_at | TEXT | Timestamp |
| hyperparameters | TEXT | JSON: learning rate, search depth, etc. |

**`games`** — one row per game
| Column | Type | Description |
|---|---|---|
| game_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key → runs |
| game_number | INTEGER | Sequential within run |
| outcome | TEXT | 'white', 'black', 'draw' |
| length | INTEGER | Number of half-moves |
| final_material | REAL | Material balance at game end |
| played_at | TEXT | Timestamp |

**`game_records`** — move-by-move PGN records for self-play games
| Column | Type | Description |
|---|---|---|
| record_id | INTEGER | Primary key |
| game_id | INTEGER | Foreign key → games |
| pgn | TEXT | Full PGN text of the game |

**`weights`** — checkpointed weight values
| Column | Type | Description |
|---|---|---|
| weight_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key → runs |
| checkpoint | INTEGER | Game number at checkpoint |
| feature_name | TEXT | Name of the feature |
| weight_value | REAL | Weight value at checkpoint |

**`weight_deltas`** — per-checkpoint weight change magnitudes
| Column | Type | Description |
|---|---|---|
| delta_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key → runs |
| checkpoint | INTEGER | Game number |
| feature_name | TEXT | Name of the feature |
| delta | REAL | Change in weight since last checkpoint |

**`feature_stats`** — per-game mean feature values (from white's perspective)
| Column | Type | Description |
|---|---|---|
| stat_id | INTEGER | Primary key |
| game_id | INTEGER | Foreign key → games |
| feature_name | TEXT | Name of the feature |
| mean_value | REAL | Mean feature value across positions (white-perspective) |

**`position_logs`** — position-level evaluation records
| Column | Type | Description |
|---|---|---|
| log_id | INTEGER | Primary key |
| game_id | INTEGER | Foreign key → games |
| ply | INTEGER | Half-move number |
| evaluation | REAL | Position score at this ply |
| td_error | REAL | TD prediction error |

**`baseline_evals`** — random mover baseline match results (disabled; `BASELINE_EVERY=None`)
| Column | Type | Description |
|---|---|---|
| eval_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key → runs |
| checkpoint | INTEGER | Game number at evaluation |
| wins | INTEGER | Engine wins vs random mover |
| losses | INTEGER | Engine losses vs random mover |
| draws | INTEGER | Draws vs random mover |

**`stockfish_evals`** — aggregate Stockfish benchmark results
| Column | Type | Description |
|---|---|---|
| eval_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key → runs |
| checkpoint | INTEGER | Weight checkpoint evaluated |
| sf_elo | INTEGER | Stockfish UCI_Elo setting |
| wins | INTEGER | Engine wins |
| losses | INTEGER | Engine losses |
| draws | INTEGER | Draws |
| source | TEXT | 'benchmark' or 'live' |
| played_at | TEXT | Timestamp |

**`stockfish_games`** — individual game PGNs from Stockfish matches
| Column | Type | Description |
|---|---|---|
| sg_id | INTEGER | Primary key |
| eval_id | INTEGER | Foreign key → stockfish_evals |
| game_number | INTEGER | Sequential within eval batch |
| pgn | TEXT | Full PGN text |
| outcome | TEXT | 'white', 'black', 'draw' |

---

## Pipeline Description

### Stage 01 — Engine (Python)

Core engine: feature extractor, linear evaluator, negamax search with alpha-beta pruning.

- `engine.py`: importable module used by all training scripts
- `engine.ipynb`: documented notebook version with 6 verification checks

### Stage 02 — Self-Play / Training (Python)

Three variants of the self-play training loop:

- `selfplay_td0.ipynb`: current primary training script — TD(0) bootstrapping, epsilon-greedy, hard-coded material
- `selfplay_softmax.ipynb`: variant — softmax move selection with temperature τ=0.05, style modifier presets
- `selfplay_mc.ipynb`: archived — original Monte Carlo return implementation (superseded)

### Stage 03 — Analysis & Reporting (R)

Visualization and analysis scripts — see File Descriptions below.

---

## Folder Structure

```
Chess Learning/                          ← Repo root
├── AGENTS.md                            ← Project memory for Posit Assistant
├── ActivitySetupGuide.md
├── README.md
├── ChessLearningDocumentation.md        ← This file
├── ChessLearningLog.md                  ← Session log
├── ChessLearningConversations.md        ← Conversation transcript
├── scripts/
│   ├── python/
│   │   ├── engine.py                    ← Importable module (feature extractor, evaluator, search)
│   │   ├── engine.ipynb                 ← Stage 01: documented notebook with verification
│   │   ├── selfplay_td0.ipynb           ← Stage 02: current training script (TD(0))
│   │   ├── selfplay_softmax.ipynb       ← Stage 02 variant: softmax + style modifiers
│   │   └── selfplay_mc.ipynb            ← Stage 02 archive: Monte Carlo version
│   └── r/
│       ├── query self play notebook.R   ← Parameterized assessment tool (set RUN_ID, GAME_START, GAME_END)
│       ├── 03_visualize.qmd             ← Full cross-run Quarto visualization
│       ├── 04_game_viewer.R             ← Shiny: game viewer with source toggle and move analysis
│       ├── td_tutorial.qmd              ← Tutorial: TD(0) learning via Scholar's Mate
│       ├── negamax_tutorial.qmd         ← Tutorial: negamax with alpha-beta; worked example
│       ├── 05_move_scorer.R             ← Parameterized: score all legal moves from any FEN
│       ├── 05_search_explorer.R         ← Shiny: interactive move scorer with board navigation
│       ├── 05_stockfish_benchmark.R     ← Parameterized: ELO benchmark (run as background job)
│       ├── 06_search_mechanics.R        ← Step-by-step: board eval + negamax depth-by-depth walkthrough
│       ├── 06_stockfish_arena.R         ← Shiny: Live Game viewer + ELO benchmark read-only viewer
│       └── 07_human_play.R              ← Shiny: human vs engine with live TD(0) updates
├── data/
│   └── chess_learning.db                ← SQLite database (10 tables)
├── viz/                                 ← Output plots
└── models/                              ← Saved weight checkpoints (future use)
```

---

## File Descriptions

### Python Scripts

| File | Description |
|---|---|
| `scripts/python/engine.py` | Importable module: 14-feature positional extractor, linear evaluator with hard-coded material, negamax search with alpha-beta, `_game_phase()` helper. Used by all training notebooks. |
| `scripts/python/engine.ipynb` | Stage 01 notebook: same content as `engine.py` with markdown explanations and 6 verification checks. Compatible with Positron and Colab. |
| `scripts/python/selfplay_td0.ipynb` | Primary training script: `play_game`, `td_update` (TD(0)), `train`; epsilon-greedy move selection; SQLite logging of all tables. Configure via `RUN_ID`, `DEPTH`, `RESUME`, `N_GAMES`. |
| `scripts/python/selfplay_softmax.ipynb` | Variant training script: softmax move selection (τ=0.05, top-10 candidates), style modifier presets (neutral/attacking/cautious/reckless/positional). RUN_ID=13+. |
| `scripts/python/selfplay_mc.ipynb` | Archived Monte Carlo version of the self-play loop. Superseded by TD(0). Retained for reference. |

### R Scripts

| File | Description |
|---|---|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END` to inspect any run slice: decisive rate, weight table, weight trajectory chart, top deltas. |
| `scripts/r/03_visualize.qmd` | Quarto document: weight trajectories by feature group, game statistics, feature-outcome rolling correlation, concept emergence timeline, final weights table. |
| `scripts/r/04_game_viewer.R` | Shiny app: step through self-play or Stockfish games move by move; "Why this move?" panel with move scoring bar chart and feature breakdown table; search tree visualization (Graphviz). |
| `scripts/r/td_tutorial.qmd` | Quarto tutorial: TD(0) learning explained via Scholar's Mate worked example. |
| `scripts/r/negamax_tutorial.qmd` | Quarto tutorial: negamax with alpha-beta pruning; game tree diagrams (Mermaid); worked example using live run_id=12 weights. |
| `scripts/r/05_move_scorer.R` | Parameterized script: scores all legal moves from any FEN position at configurable depth and run. |
| `scripts/r/05_search_explorer.R` | Shiny app: interactive move scorer; navigate positions, enter FENs, compare move rankings by depth. |
| `scripts/r/05_stockfish_benchmark.R` | Parameterized script: runs ELO benchmark vs Stockfish; designed to run as an RStudio background job. Writes to `stockfish_evals` and `stockfish_games` tables. |
| `scripts/r/06_search_mechanics.R` | Step-by-step R walkthrough: board evaluation feature-by-feature (feature × weight contribution table), negamax depth-by-depth from depth=0 through depth=3. |
| `scripts/r/06_stockfish_arena.R` | Shiny app: Live Game tab (engine vs Stockfish, real-time board) and ELO Benchmark tab (read-only viewer polling DB every 30s). Run benchmarks via `05_stockfish_benchmark.R`. |
| `scripts/r/07_human_play.R` | Shiny app: human vs engine with real TD(0) weight updates; live diagnostics (feature × weight table, evaluation trajectory); post-game analysis (weight delta chart, top-3 learning moments). |

---

## Key Decisions Made

| Decision | Rationale |
|---|---|
| Linear evaluation over neural network | Weights are directly interpretable; concept tracking is transparent |
| TD(0) over Monte Carlo returns | Structurally suppresses draw erosion; terminal draw signal only propagates one step at a time |
| Material hard-coded in evaluator | Removes material erosion problem entirely; isolates positional learning to the weight vector |
| Per-feature draw LR scale (`draw_lr_scales`) | Allows targeted suppression of draw gradient for features with unusually high draw sensitivity (currently: `king_safety`) |
| Depth=3 over depth=2 | Corrects depth-2 artifacts in `connected_rooks`; more tactical lookahead; ~60s/game overhead acceptable |
| Epsilon-greedy (ε=0.1) | Simple, effective exploration; generates decisive games without large overhead |
| `python-chess` for rules | Mature, well-tested library; avoids reimplementing complex rules |
| SQLite for storage | Single-file, portable, readable by R via DBI |
| R for visualization | User preference; excellent for producing publication-quality charts |
| Stockfish 19 for benchmarking | Provides ELO-limited opponents for strength estimation; minimum UCI_Elo = 1320 |
| Background job for Stockfish benchmark | Prevents Shiny session blocking; arena app is now read-only |

---

## Open Questions

- Why do `doubled_pawn` and `isolated_pawn` show wrong signs in run_id=13? Is this a feature design issue, a depth artifact, or genuine noise at 212 games?
- Why did `mobility` surge to dominate the evaluation in run_id=12 (~3× the next feature)? Phase-scaling (added in Session 11) should mitigate this in run_id=14+.
- Does softmax selection (run_id=13) improve over epsilon-greedy in decisive rate and weight convergence? (69.4% decisive in run_id=13 vs 84% in run_id=12 — regression, but confounded by short run length and earlier engine architecture.)
- What ELO does the engine achieve at depth=3? First result: 0W/1L/0D vs Stockfish 1320 (ELO < 1320). Benchmark with 14-feature engine pending.
- What threshold defines concept "emergence"? (Emergence = weight magnitude exceeds threshold consistently across checkpoints; calibration in progress.)
- Is the architecture ready for the 30-run experimental design? The 14-feature engine (run_id=14) is the candidate; assess once results are in.
- **Planned experiment — Stockfish-as-teacher**: Create a training variant where the engine learns from games against Stockfish (at limited ELO) rather than self-play. Adapt `selfplay_td0.ipynb` to alternate colors against Stockfish, keeping TD(0) update logic identical. Could start at ELO 1320 and step up as the engine improves. Key question: does a calibrated external opponent produce faster or more correct weight convergence than self-play? Main cost: slower games than self-play (~30s vs longer Stockfish games).

---

## Feature Improvement Candidates

Ideas to consider before beginning the 30-run experimental series. Changing features
mid-study would break cross-run comparability.

| Feature | Issue | Status |
|---|---|---|
| `center_control` | Previously counted only *attacks* on d4/d5/e4/e5; pawn occupation not credited. | **Resolved** — occupation term added in Session 11 engine redesign; normalize by ÷ 4. |
| `pawn_advancement` | Persistently wrong sign and worsening fast in run_id=12 (−0.103 at game 1200). | **Partially addressed** — redesigned in Session 11 as rank of most advanced pawn only (÷ 5) + endgame phase scaling. Correct sign in run_id=13 for first time. Monitor in run_id=14. |
| `mobility` | Surged to 0.250 (3× next feature) in run_id=12. | **Partially addressed** — phase scaling added in Session 11 (× (0.5 + 0.5 × phase)) to suppress midgame dominance. Monitor in run_id=14. |
| `doubled_pawn` | Wrong sign in run_id=12 and run_id=13. Unclear if feature design, scale, or noise. | **Open** — investigate which game positions drive wrong-signed updates. |
