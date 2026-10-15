# Learning Study — Documentation

**Activity**: Learning Study
**Workspace**: Chess Learning
**Initialized**: September 29, 2026
**Status**: Active — Session 19; neural net condition (run_id=24, 100+ games overnight); learnable material (run_id=21, 1050 games); hardcoded material baseline through run_id=19; public app deployed and crash-fixed.

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
   (Applies to the learnable-material condition; not observable in hardcoded-material runs.)

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

### Engine Conditions

Two engine conditions are maintained as separate importable modules:

| Condition | Module | Runs | Features |
|---|---|---|---|
| **Hardcoded material** | `engine.py` / `engine_hardcoded_material.py` | 1–19 | 14 positional; material hard-coded via `PIECE_VALUES` |
| **Learnable material** | `engine_learnable_material.py` | 20+ | 19 total (5 material + 14 positional); no hard-coded piece values |

The learnable material condition replaces `knight_pst` with `knight_outpost` starting run_id=21.

### Engine Architecture

```
board position
      ↓
Feature Extractor  →  named feature vector (14 or 19 features depending on condition)
      ↓
Evaluator: [hard-coded material score +] weighted dot product over features  →  position score
      ↓
Negamax Search (alpha-beta pruning, depth=3)  →  move selection
```

The feature weights *are* the knowledge. Their trajectory over training is the primary data.

**Hardcoded material condition**: Material evaluated via `PIECE_VALUES` constant
(queen=0.9, rook=0.5, bishop=0.3, knight=0.3, pawn=0.1, scaled to TD target range).
Material is not part of the learnable weight vector.

**Learnable material condition**: No `PIECE_VALUES` constant. All five piece types have
learnable weights alongside the 14 positional features (19 total).

### Sign Convention (v2 engine, runs 18+)

All features — including penalty features — are oriented so that a **higher value is always
better** for the side to move. Penalty features (`doubled_pawn`, `isolated_pawn`,
`backward_pawn`) are negated in `extract_features()`: stored as (opponent_count − my_count).
The correct learned weight is **positive for all features** in v2. Any negative weight is wrong.

(v1 engine, runs 1–17: penalty features were not negated; correct signs for those features
were negative under v1 convention.)

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
draw-gradient feature in run_id=13). Draw gradients confirmed dramatically smaller in
v2 engine — these scales may not be needed but are retained for safety.

### Move Selection

**Epsilon-greedy** (ε=0.1): with probability 0.1, a random legal move is chosen;
otherwise, the engine plays the negamax-best move. An alternative variant (`selfplay_softmax.ipynb`)
uses softmax sampling over the top-N candidates with temperature τ=0.05.

### Self-Play Loop

The engine plays both sides. After each game, weights are updated and checkpointed
to the database. The cycle repeats continuously for the duration of a training run.

### Experimental Design

- Search depth: 3
- Number of independent runs: 30 (planned, per condition)
- Each run starts from zero weights
- Concept "emergence" defined as: weight exceeds a meaningful threshold
  (to be calibrated during early runs)

---

## Chess Concept Features

### Hardcoded Material Engine (`engine.py`, runs 1–19) — 14 learnable positional features

Material evaluation is **hard-coded** and not part of the learnable weight vector.

| Feature Name | Description | Concept Tracked | Phase scaling |
|---|---|---|---|
| `passed_pawn` | Pawns with no opposing pawns on same/adjacent files | Endgame pawn play | — |
| `doubled_pawn` | Pawns on same file (penalty; negated) | Pawn structure weakness | — |
| `isolated_pawn` | Pawns with no friendly pawns on adjacent files (penalty; negated) | Pawn structure weakness | — |
| `backward_pawn` | Pawn whose stop square is controlled by an enemy **pawn** sentry + no adjacent friendly pawn support (penalty; negated) | Pawn structure weakness | — |
| `pawn_advancement` | Total rank advancement of **all** pawns from starting rank (÷ 20) | Pawn push tendency | × (0.5 + 0.5 × endgame) |
| `king_safety` | Pawn shelter (friendly pawns 1–2 ranks ahead of king) minus attacker count in king zone (÷ 3) | King protection | × phase |
| `center_control` | Attacks on + occupation of e4/d4/e5/d5 (÷ 4) | Opening principles | — |
| `rook_open_file` | Rooks on files with no pawns | Rook activity | — |
| `connected_rooks` | Whether both rooks share a rank or file | Piece coordination | — |
| `knight_pst` | PST summed over all knights (rim=−0.50, central=+0.20) (÷ 2) | Knight placement | — |
| `bishop_pair` | Having both bishops vs. opponent | Static piece advantage | — |
| `rook_seventh` | Rooks on the 7th rank (2nd for black) | Rook infiltration | — |
| `piece_development` | Minor pieces off their starting squares | Opening principles | — |
| `mobility` | Squares attacked by all non-king pieces (÷ 30) | Overall piece activity | × (0.5 + 0.5 × phase) |

### Learnable Material Engine (`engine_learnable_material.py`, runs 20+) — 19 learnable features

`knight_pst` replaced by `knight_outpost` starting run_id=21.

**5 material features** (no hard-coded piece values):

| Feature Name | Description |
|---|---|
| `material_queen` | Queen count differential (my − opponent) |
| `material_rook` | Rook count differential |
| `material_bishop` | Bishop count differential |
| `material_knight` | Knight count differential |
| `material_pawn` | Pawn count differential |

**14 positional features**: same as hardcoded material engine, except `knight_pst` → `knight_outpost`:

| Feature Name | Description | Change from hardcoded condition |
|---|---|---|
| `knight_outpost` | Knights in opponent's half not attacked by any opponent pawn (safe outpost knights) | Replaces `knight_pst` starting run_id=21 |
| *(all others)* | Same as hardcoded material engine above | — |

Features are computed from the perspective of the side to move (positive = good for side to move).
Game phase is a midgame fraction [0, 1] computed by `_game_phase(board)` from weighted piece count ÷ 24 (Q=4, R=2, N/B=1).

---

## Current Weight State

### run_id=21 (learnable material, `knight_outpost`, 1050 games from zero — v2 sign convention: all positive correct)

Correct sign (17/19): material_rook (+0.213), material_queen (+0.184), material_bishop (+0.112),
mobility (+0.074), material_knight (+0.060), center_control (+0.052), material_pawn (+0.045),
rook_seventh (+0.037), king_safety (+0.029), knight_outpost (+0.025), passed_pawn (+0.024),
connected_rooks (+0.010), piece_development (+0.008), isolated_pawn (+0.005), backward_pawn (+0.005),
rook_open_file (+0.002), bishop_pair (+0.001)

Wrong sign (2/19): doubled_pawn (−0.007), pawn_advancement (−0.031)

Material ordering fully correct (rook > queen > bishop > knight > pawn). `knight_outpost`
solidly positive throughout. `pawn_advancement` structural credit assignment issue;
`doubled_pawn` volatile random-walk. Black asymmetry 1.10:1 — best in any learnable-material run.

Benchmark (game 1040, 20 games vs SF 1320): 3W/5L/12D — estimated ELO ~1285. Note: 60% of
benchmark games hit the 80 half-move cap (now removed — next benchmark will use chess rules only).

### Historical weight state snapshots

See `AGENTS.md` (Current Weight State section) for full weight tables for run_id=17, 18, 19, 20.

---

## System Architecture

### Components

| Component | Language | Description |
|---|---|---|
| Rules engine | Python (`python-chess`) | Legal move generation, game state |
| Feature extractor | Python | Board → 14-feature (hardcoded material) or 19-feature (learnable material) named vector |
| Evaluator | Python | Hard-coded material + weighted dot product (hardcoded cond.) or fully weighted dot product (learnable cond.) |
| Search | Python | Negamax with alpha-beta pruning (depth=3) |
| Self-play loop | Python | Plays games, logs outcomes |
| TD learner | Python | TD(0) weight updates after each half-move |
| Database logger | Python (`sqlite3`) | Writes games and weights to SQLite |
| Visualization | R (`tidyverse`, `DBI`, `RSQLite`) | Weight trajectories, concept emergence, benchmarking |
| Stockfish interface | Python (`python-chess`) | ELO benchmarking (minimum UCI_Elo = 1320 in Stockfish 19) |

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

Two engine modules maintained as first-class research artifacts (one per condition):

- `engine.py` (`engine_hardcoded_material.py`): 14 positional features; material hard-coded
- `engine_learnable_material.py`: 19 features (5 material + 14 positional); `knight_outpost` replaces `knight_pst` from run_id=21

### Stage 02 — Self-Play / Training (Python)

Three active training notebooks, one per condition/variant:

- `selfplay_td0.ipynb`: hardcoded material condition; TD(0), epsilon-greedy (runs 7–19)
- `selfplay_learnable_material.ipynb`: learnable material condition; TD(0), epsilon-greedy (runs 20+)
- `selfplay_softmax.ipynb`: variant — softmax move selection with temperature τ=0.05, style modifier presets

Archived notebooks (v1 engine, prior runs):
- `selfplay_mc.ipynb` — Monte Carlo return implementation (superseded by TD(0))
- `selfplay_sf.ipynb` — Stockfish-teacher variant (run_id=16; not continued)
- `selfplay_td0_v1_runs1-14.ipynb`, `selfplay_softmax_v1_run15.ipynb`, `selfplay_sf_v1_run16.ipynb` — v1 archives

### Stage 03 — Analysis & Reporting (R)

Visualization and analysis scripts — see File Descriptions below.

---

## Folder Structure

```
Chess Learning/                                  ← Repo root
├── AGENTS.md                                    ← Project memory for Posit Assistant
├── ActivitySetupGuide.md
├── README.md
├── ChessLearningDocumentation.md                ← This file
├── ChessLearningLog.md                          ← Session log
├── ChessLearningConversations.md                ← Conversation transcript
├── ChessLearningWorkPlan.md                     ← Multi-session strategic work plan
├── scripts/
│   ├── python/
│   │   ├── engine.py                            ← Hardcoded material engine (14 positional features)
│   │   ├── engine_hardcoded_material.py         ← Archive copy of hardcoded material engine (runs 1–19)
│   │   ├── engine_learnable_material.py         ← Learnable material engine (19 features; knight_outpost)
│   │   ├── engine.ipynb                         ← Stage 01: documented notebook with verification
│   │   ├── selfplay_td0.ipynb                   ← Stage 02: hardcoded material training (TD(0))
│   │   ├── selfplay_learnable_material.ipynb    ← Stage 02: learnable material training (TD(0))
│   │   ├── selfplay_softmax.ipynb               ← Stage 02 variant: softmax + style modifiers
│   │   ├── selfplay_mc.ipynb                    ← Archive: Monte Carlo version (superseded)
│   │   ├── selfplay_sf.ipynb                    ← Archive: Stockfish-teacher variant (not continued)
│   │   ├── selfplay_td0_v1_runs1-14.ipynb       ← Archive: v1 TD(0) notebook
│   │   ├── selfplay_softmax_v1_run15.ipynb      ← Archive: v1 softmax notebook
│   │   └── selfplay_sf_v1_run16.ipynb           ← Archive: v1 Stockfish-teacher notebook
│   └── r/
│       ├── query self play notebook.R           ← Parameterized assessment tool
│       ├── 03_visualize.qmd                     ← Full cross-run Quarto visualization
│       ├── 04_game_viewer.R                     ← Shiny: game viewer with source toggle and move analysis
│       ├── td_tutorial.qmd                      ← Tutorial: TD(0) learning via Scholar's Mate
│       ├── negamax_tutorial.qmd                 ← Tutorial: negamax with alpha-beta; worked example
│       ├── 05_move_scorer.R                     ← Parameterized: score all legal moves from any FEN
│       ├── 05_search_explorer.R                 ← Shiny: interactive move scorer with board navigation
│       ├── 05_stockfish_benchmark.R             ← Parameterized: ELO benchmark (run as background job)
│       ├── 06_search_mechanics.R                ← Step-by-step: board eval + negamax depth-by-depth
│       ├── 06_stockfish_arena.R                 ← Shiny: Live Game viewer + ELO benchmark read-only viewer
│       └── 07_human_play.R                      ← Shiny: human vs engine with live TD(0) updates
├── public_app/
│   ├── app.R                                    ← Crowdsourced human-vs-engine app (shinyapps.io)
│   └── engine.py                                ← Copy of engine.py for deployment
├── data/
│   └── chess_learning.db                        ← SQLite database (10 tables)
├── viz/                                         ← Output plots
└── models/                                      ← Saved weight checkpoints (future use)
```

---

## File Descriptions

### Python Scripts

| File | Description |
|---|---|
| `scripts/python/engine.py` | Importable module (hardcoded material): 14-feature positional extractor, linear evaluator with hard-coded material (`PIECE_VALUES`), negamax search with alpha-beta, `_game_phase()` helper. Used by hardcoded-material training notebooks. |
| `scripts/python/engine_hardcoded_material.py` | Archive copy of `engine.py` as of the hardcoded-material condition (runs 1–19). Preserved as a research artifact. |
| `scripts/python/engine_learnable_material.py` | Learnable material engine: 19 features — 5 material + 14 positional. `knight_pst` replaced by `knight_outpost` (`_safe_knight_outpost()`) from run_id=21. No hard-coded piece values. |
| `scripts/python/engine.ipynb` | Stage 01 notebook: same content as `engine.py` (v2 architecture) with markdown explanations and 6 verification checks. |
| `scripts/python/selfplay_td0.ipynb` | Primary training script (hardcoded material): `play_game`, `td_update` (TD(0)), `train`; epsilon-greedy; SQLite logging. Configure via `RUN_ID`, `DEPTH`, `RESUME`, `N_GAMES`. |
| `scripts/python/selfplay_learnable_material.ipynb` | Training script (learnable material): same structure as `selfplay_td0.ipynb`; imports `engine_learnable_material`. Configure via `RUN_ID`, `DEPTH`, `RESUME`, `N_GAMES`. |
| `scripts/python/selfplay_softmax.ipynb` | Variant training script: softmax move selection (τ=0.05, top-10 candidates), style modifier presets (neutral/attacking/cautious/reckless/positional). |
| `scripts/python/selfplay_sf.ipynb` | Stockfish-teacher variant (not continued since run_id=16): plays against Stockfish at configurable ELO, TD(0) updates, built-in ELO estimation. Windows asyncio fix included. Retained for reference. |
| `scripts/python/selfplay_mc.ipynb` | Archived Monte Carlo version of the self-play loop. Superseded by TD(0). Retained for reference. |
| `scripts/python/selfplay_td0_v1_runs1-14.ipynb` | Archived v1 TD(0) training notebook (runs 1–14). |
| `scripts/python/selfplay_softmax_v1_run15.ipynb` | Archived v1 softmax training notebook (run 15). |
| `scripts/python/selfplay_sf_v1_run16.ipynb` | Archived v1 Stockfish-teacher notebook (run 16). |

### R Scripts

| File | Description |
|---|---|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END` to inspect any run slice: decisive rate, weight table, weight trajectory chart, top deltas, engine-perspective outcomes (for SF-teacher runs). |
| `scripts/r/03_visualize.qmd` | Quarto document: weight trajectories by feature group, game statistics, feature-outcome rolling correlation, concept emergence timeline, final weights table. `SHOW_RUNS` Quarto param controls which runs to include. |
| `scripts/r/04_game_viewer.R` | Shiny app: step through self-play or Stockfish games move by move; "Why this move?" panel with move scoring bar chart and feature breakdown table; search tree visualization (Graphviz). |
| `scripts/r/td_tutorial.qmd` | Quarto tutorial: TD(0) learning explained via Scholar's Mate worked example. |
| `scripts/r/negamax_tutorial.qmd` | Quarto tutorial: negamax with alpha-beta pruning; game tree diagrams (Mermaid); worked example using live run_id=12 weights. |
| `scripts/r/05_move_scorer.R` | Parameterized script: scores all legal moves from any FEN position at configurable depth and run. |
| `scripts/r/05_search_explorer.R` | Shiny app: interactive move scorer; navigate positions, enter FENs, compare move rankings by depth. |
| `scripts/r/05_stockfish_benchmark.R` | Parameterized script: runs ELO benchmark vs Stockfish at configurable ELO; designed to run as an RStudio background job. `ENGINE_MODULE` config parameter selects engine condition (`"engine"` for runs 1–19, `"engine_learnable_material"` for run 20+). No move limit — games play to natural conclusion under chess rules only (`claim_draw=True`). Writes to `stockfish_evals` and `stockfish_games` tables. |
| `scripts/r/06_search_mechanics.R` | Step-by-step R walkthrough: board evaluation feature-by-feature (feature × weight contribution table), negamax depth-by-depth from depth=0 through depth=3. |
| `scripts/r/06_stockfish_arena.R` | Shiny app: Live Game tab (engine vs Stockfish, real-time board) and ELO Benchmark tab (read-only viewer polling DB every 30s). Run benchmarks via `05_stockfish_benchmark.R`. |
| `scripts/r/07_human_play.R` | Shiny app: human vs engine with real TD(0) weight updates; engine/run/checkpoint selection menu; live diagnostics (feature × weight table, evaluation trajectory); post-game analysis (weight delta chart, top-3 learning moments). v2-aware: auto-detects active v2 human run or creates one seeded from `SEED_RUN_ID`. Logs `engine_version` and `human_color`. |

### Public App

| File | Description |
|---|---|
| `public_app/app.R` | Crowdsourced human-vs-engine Shiny app for shinyapps.io. Supabase PostgreSQL backend. Two conditions: hardcoded material (run_id=1) and learnable material (run_id=2). Chessboard.js drag-and-drop board. All DB calls wrapped in `tryCatch` — game-start crash fixed in Session 19. |
| `public_app/engine.py` | Copy of `engine.py` (v2, hardcoded material) for deployment alongside `public_app/app.R`. |
| `public_app/engine_learnable_material.py` | Copy of `engine_learnable_material.py` for deployment. |

---

## Key Decisions Made

| Decision | Rationale |
|---|---|
| Linear evaluation over neural network | Weights are directly interpretable; concept tracking is transparent |
| TD(0) over Monte Carlo returns | Structurally suppresses draw erosion; terminal draw signal only propagates one step at a time |
| Material hard-coded in evaluator (baseline condition) | Removes material erosion problem entirely; isolates positional learning to the weight vector |
| Learnable material (experimental condition) | Tests whether the engine can discover piece values on its own; first-class research question |
| Per-feature draw LR scale (`draw_lr_scales`) | Targeted suppression of draw gradient for features with unusually high draw sensitivity (currently: `king_safety`=0.05, `center_control`=0.2) |
| Depth=3 over depth=2 | Corrects depth-2 artifacts in `connected_rooks`; more tactical lookahead; ~60s/game overhead acceptable |
| Epsilon-greedy (ε=0.1) | Simple, effective exploration; generates decisive games without large overhead |
| `python-chess` for rules | Mature, well-tested library; avoids reimplementing complex rules |
| SQLite for storage | Single-file, portable, readable by R via DBI |
| R for visualization | User preference; excellent for producing publication-quality charts |
| Stockfish 19 for benchmarking only | SF-teacher experiment (run_id=16, 560 games) produced ELO regression (1267→1212) and sign regression (11/14→7/14); Stockfish plays tactically complex positions that a 14-feature linear evaluator cannot parse reliably. SF reserved for ELO benchmarking. Not formally closed off as a future approach. |
| Background job for Stockfish benchmark | Prevents Shiny session blocking; arena app is now read-only (polls DB every 30s) |
| No move limit in benchmark | 80 half-move cap distorted 60% of run_id=21 benchmark games into draws; removed in Session 17 |
| `knight_pst` → `knight_outpost` (learnable material) | PST approach generated 75.8% wrong-direction TD updates because centralized knights were frequently captured at depth=3; safe outpost definition avoids this |
| One engine module per condition | Engine files are first-class research artifacts; `engine.py`, `engine_hardcoded_material.py`, `engine_learnable_material.py` maintained separately |
| v2 sign convention: all features positive | Penalty features negated in `extract_features()` — cleaner to assess, fewer sign-tracking errors |

---

## Open Questions

- Why do `doubled_pawn` and `pawn_advancement` show wrong signs persistently across runs (including run_id=21)?
  - `doubled_pawn`: magnitude asymmetry — 54% of updates are correct-direction but a fat negative tail overwhelms; near-random walk behavior
  - `pawn_advancement`: frequency bias — 61% of updates are negative; credit assignment problem where advanced pawns lead to exchanges the engine mis-evaluates
- Is the architecture ready for the 30-run experimental design? Under the hardcoded material condition, run_id=14 (11/14, ELO ~1390) and run_id=18 (11/14, ELO ~1355) are candidate baselines. The 2 remaining wrong-sign features may reflect irreducible feature-design limitations rather than insufficient training.
- Will the learnable material condition close the ELO gap to the hardcoded material baseline (~1355) with more training? run_id=21 at ~1285 ELO suggests material weights haven't fully converged.
- Can a neural net engine (MLP with same 19 features, PyTorch) outperform the linear evaluator? Planned as the next development phase.
- Does the sequence of concept acquisition in the learnable material condition differ from the hardcoded material condition? If so, which concepts shift in order?
- What threshold defines concept "emergence"? (Calibration in progress.)
- Does human game signal improve wrong-sign feature correction beyond self-play alone? run_id=17 and 19 maintained 11/14 correct signs but did not correct persistent wrong-sign features.

---

## Feature Improvement Candidates

| Feature | Issue | Status |
|---|---|---|
| `center_control` | Previously counted only *attacks* on d4/d5/e4/e5; pawn occupation not credited. | **Resolved** — occupation term added in Session 11; normalize by ÷ 4. |
| `pawn_advancement` | Persistently wrong sign across all runs. Root cause: credit assignment — advanced pawns lead to captures the engine mis-evaluates. | **Partial** — v2 redesign (Session 14) changed to total advancement of all pawns (÷ 20). Sign improved early in run_id=18 but drifted negative at scale. Structural issue; may require a different feature formulation or reward shaping. |
| `king_safety` | Old design (safe adjacent squares) was a weak proxy; didn't capture pawn shelter or attacking piece threats. | **Resolved** — v2 redesign (Session 14): pawn shelter minus attacker count in king zone; ÷ 3; phase-scaled. Correct sign stable across v2 runs. |
| `backward_pawn` | Old definition used any attacking piece as the sentry; too broad. | **Resolved** — v2 redesign (Session 14): requires an enemy **pawn** to control the stop square (standard chess definition). Correct sign in v2 runs. |
| `mobility` | Surged to 0.250 (3× next feature) in run_id=12. | **Resolved** — phase scaling (× (0.5 + 0.5 × phase)) added in Session 11; 0.074 in run_id=21. |
| `knight_pst` | PST rewards centralized knights that get captured at depth=3; 75.8% wrong-direction TD updates in run_id=20 diagnostic. | **Resolved (learnable material condition)** — replaced by `knight_outpost` (safe outpost knights in opponent's half not attacked by opponent pawns) in Session 16; `knight_outpost` solidly positive (+0.025) at run_id=21 game 1050. `engine.py` (hardcoded material) retains `knight_pst`. |
| `doubled_pawn` | Volatile random-walk behavior; fat negative tail overwhelms 54% correct-direction updates. | **Monitoring** — correct sign in some runs (run_id=14, 17); wrong in run_id=18, 21. No feature redesign attempted yet. |
