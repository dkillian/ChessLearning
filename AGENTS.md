# Chess Learning — Project Memory

## Workspace Overview

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
- **Storage**: SQLite database at `data/chess_learning.db` — 10 tables (runs, games, game_records, weights, weight_deltas, feature_stats, position_logs, baseline_evals, stockfish_evals, stockfish_games); `stockfish_games` created by `05_stockfish_benchmark.R`, not by `setup_database()`
- **Stockfish 19**: installed via winget; used for ELO benchmarking and live game play; minimum `UCI_Elo` = 1320

---

## Current Architecture (as of Session 10)

### Engine (`scripts/python/engine.py`)
- **12 learnable positional features** (material features removed entirely)
- **Material evaluation hard-coded** in `evaluate()` via `PIECE_VALUES` constant
  (queen=0.9, rook=0.5, bishop=0.3, knight=0.3, pawn=0.1 — scaled to TD target range)
- Search: negamax with alpha-beta pruning

### Learnable Features
Pawn structure: `passed_pawn`, `doubled_pawn`, `isolated_pawn`, `pawn_advancement`
King safety: `king_safety`
Activity: `center_control`, `rook_open_file`, `connected_rooks`
Coordination: `bishop_pair`, `rook_seventh`, `piece_development`, `mobility`

### Training Script (`scripts/python/selfplay_td0.ipynb`)
- **Temporal difference (0) bootstrapping**: target for position t = `-V(s_{t+1})`; terminal uses actual outcome
- **Per-feature draw LR scales**: `draw_lr_scales = {'king_safety': 0.05}` — king_safety gets 5% LR during draws
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
| 13 | planned | 3 | Softmax selection (τ=0.05, top-10); style modifiers; `selfplay_softmax.ipynb` | TBD |

### Current Weight State (run_id=12, game 1200)
Correct sign (9/12): mobility (+0.250 — dominating), passed_pawn, rook_open_file, rook_seventh,
king_safety, piece_development, connected_rooks, bishop_pair, isolated_pawn
Wrong sign (3/12): doubled_pawn (−0.044), pawn_advancement (−0.103), center_control (−0.001, newly wrong)
Note: `mobility` at 0.250 is ~3× the next feature — structural concern for feature design

### Stockfish Integration
- **Stockfish 19** installed via `winget install Stockfish.Stockfish`
- Auto-detected at: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Stockfish.Stockfish_*\stockfish\stockfish-windows-x86-64-universal.exe`
- Accessible via `chess.engine.SimpleEngine.popen_uci(SF_PATH)` in python-chess
- ELO limiting: `sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})` — minimum is 1320 for Stockfish 19
- Always pass `ponder=False` to `sf.play()` to prevent UCI communication hangs
- `stockfish_evals` DB table logs aggregate match results (source = `'benchmark'` or `'live'`)
- `stockfish_games` DB table logs individual game PGNs (created by `05_stockfish_benchmark.R`)
- First benchmark result: engine lost to Stockfish 1320 (0W/1L/0D) — estimated ELO < 1320

---

## R Analysis Scripts

| File | Purpose |
|------|---------|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END` |
| `scripts/r/03_visualize.qmd` | Full cross-run visualization |
| `scripts/r/04_game_viewer.R` | Shiny app — game viewer with run selector |
| `scripts/r/td_tutorial.qmd` | Tutorial: TD(0) learning via Scholar's Mate walkthrough |
| `scripts/r/negamax_tutorial.qmd` | Tutorial: negamax search with alpha-beta pruning; worked example using live weights |
| `scripts/r/05_move_scorer.R` | Parameterized script — score all legal moves from any FEN at configurable depth/run |
| `scripts/r/05_stockfish_benchmark.R` | Parameterized script — run ELO benchmark vs Stockfish; designed to run as a background job so the console stays free |
| `scripts/r/05_search_explorer.R` | Shiny app — interactive move scorer; navigate positions, compare move rankings by depth |
| `scripts/r/06_search_mechanics.R` | Step-by-step walkthrough: board evaluation feature-by-feature, negamax depth-by-depth |
| `scripts/r/06_stockfish_arena.R` | Shiny app — Live Game (engine vs Stockfish) and ELO Benchmark read-only viewer (polls DB every 30s); run benchmarks via `05_stockfish_benchmark.R` |
| `scripts/r/07_human_play.R` | Shiny app — Human vs engine with real TD(0) weight updates and post-game learning diagnostics |

---

## Planned Tutorials
1. Self-play Python script walkthrough

---

## Assistant Interaction Conventions

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
