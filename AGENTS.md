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

## Current Architecture (as of Session 12)

### Engine (`scripts/python/engine.py`)
- **12 learnable positional features** (material features removed entirely)
- **Material evaluation hard-coded** in `evaluate()` via `PIECE_VALUES` constant
  (queen=0.9, rook=0.5, bishop=0.3, knight=0.3, pawn=0.1 — scaled to TD target range)
- Search: negamax with alpha-beta pruning

### Learnable Features (14 total)
Pawn structure: `passed_pawn`, `doubled_pawn`, `isolated_pawn`, `backward_pawn`, `pawn_advancement`
King safety: `king_safety` (phase-scaled × phase)
Activity: `center_control` (attacks + occupation), `rook_open_file`, `connected_rooks`
Coordination: `knight_pst`, `bishop_pair`, `rook_seventh`, `piece_development`, `mobility` (phase-scaled × (0.5+0.5×phase))
Pawn advancement: `pawn_advancement` (most advanced pawn; endgame-scaled × (0.5+0.5×end))

### Game Phase Weighting
`_game_phase(board)` returns midgame fraction [0,1] from weighted piece count / 24 (Q=4, R=2, N/B=1).
Applied to: `king_safety` (× phase), `mobility` (× (0.5+0.5×phase)), `pawn_advancement` (× (0.5+0.5×end)).

### Training Script (`scripts/python/selfplay_td0.ipynb`)
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
| 16 | 60+ | 3 | **Stockfish teacher** (SF ELO=1320); `selfplay_sf.ipynb`; move_limit=80 | ELO 1212→1267 over 60 games; no black/white asymmetry; 500-game overnight tranche running |

### Current Weight State (run_id=15, game 60 — most recent self-play run)
Correct sign (11/14): passed_pawn (+0.045), king_safety (+0.031), piece_development (+0.027),
mobility (+0.025), rook_open_file (+0.019), center_control (+0.016), backward_pawn (+0.012),
bishop_pair (+0.006), pawn_advancement (+0.005), rook_seventh (+0.003), connected_rooks (+0.002)
Wrong sign (3/14): knight_pst (−0.002), isolated_pawn (−0.006), doubled_pawn (−0.009)
Note: run_id=16 (Stockfish teacher) has same sign pattern but ~10× smaller magnitudes at 60 games;
500-game overnight tranche in progress — weights will be more developed by next session

### Stockfish Integration
- **Stockfish 19** installed via `winget install Stockfish.Stockfish`
- Auto-detected at: `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Stockfish.Stockfish_*\stockfish\stockfish-windows-x86-64-universal.exe`
- Accessible via `chess.engine.SimpleEngine.popen_uci(SF_PATH)` in python-chess
- ELO limiting: `sf.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})` — minimum is 1320 for Stockfish 19
- Always pass `ponder=False` to `sf.play()` to prevent UCI communication hangs
- `stockfish_evals` DB table logs aggregate match results (source = `'benchmark'` or `'live'`)
- `stockfish_games` DB table logs individual game PGNs (created by `05_stockfish_benchmark.R`)
- **Session 12 benchmark (run_id=14, 400 games): 3W/1L/6D at ELO 1320 — estimated ELO ~1428**
- `stockfish_evals` also receives ELO estimates from `selfplay_sf.ipynb` (source = `'sf_teacher'`)
- Windows asyncio fix required in Jupyter: `asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())` before `popen_uci()`

---

## Python Training Scripts

| File | Purpose |
|------|---------|
| `scripts/python/engine.py` | Importable module: 14-feature positional extractor, linear evaluator with hard-coded material, negamax search with alpha-beta, `_game_phase()` helper |
| `scripts/python/selfplay_td0.ipynb` | TD(0) self-play training; epsilon-greedy; current runs: 14 |
| `scripts/python/selfplay_softmax.ipynb` | TD(0) self-play; softmax move selection (τ=0.05, top-10); current runs: 15 |
| `scripts/python/selfplay_sf.ipynb` | **Stockfish teacher**: plays vs Stockfish at configurable ELO; TD(0) updates; built-in ELO estimation (clean eval batch, no weight updates); Windows asyncio fix included; current runs: 16 |

## R Analysis Scripts

| File | Purpose |
|------|---------|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END`; shows engine perspective (Win/Loss/Draw) for SF-teacher runs |
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

4. **Update `AGENTS.md`** — refresh any stale sections:
   - Current Architecture header (session number)
   - Key Run History table
   - Current Weight State
   - DB schema (table count)
   - R Analysis Scripts table (new files)

5. **Update `GitLog.md`** — append an entry if any git operations were performed this session (commands, outcomes, issues resolved)

6. **Git commit and push**:
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
