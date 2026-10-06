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
- **Storage**: SQLite database at `data/chess_learning.db`

---

## Current Architecture (as of Session 8)

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
- `selfplay_mc.ipynb` preserved as Monte Carlo archive

### Key Run History
| run_id | Games | Depth | Key change | Result |
|--------|-------|-------|------------|--------|
| 1–4 | various | 2 | MC; various fixes | Diagnosed draw equilibrium |
| 5–6 | 300 | 2 | MC + draw_lr_scale | 18–26% decisive; continued erosion |
| 7 | 600 | 2 | TD(0) | Erosion reduced; slow linear drift |
| 10 | 50 | 2 | TD(0) + frozen material | Zero erosion confirmed |
| 11 | 850 | 2 | Hard-coded material; 12 positional features | **78% decisive; all 8 significant signs correct** |
| 12 | in progress | 3 | Depth=3; king_safety draw_lr=0.05 | Running at session close |

### Current Weight State (run_id=11, ~850 games)
Top positional weights (correct sign): mobility, passed_pawn, rook_open_file, isolated_pawn, bishop_pair, rook_seventh, king_safety
Persistently wrong sign (depth-2 artifacts): doubled_pawn, pawn_advancement, connected_rooks

---

## R Analysis Scripts

| File | Purpose |
|------|---------|
| `scripts/r/query self play notebook.R` | Parameterized assessment tool — set `RUN_ID`, `GAME_START`, `GAME_END` |
| `scripts/r/03_visualize.qmd` | Full cross-run visualization |
| `scripts/r/04_game_viewer.R` | Shiny app — game viewer with run selector |
| `scripts/r/td_tutorial.qmd` | Tutorial: TD(0) learning via Scholar's Mate walkthrough |

---

## Planned Tutorials
1. Self-play Python script walkthrough
2. Negamax with alpha-beta pruning (use depth=3 for the illustrative game tree)

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
