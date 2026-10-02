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

To resume an activity, ask the assistant to read the Log and Conversations files before proceeding.

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
1. Ask the assistant to read `ChessLearningLog.md` and `ChessLearningConversations.md`
2. Confirm the current development stage before writing code.

---

## Tech Stack

- **Python**: `python-chess` for rules; custom engine for feature extraction, minimax search, TD learning
- **R**: `tidyverse`, `DBI`, `RSQLite` for visualization and analysis
- **Storage**: SQLite database at `Learning Study/data/chess_learning.db`

## Assistant Interaction Conventions

### Code edits
- Do not make any edits without explicit instruction from the user
- Once instructed, make all changes without asking for mid-sequence confirmation
- After completing edits, provide a summary of all changes made so the user can revert any or all of them

### Writing style
- Never abbreviate "temporal difference" in headings or at the start of a sentence
- "TD" is acceptable mid-sentence or in column/variable names in code

---

## Coding Conventions

- Use base R pipe `|>` (not magrittr `%>%`)
- Python scripts numbered by stage: `01_engine.py`, `02_selfplay.py`, etc.
- R scripts numbered by stage: `01_explore.R`, `02_visualize.R`, etc.
- No hard-coded file paths; use relative paths from project root
- Use `flextable` for all display tables (not `knitr::kable`)
