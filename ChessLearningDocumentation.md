# Learning Study — Documentation

**Activity**: Learning Study
**Workspace**: Chess Learning
**Initialized**: September 29, 2026
**Status**: Stages 01–03 complete. Ready for full training run.

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
6. **Pawn U-curve hypothesis**: Does `material_pawn` weight dip negative during the phase
   when major-piece weights (queen, rook) are rising sharply, before recovering?
   If this pattern is reproducible across the 30 runs in timing and direction, it suggests
   a structural artifact of the game rather than noise: the engine temporarily learns that
   "having more pawns while the opponent has an extra queen" predicts losing.
   Low cross-run variance in the dip = structural; high variance = noise.

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

A **linear evaluation function** with explicit chess concept features.

```
board position
      ↓
Feature Extractor  →  named feature vector
      ↓
Weighted dot product  →  position score
      ↓
Minimax Search (alpha-beta pruning)  →  move selection
```

The feature weights *are* the knowledge. Their trajectory over training is the primary data.

### Learning Algorithm

**Temporal Difference (TD) learning** — after each game, weights are adjusted based on
prediction error: did the evaluator correctly anticipate who would win?

This is the same learning principle used in Tesauro's TD-Gammon (1992), applied to chess.

### Self-Play Loop

The engine plays both sides. After each game, weights are updated and checkpointed
to the database. The cycle repeats continuously for the duration of a training run.

### Experimental Design

- Training run duration: 10 hours
- Number of independent runs: 30
- Each run starts from random or near-zero weights
- Concept "emergence" is defined as: weight exceeds a meaningful threshold
  (to be calibrated during early runs)

---

## Chess Concept Features

| Feature Name | Description | Concept Tracked |
|---|---|---|
| `material_queen` | Count of queens on board | Piece value learning |
| `material_rook` | Count of rooks | Piece value learning |
| `material_bishop` | Count of bishops | Piece value learning |
| `material_knight` | Count of knights | Piece value learning |
| `material_pawn` | Count of pawns | Piece value learning |
| `passed_pawn` | Pawns with no opposing pawns on same/adjacent files | Endgame pawn play |
| `doubled_pawn` | Pawns on same file (penalty) | Pawn structure weakness |
| `isolated_pawn` | Pawns with no friendly pawns on adjacent files (penalty) | Pawn structure weakness |
| `king_safety` | Count of attackers near king | King protection |
| `center_control` | Control of e4/d4/e5/d5 | Opening principles |
| `rook_open_file` | Rooks on files with no pawns | Rook activity |
| `connected_rooks` | Whether both rooks are on same rank/file | Piece coordination |

Features are computed from the perspective of the side to move (positive = good for side to move).

---

## System Architecture

### Components

| Component | Language | Description |
|---|---|---|
| Rules engine | Python (`python-chess`) | Legal move generation, game state |
| Feature extractor | Python | Board → named feature vector |
| Evaluator | Python | Feature vector → position score |
| Search | Python | Minimax with alpha-beta pruning |
| Self-play loop | Python | Plays games, logs outcomes |
| TD learner | Python | Updates weights after each game |
| Database logger | Python (`sqlite3`) | Writes games and weights to SQLite |
| Visualization | R (`tidyverse`, `DBI`, `RSQLite`) | Weight trajectories, concept emergence |

### Data Flow

```
Self-play game  →  game outcome + positions
                →  SQLite: games table
TD weight update  →  new weights
                →  SQLite: weights table (checkpointed every N games)
R reads SQLite  →  weight trajectory plots
                →  concept emergence charts
                →  cross-run comparison
```

### Database Schema

**`runs`** table:
| Column | Type | Description |
|---|---|---|
| run_id | INTEGER | Primary key |
| started_at | TEXT | Timestamp |
| hyperparameters | TEXT | JSON: learning rate, search depth, etc. |

**`games`** table:
| Column | Type | Description |
|---|---|---|
| game_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key |
| game_number | INTEGER | Sequential within run |
| outcome | TEXT | 'white', 'black', 'draw' |
| length | INTEGER | Number of moves |
| final_material | REAL | Material balance at game end |
| played_at | TEXT | Timestamp |

**`weights`** table:
| Column | Type | Description |
|---|---|---|
| weight_id | INTEGER | Primary key |
| run_id | INTEGER | Foreign key |
| checkpoint | INTEGER | Game number at checkpoint |
| feature_name | TEXT | Name of the feature |
| weight_value | REAL | Weight value at checkpoint |

---

## Pipeline Description

### Stage 01 — Engine (Python)
Build and test the core engine components:
- `01_engine.py`: feature extractor, evaluator, minimax search
- Unit tests to verify legal move generation and feature computation

### Stage 02 — Self-Play (Python)
- `02_selfplay.py`: self-play loop, TD learning, SQLite logging
- Configurable: run duration, checkpoint frequency, learning rate, search depth

### Stage 03 — Visualization (R)
- `03_visualize.R`: weight trajectory plots, concept emergence timing
- `03_compare_runs.R`: cross-run analysis, variance in emergence sequence

---

## Folder Structure

```
Chess Learning/                        ← Repo root
├── AGENTS.md
├── ActivitySetupGuide.md
├── README.md
├── ChessLearningDocumentation.md      ← This file
├── ChessLearningLog.md                ← Session log
├── ChessLearningConversations.md      ← Conversation transcript
├── scripts/
│   ├── python/
│   │   ├── engine.py                  ← Importable module (feature extractor, evaluator, search)
│   │   └── engine.ipynb               ← Stage 01: documented notebook with verification
│   │   └── selfplay.ipynb             ← Stage 02: self-play loop, TD learning, SQLite logging
│   └── r/
│       └── 03_visualize.qmd           ← Stage 03: Quarto visualization document
├── data/
│   └── chess_learning.db              ← SQLite database (run_id=1: 200-game verification)
├── viz/                               ← Output plots (populated by 03_visualize.qmd)
└── models/                            ← Saved weight checkpoints (future use)
```

## File Descriptions

| File | Description |
|---|---|
| `scripts/python/engine.py` | Importable Python module: 12-feature extractor, linear evaluator, negamax search with alpha-beta. Used by selfplay.ipynb. |
| `scripts/python/engine.ipynb` | Stage 01 notebook: same content as engine.py with markdown explanations and 6 verification checks. Works in Positron and Colab. |
| `scripts/python/selfplay.ipynb` | Stage 02 notebook: `play_game`, `td_update`, `train` functions; SQLite logging of runs/games/weights/feature_stats. |
| `scripts/r/03_visualize.qmd` | Stage 03 Quarto document: weight trajectories, normalised material weights, game stats, feature-outcome rolling correlation, concept emergence timeline. |
| `data/chess_learning.db` | SQLite database. Tables: `runs`, `games`, `weights`, `feature_stats`. |

---

## Key Decisions Made

| Decision | Rationale |
|---|---|
| Linear evaluation over neural network | Weights are directly interpretable; concept tracking is transparent |
| TD learning over supervised learning | No labeled data needed; learns purely from self-play outcomes |
| python-chess for rules | Mature, well-tested library; avoids reimplementing complex rules |
| SQLite for storage | Single-file, portable, readable by R via DBI |
| R for visualization | User preference; excellent for producing publication-quality charts |

---

## Open Questions

- What threshold defines concept "emergence"? (to be calibrated in early runs)
- What search depth is feasible within 10-hour run time?
- Should weights be initialized at zero or with small random values?
- How frequently to checkpoint weights? (every 10 games? 100 games?)
