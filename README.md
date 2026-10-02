# Chess Learning

A research project exploring how a chess engine learns through self-play — with a focus on
documenting and visualizing the **progression of chess concept acquisition** over training time.

---

## Goals

- Build a chess engine using a transparent, linear evaluation function with explicit feature weights
- Train the engine via self-play using Temporal Difference (TD) learning
- Track how specific chess concepts emerge and strengthen over the course of training
- Run multiple independent training runs to investigate whether concept acquisition
  follows a reproducible sequence

---

## Research Questions

1. Do chess concepts emerge in a predictable sequence during self-play training?
2. Which concepts are learned early (structurally necessary) vs. late (contextually acquired)?
3. Across 30 independent training runs, how consistent is the sequence of concept acquisition?
4. Is there variance in *when* a concept emerges, even if the *order* is stable?

---

## Architecture

```
python-chess (rules)
        ↓
Feature Extractor  →  Linear Evaluator (weighted sum)  →  Minimax Search (alpha-beta)
                                ↓
                        Self-Play Loop
                                ↓
                    TD Learning (weight update on outcome)
                                ↓
                        SQLite Database
                    (games, weights, metrics)
                                ↓
                    R Visualization & Analysis
```

The **feature weights are the knowledge**. By logging weight trajectories across training,
we can watch the engine "discover" chess concepts — and ask whether it always discovers
them in the same order.

---

## Chess Concepts Tracked

| Feature | Concept Being Tracked |
|---|---|
| Material count by piece type | Are relative piece values learned? (queen > rook > bishop/knight > pawn) |
| Passed pawn count | Is a pawn with no opposing pawns valued more highly? |
| Doubled pawn penalty | Does the engine learn to avoid pawn weaknesses? |
| Isolated pawn penalty | Does pawn structure matter? |
| King safety score | Is king protection learned? |
| Center control | Is control of the central squares valued? |
| Rook on open file | Is rook activity on open files learned? |
| Connected rooks | Is coordination between rooks valued? |

---

## Experimental Design

Training runs are 10 hours each. By running 30 independent runs from scratch,
we can measure:
- The mean training time at which each concept "emerges" (weight exceeds a threshold)
- The variance in emergence timing across runs
- Whether the rank order of concept emergence is consistent

Low variance → concept is structurally necessary (learned early by the logic of the game)
High variance → concept is contextually acquired (timing depends on random variation in games)

---

## Tech Stack

| Component | Tool |
|---|---|
| Chess rules | `python-chess` |
| Self-play & learning | Python (custom) |
| Data storage | SQLite |
| Visualization & analysis | R (`tidyverse`, `DBI`, `RSQLite`) |

---

## Project Structure

```
Chess Learning/
├── AGENTS.md
├── ActivitySetupGuide.md
├── README.md
└── Learning Study/
    ├── Learning StudyDocumentation.md
    ├── Learning StudyLog.md
    ├── Learning StudyConversations.md
    ├── scripts/
    │   ├── python/
    │   └── r/
    ├── data/
    ├── viz/
    └── models/
```
