# Chess Learning

A research project exploring how a chess engine learns through self-play — with a focus on
documenting and visualizing the **progression of chess concept acquisition** over training time.

---

## Goals

- Build different chess engines using different evaluation methods
  - Linear evaluation function with explicit feature weights
  - Neural net
  
- Train the engine using a few source methods
  - Self-play using Temporal Difference (TD) learning
  - Play against Stockfish at various ELO levels
  - Play against humans

- Track how specific chess concepts emerge and strengthen over the course of training

- Run multiple independent training runs to investigate whether concept acquisition
  follows a reproducible sequence

- Identify the minimum foundation needed to bootstrap learning
  - Are chess rules sufficient? 
  - Are certain board features more important than others for bootstrapping? 

---

## Research Questions

1. Do chess concepts emerge in a predictable sequence during self-play training?
2. Which concepts are learned early (structurally necessary) vs. late (contextually acquired)?
3. Is there variance in *when* a concept emerges, even if the *order* is stable?
4. How much does the engine need to be told? Does providing piece values or explicit features
   accelerate learning, or can the engine discover these things on its own?

---

## Scripts

[Temporal difference learning - Tutorial](https://dkillian.github.io/ChessLearning/scripts/r/td_tutorial.html)

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

The engine currently uses **14 learnable positional features**. Material evaluation
(queen=0.9, rook=0.5, bishop/knight=0.3, pawn=0.1) is hard-coded in the evaluator
rather than learned — but whether the engine can learn correct piece values autonomously,
and how that affects positional concept acquisition, is one of the experimental
parameters to vary in future runs.

| Feature | Concept Being Tracked |
|---|---|
| **Pawn structure** | |
| Passed pawn | Is a pawn with no opposing pawns on adjacent files valued more highly? |
| Doubled pawn penalty | Does the engine learn to avoid pawns stacked on the same file? |
| Isolated pawn penalty | Does the engine penalize pawns with no friendly neighbors? |
| Backward pawn penalty | Does the engine recognize pawns that can't be safely advanced? |
| Pawn advancement | Does the engine learn to push pawns collectively toward promotion? |
| **King safety** | |
| King safety | Is king protection (pawn shelter, attacker proximity) learned? |
| **Piece activity** | |
| Center control | Is control of the central squares (d4/d5/e4/e5) valued? |
| Rook on open file | Is rook activity on open files learned? |
| Connected rooks | Is coordination between rooks valued? |
| Mobility | Does the engine learn to maximize the number of squares it attacks? |
| Rook on seventh rank | Is rook infiltration to the opponent's back ranks recognized? |
| **Piece coordination** | |
| Knight centralization | Does the engine learn to place knights toward the center? |
| Bishop pair | Is the long-term advantage of having both bishops recognized? |
| Piece development | Does the engine learn to develop minor pieces off their starting squares? |

---

## Experimental Design

Training runs are 10 hours each. By running multiple independent runs per condition,
we can measure:
- The mean training time at which each concept "emerges" (weight exceeds a threshold)
- The variance in emergence timing across runs
- Whether the rank order of concept emergence is consistent

Low variance → concept is structurally necessary (learned early by the logic of the game)
High variance → concept is contextually acquired (timing depends on random variation in games)

### Experimental Conditions

Runs are organized around a central question: **how much does the engine need to be told?**
The conditions form a spectrum from maximally naive to maximally informed:

| Condition | What's given | What must be learned | Engine type |
|-----------|-------------|---------------------|-------------|
| Rules only | Legal move generation | Everything else | Neural net (future) |
| Learnable material, no features | Legal moves | Piece values | Linear |
| Learnable material + features | Legal moves + 14 features | Piece values + feature weights | Linear |
| Hard-coded material + features | Legal moves + piece values + 14 features | Feature weights only | Linear |

The current primary condition is **hard-coded material + features** — the most informed
starting point for the linear evaluator. Future runs will work back toward the naive end
of the spectrum. The "rules only" condition requires a neural net that can discover its
own features, rather than a linear evaluator that needs them pre-specified.

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
├── README.md
├── ChessLearningDocumentation.md
├── ChessLearningLog.md
├── scripts/
│   ├── python/       ← engine, self-play, TD learning
│   └── r/            ← visualization and analysis
├── data/             ← SQLite database
├── viz/              ← output plots
└── models/           ← saved weight checkpoints
```
