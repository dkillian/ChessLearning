# Learning Study — Session Log

---

## Session 1 — September 29, 2026

### What We Worked On

- Defined the project concept: a chess engine that learns via self-play, with the goal
  of documenting and visualizing the progression of chess concept acquisition
- Discussed and selected system architecture
- Established the experimental design (30 independent 10-hour training runs)
- Set up the project folder structure and documentation files

### What Was Decided

- Use a **linear evaluation function** (not a neural network) for transparency —
  the weights themselves are the thing to measure
- Use **Temporal Difference (TD) learning** to update weights from self-play outcomes
- Use **python-chess** for the rules engine (avoids re-implementing complex rules)
- Use **SQLite** as the data interchange format between Python and R
- Use **R** for all visualization and cross-run analysis
- The primary research interest is the *sequence* of concept learning, not playing strength
- Hypothesis: game structure imposes a partial ordering on learning; early concepts
  should be consistent across runs, later concepts more variable

- Key chess concept features to track:
  - Material counts by piece type (piece value learning)
  - Passed pawn count
  - Doubled and isolated pawn penalties
  - King safety score
  - Center control
  - Rook on open file
  - Connected rooks

### What Was Created

- `Chess Learning/AGENTS.md` — project memory file
- `Chess Learning/ActivitySetupGuide.md` — adapted from workspace guide
- `Chess Learning/README.md` — public-facing GitHub readme
- `Chess Learning/Learning Study/` — activity folder with subdirectories
- `Chess Learning/Learning Study/Learning StudyDocumentation.md`
- `Chess Learning/Learning Study/Learning StudyLog.md` (this file)
- `Chess Learning/Learning Study/Learning StudyConversations.md`

### Risks & Uncertainties

- What threshold defines concept "emergence"? Will need calibration.
- Feasible search depth within 10-hour run time is unknown until tested.
- Whether weight trajectories will be smooth enough to yield clear emergence signals
  is uncertain — may need to smooth or aggregate.
- The 30-run experimental design is aspirational; compute time will determine feasibility.

### Steps for Next Session

1. Build Stage 01: the Python chess engine components
   - Feature extractor (board → named feature vector)
   - Linear evaluator (weighted dot product)
   - Minimax search with alpha-beta pruning
2. Write basic unit tests to verify legal move generation and feature computation
3. Do a short test run (e.g., 100 games) to confirm the pipeline runs end-to-end

---

## Session 2 — September 29, 2026

### What We Worked On

- Completed Stage 01: built `engine.ipynb` (primary working file) and `engine.py` (importable module)
  - Feature extractor, linear evaluator, negamax search with alpha-beta pruning
  - 12 chess concept features tracked
  - All 6 verification checks pass
- Converted engine to Jupyter notebook format for Colab/Positron compatibility
- Benchmarked engine speed at depth 2 and depth 3
- Completed Stage 02: built `selfplay.ipynb`
  - Self-play game loop with 50-move rule / draw adjudication
  - TD learning (gradient descent on squared prediction error)
  - SQLite logging: runs, games, weights tables
- Ran 200-game verification run (run_id = 1)

### What Was Decided

- Default depth: 2 (depth 3 is ~18x slower; depth 2 gives ~5,000–9,000 games per 10-hour run)
- Move limit: 200 half-moves per game with draw adjudication (50-move rule via `board.can_claim_draw()`)
- Weight initialization: small random values to break symmetry
- Learning rate: 0.001
- Weight checkpoints: every 50 games
- Weight clipping: [-50, +50] to prevent blow-up
- `engine.py` kept as importable module; `engine.ipynb` is the documented version
  Stage 02 notebook imports from `engine.py`

### What Was Created

- `scripts/python/engine.ipynb` — Stage 01 notebook with verification cells
- `scripts/python/engine.py` — importable module (used by selfplay.ipynb)
- `scripts/python/selfplay.ipynb` — Stage 02 notebook
- `data/chess_learning.db` — SQLite database (run_id=1, 200-game verification run)

### Risks & Uncertainties

- `material_queen` weight is growing clearly (+0.32 by game 200) — learning signal confirmed
- Other weights are noisy at 200 games — expected at this scale
- `material_pawn` weight went slightly negative by game 200 — concerning but plausible
  given the random play generating noisy TD targets; should resolve at scale
- Actual throughput: ~5,200 games/10hr (vs 9,500 benchmark) — the overhead of TD updates
  and SQLite writes adds ~35% overhead. Still ample for the research design.
- Speed increases over the run (306 → 522 games/hr) as the engine plays more purposefully
  and games end faster — good sign that learning is occurring

### Steps for Next Session

1. Build Stage 03: R visualization notebook ✓ (completed in this session)
2. Run a full 10-hour training run (run_id = 2) and visualize the result
3. Consider: is the learning rate appropriate? Review weight trajectories after a full run.

---

## Session 3 — September 29, 2026

### What We Worked On

- Updated selfplay.ipynb: weight checkpoints every 10 games (down from 50)
- Added feature_stats table: per-game mean feature values from white's perspective,
  logged every game to enable feature-outcome correlation analysis
- Built Stage 03: scripts/r/03_visualize.qmd — Quarto document rendering to HTML
- Installed RSQLite package
- Verified document renders cleanly against run 1 data

### What Was Decided

- Checkpoint every 10 games: enough granularity to see fast learners (queens) smoothly
  while slow learners (pawn structure) remain distinguishable from noise
- feature_stats logs per-game mean feature values converted to white's perspective
  (black positions negated) so feature-outcome correlation is consistently interpretable
- DB path in visualize.qmd auto-detects render vs interactive context
- Duplicate weight rows (from overlapping checkpoints) deduplicated in SQL query

### What Was Created

- `scripts/r/03_visualize.qmd` — visualization document with five sections:
  1. Weight trajectories (per feature group: material, pawn structure, positional)
  2. Material weights vs canonical values (normalised to pawn weight)
  3. Game statistics (outcome rates, game length, termination type)
  4. Feature-outcome correlation (rolling, 200-game window) — requires feature_stats data
  5. Concept emergence timeline (first checkpoint where weight > 0.1)
  6. Final weights summary table

### Risks & Uncertainties

- feature_stats is empty until next training run — correlation section renders gracefully
  with a "no data yet" message
- Emergence threshold of 0.1 is arbitrary — will need calibration after a full run
- Rolling window of 200 games for correlation may need adjustment depending on run length

### Steps for Next Session

1. Run a full training run (run_id = 2, ~9,000 games at depth 2) with updated selfplay.ipynb
2. Re-render 03_visualize.qmd to see full weight trajectories and feature-outcome correlations
3. Examine pawn weight trajectory specifically for U-curve pattern
4. Calibrate emergence threshold based on observed weight magnitudes
