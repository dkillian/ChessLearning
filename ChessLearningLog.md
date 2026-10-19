# Learning Study — Session Log

---

## Run Registry

Each new run is started when a substantive change is made to the training setup.
Minor changes within a run (e.g., tranche size, LR adjustment) continue the same run_id via `RESUME = True`.

| run_id | Date | Games | depth | lr | move_limit | weight_init | epsilon | Key Change |
|--------|------|-------|-------|----|------------|-------------|---------|------------|
| 22 | 2026-10-12 | 3 | 3 | 0.001 | — | seeded run_id=21 game_200 | N/A (human) | Human play; seeded from run_id=21 game 200; learnable material |
| 23 | 2026-10-12 | ? | 3 | 0.001 | — | seeded run_id=21 game_240 | N/A (human) | Human play; seeded from run_id=21 game 240; learnable material |
| 24 | 2026-10-13 | 1305 | 3 | 0.001 | none | zero | 0.1 | **Neural net (MLP 19→32→1); TD(0)+Adam; peaked 18/19 signs at game 700; ELO peaked ~1079, declined to ~982 at game 1305; self-play collapse confirmed; run closed** |
| 25 | 2026-10-13 | 1 | 3 | 0.001 | — | seeded run_id=24 game_0 (attribution weights) | N/A (human) | Accidental: old human play app seeded from NN attribution weights before NN support added; effectively zero-initialized linear run |
| 26 | 2026-10-13 | 2+ | 3 | 0.001 | — | seeded run_id=24 game_10 | N/A (human) | **NN human play** (new NN-aware code); genuine PyTorch TD updates; ~3× larger parameter updates per halfmove than self-play |
| 27 | 2026-10-17 | 640 | 3 | 0.001 | none | zero | N/A (SF moves) | **SF teacher NN (`selfplay_sf_nn.ipynb`); plain Adam; ELO ~1105 at games 110 & 390 (above run 24 peak); slipped to ~1079 at game 440; 15/19 correct signs at game 640** |
| 28 | 2026-10-17 | 10 | 3 | 0.001 | none | zero | 0.1 | **Self-play v2 (`selfplay_nn_v2.ipynb`); AdamW + LR schedule + resignation; 11/19 signs; 50% decisive** |
| 29 | 2026-10-17 | 10 | 3 | 0.001 | none | zero | temp (first 15 moves) | **SF teacher v2 (`selfplay_sf_nn_v2.ipynb`); AdamW + LR schedule + temperature + resignation; 12/19 signs at game 10** |
| 1 | 2026-09-29 | 200 | 2 | 0.001 | 200 | random | — | Verification run; baseline pipeline |
| 2 | 2026-10-03 | 900 | 2 | 0.001→0.01 | 200 | random | — | Full run; diagnosed draw equilibrium; tested higher LR |
| 3 | 2026-10-04 | 200 | 2 | 0.001 | 200 | canonical | — | Warm-start + removed `board.can_claim_draw()`; draws persisted via rook-shuffle repetition |
| 4 | 2026-10-04 | 100 | 2 | 0.001 | 200 | canonical (×1) | 0.1 | Epsilon-greedy; 23% decisive rate; material weights eroded (scale mismatch) |
| 5 | 2026-10-04 | 50 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | Feature normalization fix; 18% decisive rate; no weight explosion |
| 6 | 2026-10-05 | 250 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | MC + reduced draw LR (draw_lr_scale=0.05); erosion halved but continued |
| 7 | 2026-10-05 | 600 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | TD(0) bootstrapping; slow linear erosion (~0.16/300 games for queen) |
| 8 | 2026-10-05 | 50 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | Frozen material weights attempt (kernel stale — frozen_features did not take effect) |
| 9 | 2026-10-05 | 50 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | Frozen material weights attempt 2 (same issue — Positron overwrote file edits) |
| 10 | 2026-10-05 | 50 | 2 | 0.001 | 200 | canonical (÷10) | 0.1 | Frozen material weights confirmed working; zero erosion on all material features |
| 11 | 2026-10-05 | 850 | 2 | 0.001 | 200 | zero (positional) | 0.1 | **Material hard-coded in evaluator; material removed from weight vector; 78% decisive rate; all positional signs correct** |
| 12 | 2026-10-05 | 1200 | 3 | 0.001 | 200 | zero (positional) | 0.1 | Depth=3; per-feature draw LR (king_safety=0.05); 84.2% decisive at game 600, 80.4% at game 1200 |
| 13 | 2026-10-07 | 212 | 3 | 0.001 | 200 | zero (positional) | softmax τ=0.05 | Softmax selection; 69.4% decisive (regression); 2:1 black-wins asymmetry; pawn_advancement correct sign for first time; center_control took over as top draw-gradient feature |
| 14 | 2026-10-07 | 400 | 3 | 0.001 | 200 | zero | 0.1 | **14-feature engine**; epsilon-greedy; draw_lr king_safety=0.05 center_control=0.2; 81.2% decisive; 11/14 correct signs; benchmark 3W/1L/6D vs SF1320 (ELO ~1428) |
| 15 | 2026-10-08 | 60+ | 3 | 0.001 | 200 | zero | softmax τ=0.05 | Softmax + 14-feature; 70% decisive; 11/14 correct signs at 60 games; 2.8:1 black asymmetry persists |
| 16 | 2026-10-08 | 560 | 3 | 0.001 | 80 | zero | N/A (SF moves) | **Stockfish teacher** (SF ELO=1320); `selfplay_sf.ipynb`; ELO 1212→1267→1212 (peaked game 60, regressed to baseline by game 560); 7/14 correct signs at 560 games; experiment abandoned |
| 17 | 2026-10-09 | 5 | 3 | 0.001 | — | seeded run_id=14 game_400 | N/A (human) | **Human play v1** via `07_human_play.R`; 5 games played (all 1-0); 11/14 correct signs maintained |
| 18 | 2026-10-10 | 450 | 3 | 0.001 | 200 | zero | 0.1 | **v2 engine from zero**; epsilon-greedy; 84.2% decisive; 11/14 correct signs (v2 convention); benchmark 2W/1L/7D vs SF1320 (~1355 ELO) |
| 19 | 2026-10-10 | 1 | 3 | 0.001 | 200 | seeded run_id=18 game_20 | N/A (human) | **v2 human play**; engine won (0-1, 102 half-moves); 11/14 correct signs (v2 convention) |
| 20 | 2026-10-11 | 510 | 3 | 0.001 | 200 | zero | 0.1 | **Learnable material** (`engine_learnable_material.py`); 19 features; 76.1% decisive; 16/19 correct signs; rook > queen plateau; benchmark 2W/4L/14D (~1285 ELO) |
| 21 | 2026-10-12 | 1050+ | 3 | 0.001 | 200 | zero | 0.1 | **`knight_outpost`** replaces `knight_pst`; learnable material; from zero; 17/19 correct signs at game 1050; ELO ~1285; best black asymmetry (1.10:1) in learnable-material runs |

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

---

## Session 4 — October 3, 2026

### What We Worked On

- Reviewed activity documentation and confirmed tranche-based training approach
- Added 5 new features to `engine.py`: `bishop_pair`, `rook_seventh`, `piece_development`,
  `mobility`, `pawn_advancement` (17 features total, 4 groups)
- Rewrote `03_visualize.qmd`: 4 feature groups, flextable tables, multi-feature delta plot
- Created `04_game_viewer.R`: Shiny app for stepping through impactful games move by move
  (uses reticulate + python-chess SVG rendering, prev/next navigation, move list highlighting)
- Started run_id = 2 with three tranches: 50 + 200 + 500 = 750 games total
- Fixed two bugs: `log_run` schema mismatch (runs table had 7 cols, INSERT used 6);
  `huxtable::set_caption` masking `flextable::set_caption` — fixed with explicit namespace

### What Was Decided

- Tranche-based training (not full 10-hour runs): run → inspect → continue
- New run (run_id = 2) rather than continuing run_id = 1, for consistent checkpoint
  granularity and complete `feature_stats` coverage from game 1
- `selfplay.ipynb` config: `N_GAMES = 500`, `RESUME = True` (left after last tranche)

### Key Finding: Draw Equilibrium

- 750 games: 747 draws, 3 black wins (all in games 1–10), 0 white wins
- All learning came from the 3 early decisive games; 740+ straight draws produced
  near-zero weight updates
- **Material weights frozen from game 10**: in draw positions, material is balanced
  (us − them ≈ 0), so the TD gradient for material features is ~0 regardless of error
- **`mobility` eroded to ~0**: has non-zero values in balanced draw positions, so
  draw target of 0 slowly pulls it toward zero
- The engine is stuck in a draw equilibrium: near-zero weights → random play →
  mostly draws → near-zero gradient → weights don't develop

### Risks & Uncertainties

- Root cause of draw equilibrium: the 50-move rule + 3-fold repetition claim
  (`board.can_claim_draw()`) causes random play to draw almost universally
- Decisive game frequency has not increased with more training (still ~0.4%)
- Whether this equilibrium can be broken without changing architecture is unknown

### Steps for Next Session

Choose one (or more) of these interventions to break the draw equilibrium:

1. **Higher learning rate** (0.01 vs 0.001) — amplify signal from decisive games;
   risk: draw erosion also amplifies
2. **Disable draw claims** — force games to play out to natural conclusion;
   risk: games get much longer, throughput drops
3. **Epsilon-greedy exploration** — random move with probability ε to break
   repetitive patterns and generate decisive outcomes
4. **Warm-start weights** — initialize material weights at canonical values
   (queen=9, rook=5, bishop=3, knight=3, pawn=1); engine plays purposefully
   from game 1, and we observe positional concept learning on top of correct material

---

## Session 5 — October 4, 2026

### What We Worked On

- Reviewed documentation and confirmed draw equilibrium diagnosis from Session 4
- Fixed `ChessLearningConversations.md`: sessions were out of chronological order;
  reordered to Session 1 → 2 & 3 → 4 → 5
- Ran tranche 4: 50 games at LEARNING_RATE=0.01 (10× increase), RESUME=True

### Key Finding: Higher LR Made Erosion Worse

- All 50 games were draws (1/2–1/2) — no decisive games generated
- Median max |delta| only slightly higher than tranche 3 (0.00047 vs 0.00029)
- One spike: max |delta| = 0.044, nearly matching tranche 1 — LR is high enough
  to learn from decisive games, but none occurred
- Several features now have wrong signs after amplified draw erosion:
  `material_bishop` (–0.087), `material_rook` (–0.003), `rook_seventh` (–0.036),
  `piece_development` (–0.049), `bishop_pair` (–0.083)
- Conclusion: higher LR alone is not the right fix; it amplifies draw erosion
  without generating decisive games

### Steps for Next Session

Applied both interventions in this session — see continuation below.

---

## Session 5 (continued) — October 4, 2026

### What We Worked On

- Applied both draw-equilibrium interventions simultaneously:
  1. Removed `board.can_claim_draw()` from `play_game()` in `selfplay.ipynb` so games
     play to checkmate or the hard 200-halfmove cap (no more optional draw claims)
  2. Warm-started weights at canonical piece values: queen=9, rook=5, bishop=3, knight=3, pawn=1
- Started run_id = 3: depth=2, lr=0.001, move_limit=200, WEIGHT_INIT='canonical'
- Ran two tranches of 100 games each (200 games total); RESUME=True for second tranche

### Key Finding: Draws Persist via Natural Termination

- 200 games: 199 draws, 1 white win (1-0) — 99.5% draw rate
- All games ended via `terminated_by = 'natural'` (not move cap)
- Average game length: 28.3 half-moves — very short
- Material weights held near canonical values (queen≈9.0, rook≈4.9, bishop≈3.0, knight≈3.0, pawn≈0.93)
- Positional weights noisy/wrong-signed: `king_safety` (–0.156), `passed_pawn` (–0.080),
  `mobility` (–0.075), `rook_open_file` (–0.040), `bishop_pair` (–0.002)
- Conclusion: removing draw claims shifted termination from claimed draws to natural draws
  (stalemate, threefold repetition enforced by python-chess); root cause not yet eliminated

### Steps for Next Session

Investigate why draws persist and choose next intervention:

1. **Diagnose natural draw types** — inspect PGN game records to determine whether
   games are ending by stalemate, threefold repetition (automatic), or 75-move rule
2. **Add randomization / exploration** — epsilon-greedy random moves to break
   repetitive patterns and generate more decisive outcomes
3. **Adjust move limit** — reduce hard cap to force more decisive outcomes sooner

---

## Session 6 — October 4, 2026

### What We Worked On

- Ran run_id = 5 (50 games) and discovered contamination: a prior undocumented
  attempt at run_id = 5 had already written data to the database, causing duplicate
  game records (games 1–50 doubled) and inflated weights
- Diagnosed the root cause of prior weight explosion: `mobility` and `pawn_advancement`
  features had raw values up to ±30 and ±40, respectively — far outside the [−1, +1]
  TD target range — creating prediction errors of 5–8× and driving weights to the ±50
  clip ceiling within 10 games
- Applied feature normalization fix: `mobility` ÷ 30, `pawn_advancement` ÷ 10 in
  `extract_features()`, bringing both into the ±1 range consistent with material features
- Updated `engine.py` and `engine.ipynb` with the normalization (also brought
  `engine.ipynb` up to date with the 5 features added in Session 4 that it was missing)
- Deleted all contaminated run_id = 5 data (7 tables, ~309k rows) and re-ran clean
  50-game tranche

### Key Findings

- **No weight explosion**: max weight magnitude is ±0.41 — normalization fix confirmed working
- **18% decisive rate**: 9 decisive out of 50 games — real improvement over earlier runs
- **Correct signs on core material features**: queen (+0.23), rook (+0.10), bishop (+0.07),
  pawn (+0.01) — all positive as expected
- **Wrong signs on several features**: `material_knight` (−0.12), `mobility` (−0.41),
  `passed_pawn` (−0.07) — likely noise at 50 games
- **Draw erosion persists**: queen started at 0.90 (canonical), eroded to 0.23 after
  50 games — 41 draws are pulling all weights toward zero faster than 9 decisive
  games can reinforce correct values

### What Was Changed

- `scripts/python/engine.py`: `mobility` ÷ 30.0, `pawn_advancement` ÷ 10.0 in
  `extract_features()`; normalization comment added
- `scripts/python/engine.ipynb`: same normalization applied; also added the 5 features
  missing since Session 4 (`bishop_pair`, `rook_seventh`, `piece_development`,
  `mobility`, `pawn_advancement`)

### Risks & Uncertainties

- Draw erosion remains the dominant force at 50 games; unclear whether the decisive-game
  signal will accumulate fast enough to stabilize correct weight signs at larger scale
- `mobility` weight (−0.41) being large and wrong-signed is concerning — may reflect
  genuine noise in 9 decisive games, or a structural artifact of the feature
- Whether 18% decisive rate is sufficient for meaningful learning over a full run
  remains to be seen

### Steps for Next Session

→ *Completed in Session 7. See below.*

---

## Session 7 — October 5, 2026

### What We Worked On

- Reviewed activity documentation files at session start
- Implemented **reduced draw learning rate** in `selfplay_mc.ipynb` (`draw_lr_scale=0.05`):
  draws use 5% of the normal learning rate to reduce erosion while preserving calibration
- Started run_id=6: 50-game tranche, then 200-game continuation (250 games total)
- Assessed run_id=6 results and diagnosed continued erosion despite the fix
- Decided to switch to **TD(0) bootstrapping** as the architectural solution
- Archived Monte Carlo notebook as `selfplay_mc.ipynb`; built `selfplay_td0.ipynb`
- Started run_id=7: 50-game initial tranche with TD(0)
- Rewrote `query self play notebook.R` as a clean, parameterized assessment tool
- Fixed two bugs in the assessment script: `names()` being pushed to SQLite; stray pipe line break

### Key Finding: TD(0) Eliminates Draw Erosion

Run_id=7 (TD(0), 50 games) vs prior runs at the same scale:

| Feature | Canonical | Run 5 (MC) | Run 6 (MC + draw_lr) | Run 7 (TD(0)) |
|---|---|---|---|---|
| `material_queen` | 0.90 | 0.23 | 0.57 | **0.876** |
| `material_rook` | 0.50 | 0.10 | 0.29 | **0.475** |
| `material_bishop` | 0.30 | 0.07 | 0.18 | **0.284** |
| `material_knight` | 0.30 | −0.12 | 0.12 | **0.267** |
| `material_pawn` | 0.10 | 0.01 | 0.01 | **0.055** |

TD(0) structurally suppresses draw erosion because the draw signal only propagates
one step at a time: the terminal draw target of 0 affects only the last position
directly. Earlier positions are targeted by `-V(s_{t+1})`, which remains non-zero
as long as the evaluator has non-zero weights. No `draw_lr_scale` parameter needed.

Weight ordering after 50 games: queen > rook > bishop ≈ knight > pawn — correct
for the first time across any run. Non-material weights are small and noisy at 50
games, which is expected.

### Run_id=6 Summary (MC + draw_lr_scale=0.05)

- 250 games total (50 + 200 tranche continuation)
- Decisive rate: 26% (improved from 18% in run 5)
- Erosion roughly halved vs run 5, but continued accumulating over 250 games
- `pawn_advancement` grew to 0.367 — equal to `material_queen` — structurally suspicious
- `material_pawn` drifted to −0.151; `material_bishop` to −0.015 (wrong signs)
- Conclusion: `draw_lr_scale` is a useful patch but not an architectural fix

### What Was Created / Changed

| File | Change |
|---|---|
| `scripts/python/selfplay_mc.ipynb` | New — archived Monte Carlo version of selfplay notebook |
| `scripts/python/selfplay_td0.ipynb` | New — TD(0) implementation; `td_update` rewritten; no `draw_lr_scale` |
| `scripts/python/selfplay.ipynb` | Added `draw_lr_scale` parameter (run_id=6 work); still present as working MC copy |
| `scripts/r/query self play notebook.R` | Rewritten as clean parameterized assessment tool (RUN_ID, GAME_START, GAME_END) |

### Steps for Next Session

1. Run a 200-game continuation of run_id=7 (RESUME=True) to confirm material weights
   hold at scale and non-material features begin to emerge with correct signs
2. If stable, consider whether to begin the 30-run experimental design with TD(0)
3. Calibrate the concept emergence threshold based on run_id=7 weight trajectories

→ *Continued in same session — see 300-game update below.*

---

## Session 7 (continued) — 300-game update

### What We Did

- Ran 250-game continuation of run_id=7 (RESUME=True), bringing total to 300 games
- Re-ran `query self play notebook.R` to assess full run

### Key Findings at 300 Games

**Material weights — slow but manageable erosion:**

| Feature | Initial | Game 50 | Game 300 | Total erosion |
|---|---|---|---|---|
| `material_queen` | 0.90 | 0.876 | 0.740 | 0.160 |
| `material_rook` | 0.50 | 0.475 | 0.387 | 0.113 |
| `material_bishop` | 0.30 | 0.284 | 0.203 | 0.097 |
| `material_knight` | 0.30 | 0.267 | 0.160 | 0.140 |
| `material_pawn` | 0.10 | 0.055 | 0.055 | 0.045 |

The queen lost 0.024 in the first 50 games, then 0.136 over the next 250. For
comparison, Monte Carlo run_id=6 lost 0.534 in 250 games. Weight ordering remains
correct throughout (queen > rook > bishop > knight > pawn).

**`mobility` drifting negative** — at −0.125, it is the largest non-material weight
by magnitude and has the wrong sign (more mobility = better, so weight should be
positive). It was −0.029 at game 50 and has drifted steadily more negative. Could
be noise from a small number of decisive games, or a genuine artifact of the feature.
Requires investigation.

**Other non-material weights** are small (all under 0.05 magnitude) and mixed-signed.
Too noisy to interpret at 300 games; expected at this scale.

### Steps for Next Session

1. Run a longer tranche (500+ games, RESUME=True on run_id=7) to determine whether
   material weights stabilize or continue slow drift
2. Investigate the `mobility` sign issue — check whether the feature computes
   correctly for both sides in decisive games
3. If weights appear to stabilize, begin planning the 30-run experimental design

→ *Continued in Session 8. See below.*

---

## Session 8 — October 5, 2026

### What We Worked On

- Continued run_id=7 to 600 games; assessed slow linear erosion (~0.16 queen erosion per 300 games)
- Implemented `frozen_features` mechanism in `selfplay_td0.ipynb` to prevent material weight erosion
- Runs 8 and 9 failed (frozen features did not take effect) due to Positron overwriting file edits on notebook save
- Diagnosed that `.ipynb` files cannot be reliably edited externally while open in Positron
- Fixed by closing the notebook and editing the JSON directly with Python; run_id=10 confirmed frozen features working (zero erosion on all material weights)
- Decided to go further: remove material features from the learnable weight vector entirely and hard-code them in `evaluate()`
- Refactored `engine.py`: removed 5 material features from `FEATURE_NAMES`, added `PIECE_VALUES` constant, updated `evaluate()` to compute material score directly
- `selfplay_td0.ipynb` updated: `FROZEN_FEATURES` removed, `RUN_ID=11`
- `query self play notebook.R` rewritten: removed canonical vector and erosion section (no longer applicable), restored `# Section ----` heading format for outline compatibility
- Ran first 50-game tranche of run_id=11

### Key Finding: Architectural Breakthrough at run_id=11

With material hard-coded in the evaluator and TD(0) learning only over the 12 positional features:

**Decisive rate: 78%** (up from 26% in the best prior run). Draw rate dropped to 22%.
The engine now plays purposefully — correct material evaluation at every position means games resolve rather than cycling into draws.

**All 8 significant positional weights emerged with correct signs after 50 games:**

| Feature | Weight | Note |
|---|---|---|
| `passed_pawn` | +0.050 | ✓ |
| `king_safety` | +0.041 | ✓ |
| `center_control` | +0.031 | ✓ |
| `mobility` | +0.027 | ✓ (was −0.036 in run 10; persistently wrong in all prior runs) |
| `rook_open_file` | +0.019 | ✓ |
| `piece_development` | +0.015 | ✓ |
| `rook_seventh` | +0.010 | ✓ |
| `bishop_pair` | +0.009 | ✓ |

The remaining 4 features (`pawn_advancement`, `doubled_pawn`, `connected_rooks`, `isolated_pawn`) are near-zero — insufficient signal at 50 games, not wrong-signed.

### Architecture as of run_id=11

| Component | Approach |
|---|---|
| Material evaluation | Hard-coded via `PIECE_VALUES` in `evaluate()` (queen=0.9, rook=0.5, bishop/knight=0.3, pawn=0.1) |
| Positional learning | TD(0) over 12 features; weights initialized at zero |
| Search | Negamax depth 2, alpha-beta |
| Exploration | Epsilon-greedy (ε=0.1) |
| Script | `selfplay_td0.ipynb` |

### What Was Created / Changed

| File | Change |
|---|---|
| `scripts/python/engine.py` | Removed 5 material features from `FEATURE_NAMES`; added `PIECE_VALUES`; `evaluate()` now hard-codes material score |
| `scripts/python/selfplay_td0.ipynb` | Removed `FROZEN_FEATURES`; `frozen_features` parameter removed from `td_update` and `train`; `RUN_ID=11` |
| `scripts/r/query self play notebook.R` | Rewritten: removed canonical/erosion sections; updated for 12 positional features; outline headings restored |

### Steps for Next Session

1. Run a 500-game continuation of run_id=11 (RESUME=True) to see positional weights
   strengthen and weaker features emerge
2. Assess whether `pawn_advancement`, `doubled_pawn`, `connected_rooks`, `isolated_pawn`
   develop correct signs with more decisive game signal
3. Consider planning the 30-run experimental design once positional learning is stable

→ *Continued in same session — see update below.*

---

## Session 8 (continued) — King_safety fix and depth=3

### King_safety Draw LR Fix — Confirmed Working

Applied `draw_lr_scales = {'king_safety': 0.05}` as a per-feature draw LR scale
in `td_update`. After 50 additional games resuming run_id=11, king_safety recovered
from +0.005 → **+0.038** — back to its pre-erosion level. All other features
unaffected. The diagnostic confirmed king_safety's draw delta (0.00513) remains
~3× larger than the next feature, but the scale reduction is now containing it.

Final state of run_id=11 at ~850 games:

| Feature | Weight | Trend |
|---|---|---|
| `mobility` | +0.14 (approx) | ↑ |
| `passed_pawn` | +0.116 | ↑ |
| `rook_open_file` | +0.066 | ↑ |
| `isolated_pawn` | +0.057 | ↑ |
| `bishop_pair` | +0.047 | → |
| `rook_seventh` | +0.046 | ↑ |
| `king_safety` | +0.038 | ↑ (recovered) |
| `piece_development` | +0.021 | → |
| `center_control` | +0.003 | → |
| `connected_rooks` | −0.007 | wrong sign (depth-2 artifact) |
| `doubled_pawn` | −0.026 | wrong sign (depth-2 artifact) |
| `pawn_advancement` | −0.038 | wrong sign (depth-2 artifact) |

### Depth=3 Run Started (run_id=12)

- `selfplay_td0.ipynb` configured: `RUN_ID=12`, `DEPTH=3`, `RESUME=False`
- All fixes carried forward: TD(0), hard-coded material, per-feature draw LR
- Run started at session close; game rate ~60s/game (move_limit=200)
- First checkpoint not yet reached at 10 minutes when session ended

### Steps for Next Session

1. **Evaluate run_id=12** (depth=3) — assess decisive rate, positional weight signs,
   and whether `doubled_pawn`, `pawn_advancement`, `connected_rooks` correct themselves
2. **Generate negamax tutorial** — Quarto document explaining negamax with alpha-beta
   pruning; use depth=3 for the illustrative game tree (recommended for pedagogy)
3. If depth=3 looks stable, assess whether the architecture is ready for the
   30-run experimental design

---

## Session 9 — October 6, 2026

### What We Worked On

- Reviewed activity documentation
- Assessed run_id=12 (depth=3) at ~552 games using `query self play notebook.R`

### Key Finding: Depth=3 Confirms "Depth-2 Artifact" Hypothesis

**Decisive rate: 84.2%** (0-1: 225, 1-0: 240, draws: 87) — up from 78% in run_id=11 at depth=2.
White wins slightly exceed black wins (43.5% vs 40.8%), consistent with first-mover advantage at greater depth.

**`connected_rooks` corrected to positive sign (+0.012)** at depth=3, after being persistently
negative (−0.007) throughout all depth=2 runs. This confirms the Session 8 hypothesis that
`connected_rooks`, `doubled_pawn`, and `pawn_advancement` were depth-2 artifacts.

**Positional weights at max checkpoint (~500 games):**

| Feature | run_id=12 (~500 ckpt) | run_id=11 (~850 games) | Sign |
|---|---|---|---|
| `mobility` | +0.151 | +0.14 | ✓ |
| `passed_pawn` | +0.109 | +0.116 | ✓ |
| `rook_open_file` | +0.046 | +0.066 | ✓ |
| `rook_seventh` | +0.040 | +0.046 | ✓ |
| `king_safety` | +0.017 | +0.038 | ✓ |
| `piece_development` | +0.017 | +0.021 | ✓ |
| `connected_rooks` | **+0.012** | −0.007 | ✓ **(corrected at depth=3)** |
| `center_control` | +0.009 | +0.003 | ✓ |
| `bishop_pair` | +0.006 | +0.047 | ✓ |
| `isolated_pawn` | +0.004 | +0.057 | ✓ |
| `doubled_pawn` | −0.015 | −0.026 | ✗ |
| `pawn_advancement` | −0.037 | −0.038 | ✗ |

`isolated_pawn` and `bishop_pair` are smaller than in run_id=11 at comparable scale —
run_id=12 is still at ~500 games, and run_id=11 showed these features strengthen
between games 50 and 850. Likely to develop further.

`doubled_pawn` and `pawn_advancement` remain wrong-signed. Unclear whether these
are also depth artifacts or genuinely slow learners; need more games at depth=3 to assess.

### Additional Work Completed This Session (Evening)

The session continued significantly after the initial log entry above. Key additions:

**Tooling ecosystem built:**
- `negamax_tutorial.qmd` — Quarto tutorial: game trees, minimax/negamax, alpha-beta pruning with Mermaid diagrams, worked example using live run_id=12 weights
- `05_move_scorer.R` — parameterized script: score all legal moves from any FEN at configurable depth/run
- `05_search_explorer.R` — Shiny app: interactive move scorer with board navigation and move ranking bar chart
- `06_search_mechanics.R` — step-by-step R walkthrough: board evaluation feature-by-feature (feature × weight contribution table), negamax depth-by-depth from depth=0 through depth=3
- `06_stockfish_arena.R` — Shiny app: Live Game (our engine vs Stockfish, real-time board animation) and ELO Benchmark (batch matches, win-rate chart, interpolated ELO estimate, trajectory across checkpoints)
- `07_human_play.R` — Shiny app: human vs engine with real TD(0) weight updates to a new run; live diagnostics (feature × weight table, evaluation trajectory chart) and post-game analysis (weight delta chart, top-3 learning moments, TD error vs self-play distribution)

**Stockfish 19 installed and integrated:**
- Installed via `winget install Stockfish.Stockfish`
- Confirmed working via python-chess; `SF_PATH` auto-detected from WinGet packages directory
- `stockfish_evals` DB table added (source = `'benchmark'` or `'live'`)

**Key analytical findings:**
- `center_control` feature only counts attacks, not pawn occupation — 1.e4 scores 1 (for attacking d5) rather than for occupying e4. Recorded in Feature Improvement Candidates in ChessLearningDocumentation.md.
- Depth-0 opening move scores: mobility dominates (~0.037 contribution), piece development second (~0.019); all other features near zero at move 1
- Random mover baseline: 20W/0L/0D at every checkpoint from game 100–600 — ceiling effect, baseline is useless. `BASELINE_EVERY` set to `None` going forward.

**Overnight jobs running at session close:**
- run_id=12 training: games 700 → 1200 (500-game tranche, `BASELINE_EVERY=None`)
- Stockfish ELO benchmark: first run at 3 ELO levels (1000, 1200, 1400), 10 games each, depth=3 — slow (~70 min) due to long games against Stockfish; patch queued

### Steps for Next Session

1. **Check benchmark results** — view win-rate chart in `06_stockfish_arena.R`; patch `move_limit` 200 → 80 and default `bm_n_games` 10 → 5 before next benchmark run
2. **Assess run_id=12 at game 1200** — check decisive rate, weight signs for `doubled_pawn` and `pawn_advancement`, compare weights to game 700
3. **Consider replacing `BASELINE_EVERY`** with Stockfish-at-low-ELO evaluation in `selfplay_td0.ipynb` for future runs (more discriminating than random mover)
4. **30-run experimental design** — assess readiness; weights appear stable at depth=3

---

## Session 10 — October 7, 2026

### run_id=12 Assessment at Game 1200

Decisive rate dropped from 84.2% (game 600) to **80.4%** (game 1200). Draw rate climbed to 19.6%.
Black now outperforms White (42.6% vs 37.8%), a reversal from game 600.

**Weight changes game 700 → 1200:**

| Feature | Game 700 | Game 1200 | Status |
|---|---|---|---|
| `mobility` | +0.151 | **+0.250** | ✓ but surging — dominating evaluation |
| `passed_pawn` | +0.109 | +0.083 | ✓ |
| `pawn_advancement` | −0.037 | **−0.103** | ✗ worsening fast |
| `doubled_pawn` | −0.015 | **−0.044** | ✗ worsening |
| `center_control` | +0.009 | **−0.001** | ✗ **crossed zero** — new wrong sign |
| `bishop_pair` | +0.006 | +0.024 | ✓ growing |
| `isolated_pawn` | +0.004 | +0.019 | ✓ growing |

Correct signs: 9/12 (was 10/12 at game 700 — `center_control` crossed zero).
`mobility` at 0.250 is ~3× the next feature, raising structural concerns.
These patterns suggest feature design issues may need to be addressed before a 30-run study.

### Stockfish Benchmark — Resolution

Three successive benchmark hangs in Shiny diagnosed and resolved:
- Root causes: synchronous observer blocking session, missing `ponder=FALSE`, `move_limit=200` too high
- **Fix:** moved benchmark to standalone script `05_stockfish_benchmark.R` designed as an RStudio background job
- Arena app (`06_stockfish_arena.R`) is now read-only (polls DB every 30s for new results)
- Key discovery: **Stockfish 19 minimum UCI_Elo = 1320** (not 1000 as assumed)
- First successful result: **0W/1L/0D vs Stockfish 1320** — engine ELO is below 1320

**PGN capture added:** each benchmark game is now stored in `stockfish_games` DB table with full PGN.
Games can be reviewed in `04_game_viewer.R` via the "Stockfish vs Engine" source toggle.

### Arena App Bug Fixes

`06_stockfish_arena.R` Live Game tab had persistent reactive context error:
- First fix: replaced `invalidateLater + isolate` with `reactiveTimer(600)`
- Final fix: replaced `current_board()` reactive expression calls inside `advance_one_move()` with `make_board_now()` — a plain function reading `move_history()` (a reactiveVal) directly, avoiding the reactive context requirement

### Game Viewer Enhancements (`04_game_viewer.R`)

- **Source toggle**: "Self-play" vs "Stockfish vs Engine" — each with appropriate selectors
- **"Why this move?" panel** (button-triggered):
  - Depth selector + Analyze button
  - Bar chart of all legal moves scored at the position (played move highlighted ★)
  - Feature breakdown table: Feature | Weight | Move1 (Value | Contrib) | Move2 | Move3
- **Search tree** (Graphviz DOT via `grViz()`):
  - Top-3 branches per level, 3 levels deep
  - Blue box = candidate move; orange diamond = opponent responses; green ellipse = our replies

### New: `selfplay_softmax.ipynb`

Built from `selfplay_td0.ipynb` by Python JSON editing. Key changes:
- **Softmax selection** replaces epsilon-greedy: scores top-N candidates, samples proportionally to $$e^{s_i/\tau}$$
- **Style modifiers**: 5 presets (neutral, attacking, cautious, reckless, positional) — each a delta vector added to weights during move selection only, leaving the TD learning signal unchanged
- Config: `STYLE='neutral'`, `TEMPERATURE=0.05`, `N_CANDIDATES=10`, `RUN_ID=13`

### Steps for Next Session

1. **Open `selfplay_softmax.ipynb`** — verify config (RUN_ID=13, STYLE='neutral'), run initial tranche (~500 games)
2. **Run Stockfish benchmark** via `05_stockfish_benchmark.R` as a background job after games complete
3. **Compare run_id=13 vs run_id=12**: decisive rate, weight trajectories, ELO — does softmax improve over epsilon-greedy?
4. **Investigate feature design issues**: why do `doubled_pawn` and `pawn_advancement` persistently show wrong signs? Consider looking at which game positions drive their updates.
5. **Style variant run**: once neutral baseline is established, try a style variant (e.g., `attacking`) and compare convergence

---

## Session 11 — October 7, 2026

### What We Worked On

- Reviewed and fully updated `ChessLearningDocumentation.md` (had not been updated since Session 1)
- Diagnosed and fixed four bugs in `selfplay_softmax.ipynb`
- Assessed run_id=13 (softmax, 212 games)
- Implemented major architectural improvements to `engine.py` (12 → 14 features)
- Configured both training scripts for the new architecture
- Launched run_id=14 (overnight, epsilon-greedy)

### Bugs Fixed in `selfplay_softmax.ipynb`

| Bug | Symptom | Fix |
|---|---|---|
| `negamax` not imported | `NameError` on first game | Added to `from engine import ...` |
| `train()` passing `epsilon=epsilon` to `play_game()` | `TypeError`; softmax params silently ignored | Replaced with `style=`, `temperature=`, `n_candidates=` |
| `numpy` not installed | `ModuleNotFoundError` | Added to pip install cell |
| Alpha-beta regression | 192s/game (3.2× slower than run_id=12) | Shared alpha window across root moves in `softmax_move_selection` |

### run_id=13 Assessment (212 games)

- **Decisive rate: 69.4%** — significant regression from run_id=12's 84%
- **Black wins 2:1 over white** (45.8% vs 23.6%) — origin unclear; could be noise or softmax bias
- **Weights (10/12 correct signs)**: `pawn_advancement` positive for first time across all runs
- **Draw-gradient analysis**: `king_safety` fix confirmed working (14× reduction); `center_control` took over as top draw-gradient feature (0.00131)
- **Speed after fix**: ~30s/game (faster than run_id=12's 60s; alpha-beta sharing effective)

### Engine Architecture Changes (`engine.py`)

Feature count: **12 → 14 learnable positional features**

| Change | Detail |
|---|---|
| `center_control` | Added pawn occupation of d4/d5/e4/e5; normalize by /4 |
| `pawn_advancement` | Redesigned as rank of most advanced pawn only; normalize by /5 |
| `backward_pawn` (new) | Pawn whose front square is attacked by enemy pawn + no adjacent support; negated penalty |
| `knight_pst` (new) | PST summed over all knights (rim=−0.50, central=+0.20); normalize by /2 |
| `_game_phase()` (new) | Midgame fraction [0,1] from weighted piece count / 24 |
| `king_safety` | Phase-scaled: × phase |
| `mobility` | Phase-scaled: × (0.5 + 0.5 × phase) |
| `pawn_advancement` | Endgame-scaled: × (0.5 + 0.5 × (1 − phase)) |

### Training Script Configuration

Both scripts updated for new 14-feature architecture:

| Setting | `selfplay_td0.ipynb` (run_id=14) | `selfplay_softmax.ipynb` (run_id=15) |
|---|---|---|
| WEIGHT_INIT | zero | zero |
| RESUME | False | False |
| N_GAMES | 400 | 400 |
| DRAW_LR_SCALES | king_safety: 0.05, center_control: 0.2 | same |

### Overnight Job

**run_id=14** — epsilon-greedy, 14-feature engine, 400 games — launched at ~11:15 PM.
Speed unknown with new features (estimate 40–70s/game); check first 10 games on wake.

### Steps for Next Session

1. **Check run_id=14 results** — confirm speed, decisive rate, weight signs for all 14 features
2. **Assess draw-gradient analysis** for run_id=14 — check whether new features need draw_lr_scales
3. **Investigate white/black asymmetry in run_id=13** — confirm whether it persists or was early noise
4. **Run Stockfish benchmark** on run_id=14 weights — does the new architecture improve ELO above 1320?
5. **Clarify "two apps" question** — four Shiny apps were open this session; user wanted to test two but we got sidetracked
6. **Consider run_id=15** (softmax, 14 features) once run_id=14 is assessed

---

## Session 12 — October 8, 2026

### What We Worked On

- Reviewed and updated `ChessLearningDocumentation.md` to reflect 14-feature engine architecture (feature table, weight state, open questions, file descriptions, normalization notes)
- Assessed run_id=14 (400 games), run_id=15 (60 games), and run_id=16 (60 training + 10/20 eval games)
- Ran Stockfish benchmark for run_id=14 at ELO 1320
- Built `selfplay_sf.ipynb` — new Stockfish-teacher training variant with built-in ELO estimation
- Fixed three Shiny app bugs and one database integrity issue
- Started overnight tranche for run_id=16 (N_GAMES=500, N_EVAL=30)

### Key Findings

**run_id=14 (epsilon-greedy, 14-feature, 400 games):**
- 81.2% decisive rate; 11/14 correct weight signs
- `mobility` dominance resolved by phase scaling (0.100 vs 0.250 in run_id=12)
- `passed_pawn` back on top (+0.122); `backward_pawn` learned correct sign immediately
- Black/white asymmetry persists: 46.5% black wins vs 34.8% white
- **Stockfish benchmark: 3W/1L/6D at ELO 1320 — first-ever engine wins against a rated opponent**; estimated ELO ~1428

**run_id=15 (softmax, 14-feature, 60 games):**
- 70% decisive rate (improving from 65% at 20 games)
- 11/14 correct signs at 60 games
- Black/white asymmetry worsening under softmax: 2.8:1 (51.7% vs 18.3%)
- `passed_pawn` leading (+0.045), `knight_pst` and `doubled_pawn` still wrong sign

**run_id=16 (Stockfish teacher, SF ELO=1320, 60 training games):**
- 0W/4L/6D in first 10 training games; all 6 draws hit 80-move limit (engine surviving, not collapsing)
- **No black/white asymmetry** — color is balanced by design (alternating by game_number parity); decisive games split evenly across colors
- ELO trajectory: 1212 (10 games) → 1267 (60 games), 95% CI 1088–1421 after 20 eval games
- At 10 games, positional weights are near-zero → ELO 1212 reflects the hard-coded material evaluation baseline
- Weight signs at 60 games: 10/14 correct (mostly correct; knight_pst, doubled_pawn, bishop_pair still noisy)

**Database fix:**
- Discovered 557 duplicate rows in `weights` table across all run_ids (some with different values, from accidental tranche restarts without RESUME=True)
- Retained MIN(weight_id) per (run_id, game_number, feature_name); deleted 557 duplicates

### New Scripts / Apps

| File | Description |
|---|---|
| `scripts/python/selfplay_sf.ipynb` | Stockfish-teacher training: plays against Stockfish at configurable ELO, TD(0) updates, built-in ELO estimation (separate eval batch, no weight updates). `RUN_ID=16`, `MOVE_LIMIT=80`, Windows asyncio fix included. |

### Bugs Fixed

| File | Bug | Fix |
|---|---|---|
| `scripts/r/07_human_play.R` | Column names `n_half_moves` and `elapsed_sec` didn't match actual DB schema | Changed to `n_halfmoves` and `duration_s` |
| `scripts/r/07_human_play.R` | `draw_lr_scales` missing `center_control = 0.2` in two locations | Added to function default and `end_game()` call site |
| `scripts/r/06_stockfish_arena.R` | Dead `play_one_game_vs_sf()` function left over from old architecture | Removed |
| `scripts/r/query self play notebook.R` | Outcomes section only showed raw game outcomes (1-0, 0-1, 1/2-1/2) | Added engine perspective columns (engine_color, engine_result) and summary table |
| `scripts/python/selfplay_sf.ipynb` | `chess.engine.SimpleEngine.popen_uci()` raised `NotImplementedError` on Windows | Added `asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())` to Stockfish setup cell |

### Overnight Job

**run_id=16** — Stockfish teacher, SF ELO=1320, depth=3, N_GAMES=500, N_EVAL=30 — started ~9:28 PM, estimated completion before 8 AM. RESUME=True from game 61. ELO estimate at game ~560 will be logged to `stockfish_evals`.

### Steps for Next Session

1. **Check run_id=16 overnight results** — ELO estimate at ~560 total games; assess weight signs and decisive rate; compare ELO trajectory to self-play (run_id=14 estimated ~1428)
2. **Assess run_id=15** — check if more tranches ran overnight; compare softmax vs epsilon-greedy weight convergence on 14-feature engine
3. **Compare SF teacher vs self-play** — plot ELO trajectory and weight sign acquisition rate side-by-side for run_id=14 vs run_id=16
4. **Consider next SF teacher experiment** — raise SF_ELO to 1400 once run_id=16 is competitive at 1320; or compare SF_ELO=1320 vs 1500 as separate runs
5. **Update `05_stockfish_benchmark.R`** — run fresh benchmark for run_id=15 at 60+ games

---

## Session 13 — October 9, 2026

### What We Worked On

- Assessed run_id=16 overnight results (560 total games)
- Decided to abandon the Stockfish teacher experiment; return to epsilon-greedy self-play
- Fixed crash bug in `07_human_play.R` and played one human game (run_id=17)

### run_id=16 Full Assessment (560 games, Stockfish teacher)

**Outcomes (engine perspective vs SF 1320):**
- Win: 40 (7.1%), Draw: 366 (65.4%), Loss: 154 (27.5%)
- 65% of draws terminated by move_limit (80 halfmoves) — engine surviving, not winning

**ELO trajectory:**

| Games | Eval games | W/L/D | Score | Est. ELO |
|-------|------------|-------|-------|----------|
| 10    | 10         | 0/3/7 | 0.350 | ~1212    |
| 60    | 20         | 2/5/13| 0.425 | ~1267    |
| 560   | 30         | 3/12/15| 0.350 | ~1212   |

ELO peaked at game 60 then returned to baseline — no net improvement from 500 more games of SF teaching.

**Weight signs at game 560: 7/14 correct** — major regression from 11/14 at game 60.

Notable wrong-sign weights: `passed_pawn` at −0.022 (largest by magnitude, wrong sign — was the best-learned feature in run_id=14); `doubled_pawn` at +0.016; `backward_pawn` at +0.003.

### Decision: Abandon SF Teacher

The Stockfish teacher at ELO 1320 (the minimum) sends confounding TD signals for pawn structure features. The 500-game overnight tranche produced no ELO gain and worsened sign correctness from 11/14 to 7/14. Key factors:

- SF plays tactically complex positions that a 14-feature linear evaluator cannot parse
- Move_limit draws mask losses, warping the draw-gradient signal for pawn features
- ELO regression and sign regression are consistent with each other

**Recommendation:** Use Stockfish only for benchmarking (as in run_id=14). Revisit SF teacher after future feature improvements bring the engine closer to SF's playing strength.

### Bug Fix: `07_human_play.R`

`create_new_run()` referenced `epsilon` and `weight_init` columns in the INSERT statement, but neither column exists in the `runs` table. Caused a crash when clicking "Start Session". Fixed by removing both columns from the INSERT/SELECT.

### Human Play — run_id=17, Game 1

- Seeded from run_id=14, game 400 (best checkpoint: 11/14 correct signs, ELO ~1428)
- Outcome: 1-0, 83 halfmoves, natural conclusion
- TD error: 0.0542 — 77th percentile of run_id=14 self-play distribution (avg 0.0442)
- Top weight movers: `isolated_pawn` (−0.00184), `backward_pawn` (−0.00151), `piece_development` (+0.00101) — all in correct direction
- Sign correctness after game 1: 11/14 maintained (no regression)
- Human games provide stronger learning signal than average self-play (higher TD error = engine predictions more wrong = more surprise)

### Steps for Next Session

1. **Continue playing human games** via `07_human_play.R` (run_id=17, seeded from run_id=14 game 400); track whether the 3 wrong-sign features (`backward_pawn`, `knight_pst`, `pawn_advancement`) correct themselves with human game signal
2. **Run epsilon-greedy self-play** (run_id=18, seeded from run_id=14 game 400) targeting 1000–1200 games — longer run than run_id=14 to see if remaining wrong-sign features resolve
3. **Use SF only for benchmarking** — once run_id=18 is established, benchmark against SF 1320 to compare ELO to run_id=14

---

## Session 14 — October 10, 2026

### What We Worked On

- Reconstructed aborted session work from git diff and DB — all code and data intact; only the conversation transcript was lost
- Implemented **v2 engine redesign** — three feature changes in `engine.py`:
  - `backward_pawn`: tightened definition to require an enemy *pawn* (sentry) controlling the stop square, not any attacking piece
  - `king_safety`: complete redesign — pawn shelter (friendly pawns 1–2 ranks ahead of king) minus attacker count in king zone; normalized by ÷ 3; replaces old "safe adjacent squares" heuristic
  - `pawn_advancement`: changed from rank of single most advanced pawn (÷ 5) to total advancement of all pawns from starting rank (÷ 20); captures collective pawn mass advancement
- Archived v1 training notebooks with version suffixes (`selfplay_td0_v1_runs1-14.ipynb`, `selfplay_softmax_v1_run15.ipynb`, `selfplay_sf_v1_run16.ipynb`)
- Built **`public_app/`** — crowdsourced human-vs-engine Shiny app targeting shinyapps.io deployment; uses Supabase PostgreSQL backend; all public games contribute TD(0) updates to a shared run; chessboard.js drag-and-drop board
- Updated `07_human_play.R`: added `engine_version` tracking, `SEED_RUN_ID = 18L`, helper functions for v2 run detection (`get_active_human_run`, `get_latest_game_number`, `parse_source_run_id`), `human_color` column logging
- Continued run_id=17 to 5 human games (all white wins); sign correctness maintained at 11/14
- Started **run_id=18** (v2 self-play, epsilon-greedy, depth=3, 20 games planned from zero)
- Session aborted during attempted human play against v2 engine; crash source unidentified (likely Python/reticulate-level issue during depth=3 search); no data loss
- Cleaned up stale run_id=19 (created with zero weights; deleted and will be recreated from checkpoint 10)

### run_id=17 at Game 5 (human play, v1 engine, seeded from run_id=14 game 400)

All five games: white wins (1-0). Sign correctness: 11/14 maintained — same 3 persistent wrong-sign features.

| Feature | Weight | Status |
|---|---|---|
| `passed_pawn` | +0.120 | ✓ |
| `mobility` | +0.105 | ✓ |
| `center_control` | +0.052 | ✓ |
| `rook_open_file` | +0.047 | ✓ |
| `rook_seventh` | +0.044 | ✓ |
| `king_safety` | +0.032 | ✓ |
| `piece_development` | +0.027 | ✓ |
| `backward_pawn` | +0.023 | ✗ |
| `bishop_pair` | +0.018 | ✓ |
| `connected_rooks` | +0.010 | ✓ |
| `isolated_pawn` | −0.001 | ✓ |
| `knight_pst` | −0.012 | ✗ |
| `doubled_pawn` | −0.019 | ✓ |
| `pawn_advancement` | −0.023 | ✗ |

### run_id=18 at Checkpoint 10 (v2 engine, 10 games from zero)

All weights are noise at this scale (max magnitude ~0.005). Too early to assess signs meaningfully. First real checkpoint for assessment will be at game 20.

### New Scripts / Files Created

| File | Description |
|---|---|
| `public_app/app.R` | Crowdsourced human-vs-engine Shiny app; Supabase PostgreSQL backend; chessboard.js drag-and-drop; shared TD(0) weight updates across all public players |
| `public_app/engine.py` | Copy of `engine.py` for deployment alongside `public_app/app.R` |
| `scripts/python/selfplay_td0_v1_runs1-14.ipynb` | Archived v1 TD(0) training notebook (runs 1–14) |
| `scripts/python/selfplay_softmax_v1_run15.ipynb` | Archived v1 softmax training notebook (run 15) |
| `scripts/python/selfplay_sf_v1_run16.ipynb` | Archived v1 Stockfish-teacher notebook (run 16) |

### Session 14 (continued) — run_id=18 assessment and first v2 human game

**run_id=18 completed (20 games).** Checkpoints at game_number=0, 10, 20.

**Draw gradient analysis (corrected):** All v2 draw gradients dramatically smaller than v1. `king_safety` dropped to dead last (#14, 0.0000007) — was #1 in v1 requiring draw_lr=0.05. `center_control` #12 (0.000003) — was #2 requiring draw_lr=0.2. `pawn_advancement` #9 (0.000005) — was high in v1 (causing erosion). The `draw_lr_scales` corrections may not be needed for v2, though retained for safety pending more data. Note: based on only 2 draw games — confirm at larger scale.

**run_id=19 created** — v2 human play run, seeded from run_id=18 game_number=20. First human game played: engine won as black (0-1), 102 half-moves. Gameplay noticeably stronger than v1 despite only 20 training games.

**Weight signs after game 1 (run_id=19):** **12/14 correct** — improvement over v1's persistent 11/14.
- `pawn_advancement` (+0.000357): ✓ **correct for first time in v2** (was wrong in all v1 runs)
- `knight_pst` (+0.000435): ✓ **correct for first time** (was persistently wrong in v1)
- `backward_pawn` (+0.005): ✗ still wrong
- `connected_rooks` (−0.001): ✗ flipped wrong (was correct in v1 at depth=3)

**Bug noted:** Duplicate weight rows in `weights` table for run_id=19, game_number=1 — 28 rows instead of 14. Values identical; weights correct but `log_human_game` appears to fire twice. Fix next session.

### Steps for Next Session

1. **Fix double-insert bug** in `07_human_play.R` — `log_human_game` fires twice at game end; investigate whether `end_game()` is being triggered from both the human-move and engine-move observers
2. **Continue human games** (run_id=19, v2 engine) — track whether `pawn_advancement` and `knight_pst` hold correct signs; watch `backward_pawn` and `connected_rooks`
3. **Update `engine.ipynb`** to v2 — still on v1 (has material features in weight vector; old king_safety, pawn_advancement, backward_pawn implementations)
4. **Consider longer v2 self-play run** — run_id=18 is only 20 games; 400+ games needed to meaningfully compare sign correctness and decisive rate to v1 run_id=14

---

## Session 15 — October 11, 2026

### Summary

Documentation and architecture session. Corrected a fundamental misunderstanding of
the v2 sign convention, assessed run_id=18 fully, ran a Stockfish benchmark, created
a strategic work plan, expanded the experimental design framework, and launched the
first learnable-material run (run_id=20).

### Key Findings

**v2 sign convention corrected:** All 14 positional features should have positive weights
in v2. Penalty features (`doubled_pawn`, `isolated_pawn`, `backward_pawn`) are negated
in `extract_features()` — stored as (opponent_count − my_count) — so a positive weight
is always correct. Previous assessments for runs 18 and 19 were re-scored under this
corrected convention. AGENTS.md was updated accordingly.

**run_id=18 full assessment (380 games):**
- Decisive rate: 84.2% — matches the best prior self-play result (run_id=12 at 84%)
- Correct signs: 11/14 (v2 convention)
- Wrong signs: `doubled_pawn` (−), `pawn_advancement` (−), `knight_pst` (−)
- Draw suppression confirmed: `king_safety` ~31× suppressed in draws; `center_control` ~4.5×
- `isolated_pawn` started with correct negative sign, crossed zero around game 150, drifted positive

**Stockfish benchmark (run_id=18, game 450):**
- Result: 2W/1L/7D vs SF 1320
- Score: 0.55, estimated ELO ~1355
- Comparison: run_id=14 (v1) scored 0.60 → ~1390 ELO
- Difference is within noise at 10 games per benchmark

**run_id=19 sign assessment corrected:**
- Under v2 convention: 11/14 correct (not 12/14 as logged last session)
- `backward_pawn` (+0.005) is now correctly classified as correct (was misclassified)
- `isolated_pawn` (−0.003) and `doubled_pawn` (−0.002) are wrong (were misclassified)

**Double-insert bug (07_human_play.R):** Confirmed fixed via if/else branching.
Deleted 14 duplicate weight rows from DB for run_id=19 game_number=1.

**Experimental design framework expanded:**
- Conditions framed as a spectrum from naive (no rules) to fully informed (hard-coded material + explicit features)
- Linear evaluator covers conditions 3–5; neural net needed for conditions 1–2
- Engine files to be maintained as first-class research artifacts, one per condition

**Engine versioning:**
- `engine_hardcoded_material.py` — clean archive of the current condition (runs 1–19)
- `engine_learnable_material.py` — new condition: 19 learnable features (5 material + 14 positional); no hard-coded piece values

**run_id=20 launched (learnable material condition):**
- 20-game check: 85% decisive, 16/19 correct signs, all 5 material weights positive
- Material ordering not yet correct (rook > pawn > bishop ≈ knight > queen; target: queen > rook > bishop ≈ knight > pawn)
- Overnight tranche running

### New Scripts / Files Created

| File | Description |
|---|---|
| `ChessLearningWorkPlan.md` | Multi-session strategic work plan |
| `scripts/python/engine_hardcoded_material.py` | Archive of current engine condition (runs 1–19): 14 positional features, hard-coded material |
| `scripts/python/engine_learnable_material.py` | New engine condition (run 20+): 19 features — 5 material + 14 positional; no hard-coded piece values |
| `scripts/python/selfplay_learnable_material.ipynb` | Training notebook for run_id=20; imports `engine_learnable_material` |

### Overnight Jobs

**run_id=20** (`selfplay_learnable_material.ipynb`) running overnight. Will resume with
`RESUME = True` for additional tranches as needed.

### Steps for Next Session

1. **Assess run_id=20 overnight results** — outcomes, weight ordering, sign correctness
2. **Check material weight ordering** — watch for queen to pull ahead of pawn/rook as more games accumulate
3. **Diagnose the 3 persistently wrong-sign features** — `doubled_pawn`, `pawn_advancement`, `knight_pst` (delta distribution, trajectory, feature-outcome correlation)
4. **If run_id=20 is viable** — add engine condition menu to `07_human_play.R` so the human player can choose which engine configuration to play against
5. **Update `engine.ipynb`** to v2 architecture

---

## Session 16 — October 12, 2026

### What We Worked On

- Assessed run_id=20 overnight results (510 games, learnable material)
- Fixed Stockfish benchmark script (`05_stockfish_benchmark.R`): wrong engine import bug; added `ENGINE_MODULE` config parameter
- Re-ran Stockfish benchmark for run_id=20 (20 games at ELO 1320)
- Performed wrong-sign feature diagnostics on run_id=20: trajectory plots, outcome-correlated feature distributions, weight delta distributions
- Replaced `knight_pst` with `knight_outpost` in `engine_learnable_material.py`
- Launched run_id=21 (learnable material, `knight_outpost`, from zero; first 50-game tranche running at session close)

### Key Findings

**run_id=20 at 510 games:**
- 76.1% decisive (0-1: 42.9%, 1-0: 33.1%, draws: 23.9%); 1.3:1 black asymmetry
- 16/19 correct signs (wrong: `doubled_pawn`, `pawn_advancement`, `knight_pst`)
- Material weights plateaued early — rook (0.186) still exceeds queen (0.116) at game 510, barely moved since game 20
- 14 draws at ELO 1320 is expected given unconverged material

**Stockfish benchmark (run_id=20, game 510, 20 games at ELO 1320):**
- 2W / 4L / 14D; score 0.45; estimated ELO ~1285
- 70% of games hit the 80 half-move cap
- ~70 ELO points below the hardcoded-material baseline (run_id=18, ~1355)
- Note: initial benchmark attempt was invalid — `05_stockfish_benchmark.R` imported `engine.py` (hardcoded material) instead of `engine_learnable_material.py`; material weights were being ignored entirely. Fixed by adding `ENGINE_MODULE` config parameter. 13 invalid benchmark rows deleted and re-run with correct engine.

**Consistent ELO comparison (all using same formula):**

| run_id | Games | W/L/D | Score | ELO est. |
|--------|-------|-------|-------|----------|
| 14 | 10 | 3/1/6 | 0.60 | ~1390 |
| 18 | 10 | 2/1/7 | 0.55 | ~1355 |
| 20 | 20 | 2/4/14 | 0.45 | ~1285 |

**Wrong-sign feature diagnostics (three distinct failure modes):**

| Feature | Failure mode | Detail |
|---------|-------------|--------|
| `doubled_pawn` | Magnitude asymmetry | 54% positive updates (right direction!) but fat negative tail overwhelms; near-random walk |
| `pawn_advancement` | Frequency bias (moderate) | 61% negative, equal magnitudes; credit assignment — advanced pawns lead to exchanges engine mis-evaluates |
| `knight_pst` | Frequency bias (severe) | 75.8% negative, equal magnitudes; PST rewards centralized knights that get captured at depth=3 |

**Feature redesign: `knight_pst` → `knight_outpost`:**
- New definition: knights in opponent's half not attacked by any opponent pawn (stable outpost knights)
- Implemented as `_safe_knight_outpost()` in `engine_learnable_material.py`
- `_KNIGHT_PST` table and `_knight_pst_score()` removed
- `engine.py` (hardcoded material baseline) left unchanged
- Fixed `AttributeError: module 'chess' has no attribute 'BB_RANK_0'` — python-chess uses 1-indexed rank constants (`BB_RANK_1` through `BB_RANK_8`)

### Scripts Changed

| File | Change |
|------|--------|
| `scripts/python/engine_learnable_material.py` | `knight_pst` → `knight_outpost`; `_safe_knight_outpost()` replaces `_knight_pst_score()` and `_KNIGHT_PST` |
| `scripts/r/05_stockfish_benchmark.R` | Added `ENGINE_MODULE` config parameter (default `"engine_learnable_material"`); replaced hardcoded `"engine"` import |

### Steps for Next Session

1. **Assess run_id=21 at 50 games** — check sign correctness, especially `knight_outpost`; compare decisive rate to run_id=20
2. **Launch overnight tranche** — run_id=21, 500+ games
3. **Architecture decision** — after ~500 games of run_id=21, decide: continue learnable material exploration or return to hardcoded material baseline for the 30-run experimental design
4. **Update `ChessLearningWorkPlan.md`** — plan is stale relative to current learnable material phase

---

## Session 17 — October 13, 2026

### What We Worked On

- Assessed run_id=21 overnight results (1033 games then 1050 after final tranche)
- Ran Stockfish benchmark for run_id=21 (20 games at ELO 1320)
- Removed 80 half-move limit from `05_stockfish_benchmark.R` — games now play to natural conclusion under chess rules only
- Discussed neural net engine implementation plan (Phase 1: MLP with same 19 features, PyTorch)
- Corrected AGENTS.md: Stockfish teacher approach not "abandoned" — just not continued since run_id=16
- Discussed cross-run co-learning and K+R vs K endgame failure (hardcoded heuristic recommended)
- Session-end git commit and push

### Key Findings

**run_id=21 at 1033 games (overnight assessment):**
- Material ordering dramatically improved: rook (0.213) > queen (0.184) > bishop (0.109) > knight (0.058) > pawn (0.043) — best in any learnable-material run
- `knight_outpost` (+0.024) still positive and growing — redesign confirmed working
- BUT: 15/19 correct signs — `bishop_pair` and `isolated_pawn` temporarily wrong; `rook_open_file` and `piece_development` fading; `pawn_advancement` at −0.031

**run_id=21 at 1050 games (final tranche):**
- 17/19 correct signs — `bishop_pair` and `isolated_pawn` self-corrected
- Wrong: `doubled_pawn` (−0.007, volatile random-walk), `pawn_advancement` (−0.031, structural)
- Black asymmetry 1.10:1 — best in any learnable-material run
- Decisive rate: 74.4%

**Stockfish benchmark (run_id=21, game 1040, 20 games at ELO 1320):**
- 3W / 5L / 12D; score 0.45; estimated ELO ~1285
- 12/20 games (60%) hit 80 half-move cap — benchmark still distorted by move limit
- Same ELO as run_id=20 (510 games) despite better weights and sign count
- Note: 80 half-move limit removed from benchmark script — next run will use chess rules only

**ELO comparison (consistent formula):**

| run_id | Games | W/L/D | Score | ELO est. |
|--------|-------|-------|-------|----------|
| 14 | 10 | 3/1/6 | 0.60 | ~1390 |
| 18 | 10 | 2/1/7 | 0.55 | ~1355 |
| 20 | 20 | 2/4/14 | 0.45 | ~1285 |
| 21 | 20 | 3/5/12 | 0.45 | ~1285 |

### Scripts Changed

| File | Change |
|------|--------|
| `scripts/r/05_stockfish_benchmark.R` | Removed `move_limit = 80L` — games now play to natural conclusion |
| `scripts/r/07_human_play.R` | Session 16 carry-over: engine/run/checkpoint selection menu; claim_draw fix; ENGINE_MODULE config |
| `scripts/python/selfplay_td0.ipynb` | Session 16 carry-over: claim_draw=True fix |
| `scripts/python/selfplay_learnable_material.ipynb` | Session 16 carry-over: claim_draw=True fix |
| `scripts/r/03_visualize.qmd` | Session 16 carry-over: SHOW_RUNS Quarto param; backward_pawn + knight_pst/outpost added |

### Steps for Next Session

1. **Implement neural net engine (Phase 1)** — MLP with same 19 features; PyTorch 2.8.0 confirmed available; Adam optimizer; save weights as `.pt` files to `models/`; drop-in replacement for `evaluate()` in negamax search; new training notebook `selfplay_nn.ipynb`
2. **Run first 50-game tranche on neural net** — verify training loop, check feature attributions
3. **Re-run Stockfish benchmark on run_id=21** — now without move limit for cleaner ELO estimate
4. **Continue run_id=21 overnight** — target 2000+ total games before architecture decision

---

## Session 18 — October 13, 2026

### Overview

Major session: implemented the neural net engine condition end-to-end, ran the first 100 NN self-play games, added NN support to the human play and benchmark scripts, rebuilt the public app for multi-condition play, and deployed to shinyapps.io. Research purpose formally reframed from a 30-run reproducibility study to a broader investigation of how engines learn chess across conditions and architectures.

### Key Findings

**Neural net condition (run_id=24):**
- Decisive rate: 44% (games 1–50) → **70%** (games 51–100) — fast improvement
- Game length: 110 → 69 halfmoves (engine playing sharper)
- **pawn_advancement positive at both game 50 (+0.101) and game 100 (+0.120)** — persistent correct sign; never achieved in any linear run at any game count
- **Material ordering corrected to queen > rook by game 100** — linear learnable material (run_id=21) had rook > queen at 1050 games and never fixed it
- Attribution weights at game 100: 13/19 correct signs
- Attribution stability: dV/dx_i values differ by <0.002 across opening/midgame/endgame reference positions — validates attribution proxy methodology at this stage

**Human vs NN (run_id=26):**
- Human games cause **~3× larger parameter updates per halfmove** than self-play at equivalent training stage (0.0042 vs 0.0013 per halfmove)
- Self-play learning decelerating fast: 0.141 → 0.047 Frobenius norm per game from games 10–50
- Interpretation: human play introduces novel positions with higher TD surprise; structurally reduces the exploration collapse that self-play is prone to

**Research reframe:** Project purpose is not a fixed 30-run reproducibility study but an open investigation of how chess engines learn — across different architectures (linear hardcoded material, linear learnable material, neural net), training regimes (self-play, human play, Stockfish teacher), and feature configurations. All conditions are first-class.

### Scripts Added / Changed

| File | Change |
|---|---|
| `scripts/python/engine_nn.py` | **New** — ChessNet MLP (19→32→1, tanh); `evaluate()`, `negamax()`, `get_best_move()`; `attribution_weights()` (dV/dx_i at starting position); `nn_td_update()` (R-friendly interface); `make_optimizer()`; `compute_baseline_norm()`; `model_to_b64()` / `model_from_b64()` for Supabase storage |
| `scripts/python/selfplay_nn.ipynb` | **New** — TD(0)+Adam self-play training for NN condition; same DB schema as linear conditions; attribution weights logged instead of raw params |
| `scripts/r/05_stockfish_benchmark.R` | Added `engine_nn` ENGINE_MODULE option; branches on NN: loads `.pt` file via `load_model()` instead of weights from DB |
| `scripts/r/07_human_play.R` | **Major additions**: engine_nn option in ENGINES + UI; `session_nn_model`, `session_nn_optimizer`, `session_nn_attrs` reactives; `is_nn_engine()` / `get_engine_params()` helpers; NN-aware `record_position()`, `update_eval_history()`, `play_engine_move()`; NN New Game init (loads `.pt`, creates optimizer, creates run); NN `end_game()` (calls Python `nn_td_update()` via reticulate); `log_human_game_nn()` (saves `.pt` + logs attribution weights); post-game diagnostics: Frobenius norm value box, norm vs self-play baseline comparison, session norm history chart; NN-aware live feature table (attribution × value) |
| `scripts/r/03_visualize.qmd` | Added "Stockfish Benchmark — ELO Estimates" section: ELO formula (display equation), flextable of all benchmark results, trajectory plot with 95% Wilson CI error bars |
| `scripts/r/seed_supabase.R` | **New** — one-time script: creates `nn_models` table in Supabase; seeds run_id=2 (learnable material from local run_id=21) and run_id=3 (NN from local run_id=24) |
| `public_app/app.R` | **Rewritten** — multi-condition support (2 conditions live: hardcoded material + learnable material); Supabase session pooler (IPv4, `aws-0-us-east-1.pooler.supabase.com`); condition selector UI; dynamic engine loading; separate linear/NN game paths; NN removed pending CPU-only torch solution |
| `public_app/engine_learnable_material.py` | **New** — copy of learnable material engine for deployment |
| `ChessLearningDocumentation.md` | Major update: brought current from Session 12 to Session 18; updated engine conditions, feature tables, weight state, file descriptions, open questions, key decisions |
| `AGENTS.md` | Added `ChessLearningDocumentation.md` to session-end procedure (Step 4); renumbered subsequent steps |

### Bugs Found / Fixed

- **`07_human_play.R`**: `session_weights()` was NULL for NN sessions; `feature_table` and `play_engine_move()` tried to call `as.list(NULL)` → fixed via `get_engine_params()` helper
- **`engine_nn.py` module caching**: `model_to_b64` added mid-session; module cached without it in R session → fixed via `importlib.reload()` in seeding script
- **Public app IPv6**: shinyapps.io couldn't reach Supabase direct host (`db.*.supabase.co` resolved to IPv6) → switched to session pooler (`aws-0-us-east-1.pooler.supabase.com`, IPv4) with project-scoped username (`postgres.betniftrhjpeacfzjoym`)
- **Public app Python crash (CUDA torch)**: `virtualenv_install(c("chess", "torch"))` on Linux installs CUDA torch (~554MB GPU packages) → removed NN condition from public app; only `chess` installed; NN deployment requires CPU-only torch solution
- **Public app game start disconnect**: app still crashes when starting a game — not yet resolved; to debug next session

### Overnight Job

- **NN self-play run_id=24**: RESUME=True, N_GAMES=400 (targeting game 500 total); started ~10pm; should complete by ~8am
- ELO benchmark for run_id=24 should be run after overnight completes

### Steps for Next Session

1. **Assess NN overnight run** — check decisive rate, attribution weights, pawn_advancement at ~500 games
2. **Benchmark NN at game ~500** — `05_stockfish_benchmark.R` with ENGINE_MODULE=`"engine_nn"`, RUN_ID=24
3. **Debug public app game-start disconnect** — check shinyapps.io logs; likely Python/reticulate issue on first game
4. **CPU-only torch for public NN deployment** — explore `requirements.txt` with `--index-url https://download.pytorch.org/whl/cpu` or Posit Connect
5. **Supabase `nn_models` table seeding** for run_id=3 already done; reseed after overnight run if weights improve meaningfully

---

## Session 19 — October 14, 2026

### What We Worked On

- Discussed ELO benchmark procedure for run_id=24 (neural net)
- Reviewed activity documentation (ChessLearningDocumentation.md, ChessLearningLog.md)
- Diagnosed and fixed the public app game-start crash
- Redeployed public app to shinyapps.io

### Key Findings

**ELO benchmark for run_id=24:** `05_stockfish_benchmark.R` already had `RUN_ID=24` but `ENGINE_MODULE` was incorrectly set to `"engine_learnable_material"` — corrected to `"engine_nn"`. CHECKPOINT=NULL (latest) and N_GAMES=20 are appropriate.

**Public app crash (game-start disconnect):**
- Root cause: gray Shiny disconnect overlay on game start = uncaught R error in the `btn_new_game` observer
- Primary culprit: `load_latest_weights()` called with no `tryCatch` — a Supabase connection failure propagated and killed the session
- Secondary: `record_position(b)` was outside the `tryCatch` block in `play_engine_move()`, so a Python error there was also unhandled
- Tertiary: `next_game_number()` and `log_linear_game()` in `end_game()` also lacked error handling

### Bugs Fixed

| File | Bug | Fix |
|---|---|---|
| `public_app/app.R` | `load_latest_weights()` in `btn_new_game` had no error handling — Supabase failure caused gray disconnect | Wrapped in `tryCatch`; falls back to zero weights + shows warning notification |
| `public_app/app.R` | `record_position(b)` was outside the `tryCatch` block in `play_engine_move()` | Moved inside `tryCatch` |
| `public_app/app.R` | `next_game_number()` in `end_game()` had no error handling | Wrapped in `tryCatch`; falls back to `1L` |
| `public_app/app.R` | `log_linear_game()` in `end_game()` had no error handling | Wrapped in `tryCatch`; shows warning notification on failure |

**Engine files confirmed identical** between `scripts/python/` and `public_app/` — no sync needed.

### Deployment

- Redeployed to https://dkillian.shinyapps.io/chess-learning/ via `rsconnect::deployApp()` with `appId=18172172`
- Note: using `appName="ChessLearning"` (wrong) caused a 402 error (app limit); correct `appName="chess-learning"` with `appId` resolves this

### Background Jobs at Session Close

Both Stockfish benchmark jobs in RStudio Jobs pane: **succeeded** (completed earlier this session).
No active self-play training jobs.

### Steps for Next Session

1. **Test the public app** — play a game at https://dkillian.shinyapps.io/chess-learning/; confirm game-start no longer disconnects; check if error notification appears (indicates DB issue worth diagnosing)
2. **Run ELO benchmark for run_id=24** — `ENGINE_MODULE` now set correctly to `"engine_nn"`; run as background job
3. **Assess NN run status** — check what game number the overnight run reached; assess attribution weights, decisive rate, pawn_advancement
4. **CPU-only torch for NN public deployment** — if game-start is now stable, consider adding NN condition via `requirements.txt` approach with `--index-url https://download.pytorch.org/whl/cpu`

---

## Session 22 — October 18–19, 2026

### What We Worked On

- Reviewed and corrected activity documentation: fixed session ordering in Log.md (Session 20 before 21), updated Run Registry (run_id=24 to closed, runs 27/28/29 added)
- Updated ChessLearningDocumentation.md: status header, Current Weight State (run_id=24 closed, run_id=27 added), Stage 02 pipeline, Folder Structure, File Descriptions, Open Questions
- Created four MCTS training notebooks implementing the AlphaZero Stage 2–5 progression:
  - `selfplay_mcts_v1.ipynb` — Stage 2: MCTS + ValueNet(input_dim=19), run_id=30
  - `selfplay_mcts_v2.ipynb` — Stage 3: MCTS + ValueNet(input_dim=768), run_id=31
  - `selfplay_mcts_v3.ipynb` — Stage 4: DualNet + dual loss (MSE + CE), run_id=32
  - `selfplay_mcts_v4.ipynb` — Stage 5: ResNet(5 blocks, 128 filters), run_id=33
- Fixed three successive schema errors discovered during v1 smoke test:
  1. `runs` table: no `hyperparameters` column — fixed to actual columns (`depth`, `learning_rate`, `notes`, `engine_version`)
  2. `start_game` variable dropped by repair script — restored in all four notebooks
  3. `game_records` and `feature_stats` use `(run_id, game_number)`, not `game_id`
- Ran 1-game smoke test of selfplay_mcts_v1: draw, 602 halfmoves, 5.5 min
- Started overnight run: run_id=30, N_GAMES=80, RESUME=True, ~midnight Oct 19

### Key Findings

**Schema corrections (all four mcts notebooks):**
- `runs`: `run_id, started_at, depth, learning_rate, move_limit, notes, last_updated, engine_version`
- `game_records`: `record_id, run_id, game_number, pgn, td_error` (no `game_id` foreign key)
- `feature_stats`: `stat_id, run_id, game_number, outcome, feature_name, mean_value` (no `game_id`)

**Smoke test — game 1 (run_id=30):**
- Outcome: draw, 602 halfmoves, 328 seconds (~5.5 min)
- All 19 feature means near zero (fresh model, expected)
- Training step executed (602 positions > BATCH_SIZE=64)
- Long game explained: 50-move counter resets on captures; random MCTS generates frequent captures

**Speed:** 5 min/game at N_SIMULATIONS=200 with ValueNet(19) → 80 games ≈ 7 hours overnight

**Weight checkpoint at game 10 not yet verified** — first fires tonight; check in morning

### Scripts Changed / Created

| File | Change |
|---|---|
| `ChessLearningLog.md` | Fixed session ordering; updated run_id=24; added runs 27/28/29 |
| `ChessLearningDocumentation.md` | Updated status, weight state, Stage 02 pipeline, Folder Structure, File Descriptions, Open Questions |
| `scripts/python/selfplay_mcts_v1.ipynb` | New: Stage 2 — MCTS + ValueNet(19), run_id=30 |
| `scripts/python/selfplay_mcts_v2.ipynb` | New: Stage 3 — MCTS + ValueNet(768), run_id=31 |
| `scripts/python/selfplay_mcts_v3.ipynb` | New: Stage 4 — DualNet + dual loss (MSE+CE), run_id=32 |
| `scripts/python/selfplay_mcts_v4.ipynb` | New: Stage 5 — ResNet(5 blocks, 128 filters), run_id=33 |

### Overnight Job

- **selfplay_mcts_v1.ipynb** — run_id=30, N_SIMULATIONS=200, N_GAMES=80, RESUME=True; started ~midnight Oct 19
- First weight checkpoint fires at game 10; no checkpoint yet before overnight

### Steps for Next Session

1. **Assess overnight run_id=30** — decisive rate, game lengths, loss trajectory; confirm game 10 weight checkpoint logged correctly
2. **Check weight signs at game 10, 20, etc.** — any learning signal from game-outcome targets?
3. **Update `05_stockfish_benchmark.R`** to support `ENGINE_MODULE="engine_nn_v2"` (flagged Session 21, still pending)
4. **Run ELO benchmark** on run_id=30 once overnight completes
5. **Consider hard move limit** (e.g., 200 halfmoves) for mcts notebooks to prevent very long games
6. **If v1 results positive**, plan v2 run (run_id=31, ValueNet(768)) for comparison

---

## Session 20 — October 15, 2026

### What We Worked On

- Reviewed activity documentation (`ChessLearningDocumentation.md`, `ChessLearningLog.md`)
- Ran status queries on the SQLite database
- Discovered run_id=24 reached **994+ games** (far beyond 500-game overnight target) and reviewed all four Stockfish benchmarks
- Discussed self-play collapse diagnosis and DeepMind / AlphaGo Zero solutions
- Implemented **past-checkpoint opponent mechanism** in `selfplay_nn.ipynb`
- Ran 10-game test tranche (games 996–1005) to verify the mechanism
- Updated all documentation files
- Initiated overnight tranche with past-opponent mechanism active

### Key Findings

**run_id=24 Stockfish benchmark trajectory — ELO is declining with training:**

| Checkpoint | W/L/D | ELO est. |
|---|---|---|
| Game 50 | 0/12/8 | ~1079 |
| Game 380 | 2/14/4 | ~1079 |
| Game 544 | 1/14/5 | ~1051 |
| Game 900 | 1/16/3 | ~982 |

ELO declining despite attribution metrics improving — classic self-play collapse. Network is overfitting to its own game distribution. Discussed AlphaGo Zero's solutions: MCTS training targets, past-checkpoint opponent pools, large replay buffer.

**Past-checkpoint opponent mechanism (implemented this session):**
- `play_game()` gains `opponent_model=None` parameter; learner randomly assigned WHITE or BLACK
- Positions returned as `(features, color, is_learner)` 3-tuples
- `td_update()` skips gradient updates on opponent positions but uses them as bootstrap targets
- `train()` gains `past_opponent_freq` and `past_opponent_checkpoints` params; loads `.pt` files once before loop
- Config: `PAST_OPPONENT_FREQ = 0.2`, `PAST_OPPONENT_CHECKPOINTS = [100, 200]`

**10-game test tranche (games 996–1005) — promising early signal:**
- **17/19 correct signs** at game 1005 — up from 15/19 at game 910 (had been stuck at 15/19 since game 200)
- Three persistent wrong-sign features corrected: `center_control` (−0.017 → +0.059), `rook_open_file` (−0.030 → +0.058), `connected_rooks` (−0.023 → +0.011)
- Still wrong: `bishop_pair` (−0.057), `isolated_pawn` (−0.004, borderline)
- Decisive rate: 50% (5/10) — down from 78% in pre-patch games 946–995; likely noise at n=10 but worth monitoring
- Weight changes are large (past-opponent games generate stronger TD signals than self-play, same effect as human play)

### Scripts Changed

| File | Change |
|---|---|
| `scripts/python/selfplay_nn.ipynb` | Implemented past-checkpoint opponent: `play_game()`, `td_update()`, `log_feature_stats()`, `log_position_logs()`, `train()`, config cell all updated |

### Overnight Job

- **NN self-play run_id=24**: RESUME=True, N_GAMES=250, PAST_OPPONENT_FREQ=0.2, checkpoints [100, 200]; started ~11pm
- Latest model before overnight: `nn_run_24_game_994.pt`; overnight will resume from game 995+
- Both checkpoint files confirmed present: `nn_run_24_game_100.pt`, `nn_run_24_game_200.pt`

### Steps for Next Session

1. **Assess overnight results** — check decisive rate, sign correctness, whether 17/19 holds after 250 more games
2. **Run ELO benchmark on game 1005+ checkpoint** — real test of whether past-opponent mechanism improves external playing strength
3. **Build `selfplay_sf_nn.ipynb`** — Stockfish teacher for the NN condition; mirrors `selfplay_nn.ipynb` but plays vs Stockfish at ELO 1320; TD(0)+Adam updates from those games
4. **Play human games** via `07_human_play.R` with NN engine to inject novel positions
5. **Reseed Supabase** — run `seed_supabase.R` to update run_id=3 (NN) with latest weights after overnight

---

## Session 21 — October 17, 2026

### What We Worked On

- Assessed run 24 completion (game 1305): ELO plateau confirmed at ~982; 12/19 correct signs; decided to treat as a closed result
- Human play run 26 (games 4–5): signs fluctuated 14→11→15/19; identified per-game seed not recorded in DB as a data quality gap
- Added `seed_run_id` and `seed_game_number` columns to `games` and `stockfish_games` tables
- Updated `07_human_play.R` and `05_stockfish_benchmark.R` to log seed columns; backfilled `stockfish_games` historical rows
- Created and ran `selfplay_sf_nn.ipynb` (SF teacher, plain Adam) — run_id=27; ran to 640 games
- Assessed run 27: ELO ~1105 at games 110 and 390 (above run 24's peak of ~1079), slipped to 1079 at game 440; material features converging strongly; sign count volatile (peaked 18/19 at game 340)
- Read the AlphaZero preprint (background/alphazero_preprint.pdf): identified L2 regularisation, LR schedule, temperature move selection, and resignation as actionable near-term improvements
- Created `selfplay_sf_nn_v2.ipynb` (run_id=29, SF teacher + AdamW + LR schedule + temperature + resignation)
- Created `selfplay_nn_v2.ipynb` (run_id=28, self-play + AdamW + LR schedule + resignation)
- Ran 10-game test tranches: run 28 (11/19 signs, 50% draws), run 29 (12/19 signs vs run 27's 17/19 at same milestone)
- Read `PositronAssistantConversations.md`: found a Stage 2–5 MCTS implementation roadmap and hand-off to implement `engine_nn_v2.py`
- Implemented and fully tested `engine_nn_v2.py` (all 13 checks passing)
- DB cleanup: deleted duplicate game_id=9924 (run 28 game_number=1)

### Key Findings

**run_id=24 (NN self-play) — confirmed collapse:**
- Game 1305 benchmark: 0W/15L/5D vs SF1320 → ELO ~982
- Trajectory: 1079 → 1079 → 1051 → 982 → 982 — stable floor, no recovery
- 12/19 correct signs at game 1305 (peaked 18/19 at game 700); run treated as complete

**run_id=27 (SF teacher, Adam) — genuine improvement over self-play:**
- ELO ~1105 at games 110 and 390 — above run 24's all-time peak (~1079)
- Slipped to 1079 at game 440 benchmark; 640 games total, no further benchmark taken
- Material features converging monotonically: material_queen +0.442 at game 440
- Sign count oscillated 11–18/19 (peaked at game 340); positional features unstable
- 640 game weights: 15/19 correct; wrong: bishop_pair, doubled_pawn, piece_development, connected_rooks

**v2 notebooks (runs 28 and 29) — AlphaZero improvements added:**
- `torch.optim.AdamW` with `weight_decay=1e-4` (L2 regularisation)
- `MultiStepLR` step-decay schedule: LR drops 10× at game milestones [200, 500]
- Temperature move selection for first 15 full moves (SF teacher only; self-play keeps epsilon-greedy)
- Resignation when engine value ≤ −0.9 for 4 consecutive engine half-moves
- Two bugs fixed: misplaced `consecutive_bad = 0` in game-over block; `MultiStepLR` `KeyError` from `last_epoch >= 0` on fresh optimizer (fixed with session-relative milestone conversion)

**engine_nn_v2.py — Stage 2–5 engine, all tests passing:**
- `board_to_tensor()`: 768-dim bitboard (12 planes × 64 squares, White's perspective)
- `MCTSNode`: PUCT UCB formula; root pre-expanded before simulation loop
- `mcts_search()`: works with ValueNet (uniform policy), DualNet, ResNet; `board_to_input` param for Stage 2 (19-feature) compatibility
- `ValueNet(input_dim, hidden)`: Stages 2–3; `input_dim=19` for Stage 2, `768` for Stage 3
- `DualNet(input_dim, hidden)`: Stage 4; policy head with masked softmax over legal moves
- `ResNet(input_dim, n_filters, n_blocks)`: Stage 5; 5 residual blocks, 128 filters
- `attribution_weights()`: plane-aggregated (12 values) for 768-dim; `policy_top5` key for Dual/ResNet
- `save_model` / `load_model` / `model_to_b64` / `model_from_b64`: same interface as `engine_nn.py`

### Scripts Changed / Created

| File | Change |
|---|---|
| `data/chess_learning.db` | Added `seed_run_id`, `seed_game_number` to `games` and `stockfish_games`; deleted duplicate row |
| `scripts/r/07_human_play.R` | `log_human_game()` and `log_human_game_nn()` now log seed columns; call sites pass `source_run_info()` |
| `scripts/r/05_stockfish_benchmark.R` | Logs `seed_run_id` / `seed_game_number`; filename fix: tries `nn_sf_run_*` before `nn_run_*` |
| `scripts/python/selfplay_sf_nn.ipynb` | New: SF teacher (Adam); fixed double-logging bug; removed move limit |
| `scripts/python/selfplay_sf_nn_v2.ipynb` | New: SF teacher (AdamW + LR schedule + temperature + resignation); run_id=29 |
| `scripts/python/selfplay_nn_v2.ipynb` | New: self-play (AdamW + LR schedule + resignation); run_id=28 |
| `scripts/python/engine_nn_v2.py` | New: Stage 2–5 engine (board_to_tensor, MCTS, ValueNet, DualNet, ResNet) |
| `scripts/python/update_v2_notebooks.py` | Utility script used to patch v2 notebooks |

### Steps for Next Session

1. **Run 29 (SF teacher v2) overnight** — 200+ game tranche; benchmark at game 100
2. **Run 28 (self-play v2) overnight** — parallel; benchmark at game 100
3. **Compare run 28/29 sign trajectories against run 24/27** at game 100 to test AdamW effect
4. **Write `selfplay_mcts_v1.ipynb`** (Stage 2) — MCTS + `ValueNet(input_dim=19)` + replay buffer; coordinate with Positron assistant
5. **Update `05_stockfish_benchmark.R`** for `engine_nn_v2` module name
6. **Assess run 27 game 640 ELO** — no benchmark taken past game 440; run one before continuing
