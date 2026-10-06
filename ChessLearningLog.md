# Learning Study — Session Log

---

## Run Registry

Each new run is started when a substantive change is made to the training setup.
Minor changes within a run (e.g., tranche size, LR adjustment) continue the same run_id via `RESUME = True`.

| run_id | Date | Games | depth | lr | move_limit | weight_init | epsilon | Key Change |
|--------|------|-------|-------|----|------------|-------------|---------|------------|
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
| 12 | 2026-10-05 | — | 3 | 0.001 | 200 | zero (positional) | 0.1 | Depth=3; per-feature draw LR (king_safety=0.05); in progress at session close |

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
