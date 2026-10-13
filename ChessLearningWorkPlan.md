# Chess Learning — Strategic Review and Work Plan

*Prepared October 11, 2026 (Session 15) — updated October 12, 2026 (Session 16)*

---

## Where We've Come From

The project started September 29 and has run for 14 sessions across 12 days.
The arc of development falls into three phases:

**Phase 1 — Fighting the draw equilibrium (Sessions 1–8, runs 1–10)**
The engine started by drawing virtually every game. The root cause was a feedback
loop: near-zero weights → random play → draws → near-zero gradient → weights don't
develop. The fix came in layers:
- Epsilon-greedy exploration broke the immediate repetition pattern
- Monte Carlo → TD(0) eliminated draw erosion structurally
- Hard-coding material removed the largest source of gradient noise

**Phase 2 — Getting positional weights right (Sessions 8–12, runs 11–16)**
With material hard-coded and TD(0) in place, the 78% decisive breakthrough came
immediately. Depth=3 pushed it to 84% and corrected the `connected_rooks`
depth-2 artifact. The 14-feature engine (run_id=14) achieved 11/14 correct signs
and an estimated ELO ~1428 vs Stockfish 1320. The Stockfish-teacher experiment
(run_id=16) was tried and abandoned — it produced ELO regression and sign regression.

**Phase 3 — v2 engine and human play (Sessions 13–14, runs 17–19)**
Three feature redesigns addressed persistent wrong-sign features:
`backward_pawn` (tighter sentry definition), `king_safety` (pawn shelter − attacker
count), `pawn_advancement` (total advancement rather than single max rank).
The v2 engine was archived cleanly. Run_id=18 (380 games from zero) achieved
84.2% decisive and 11/14 correct signs. Human play app (`07_human_play.R`) and a
public crowdsourced app (`public_app/`) were built.

---

## Where We Are Now

*(Updated Session 16, October 12, 2026)*

**Engine state:**
- Two engine conditions in active development:
  - `engine.py` (hardcoded material, v2): 14 positional features — baseline for 30-run design
  - `engine_learnable_material.py` (learnable material): 19 features — experimental; `knight_outpost` replaced `knight_pst` in Session 16
- run_id=20 (learnable material, 510 games): 76.1% decisive; 16/19 correct signs; ELO ~1285; material weights plateaued (rook > queen)
- run_id=21 (learnable material, `knight_outpost`, from zero): first 50-game tranche in progress
- run_id=18 (hardcoded material v2 baseline): 84.2% decisive; 11/14 correct signs; ELO ~1355

**Resolved this session:**
- Double-insert bug in `07_human_play.R`: ✅ fixed (Session 15)
- Wrong-sign diagnostics: ✅ completed — three distinct failure modes identified (Session 16)
- `knight_pst` redesigned → `knight_outpost`: ✅ implemented (Session 16)
- Stockfish benchmark for learnable material: ✅ 2W/4L/14D, ELO ~1285 (Session 16)

**Still stale:**
- `ChessLearningDocumentation.md`: reflects Session 14 state
- `engine.ipynb`: not updated to v2 architecture

**Strategic position:**
The learnable material condition (run_id=20/21) is an active experiment running in parallel
with the hardcoded-material baseline. The 30-run experimental design has not yet started.
The key open question is whether the learnable material condition can match the hardcoded
baseline's performance — or whether we proceed with the 30-run design using hardcoded material
while continuing to explore learnable material separately.

---

## The Key Open Question

**Are we ready to begin the 30-run experimental design?**

Evidence FOR readiness:
- 84.2% decisive rate — stable across two major engine versions (run_id=12 and run_id=18)
- 11/14 correct signs — consistent across v1 (run_id=14, 400 games) and v2 (run_id=18, 380 games)
- TD(0) + hard-coded material is architecturally clean
- The three wrong-sign features have been persistent across many runs — if they haven't
  corrected by now, they may simply need more data (slow learners) rather than more architecture work

Evidence AGAINST:
- Run_id=18 is only 380 games from zero — we haven't yet seen what v2 does at 1000+ games
- `doubled_pawn` is a basic, well-established chess concept — a correct linear evaluator
  should learn it; persistent wrong sign is a flag
- `pawn_advancement` had a promising early positive sign under v2 but drifted
  negative by game 380 — the trajectory is worth understanding before locking in the design
- ELO for v2 is unknown (last benchmark was run_id=14)

**Recommendation:** One more focused session of diagnostics before committing to the 30-run design.

---

## Work Plan

### Priority 1 — Bug Fix (~30 min)

**Fix double-insert in `07_human_play.R`**
- `log_human_game` fires twice at game end
- Investigate whether `end_game()` is triggered from both the human-move and engine-move observers
- Delete the duplicate weight rows in DB for run_id=19 game 1 (28 rows → 14)

### Priority 2 — Diagnostics on the 3 Wrong-Sign Features (~1 session)

The three persistent wrong features need a diagnosis before we decide to move forward
or redesign. Three angles:

1. **Delta distribution by sign** — for `doubled_pawn`, do the weight updates
   skew positive or negative? Are there specific game-phase positions driving the wrong-sign updates?

2. **Trajectory inspection** — did any of the three show correct sign early in run_id=18
   and then cross zero (as `isolated_pawn` did)? If so, the engine is *learning then
   unlearning* — which points to a conflicting signal, not a missing signal.

3. **Feature distribution by outcome** — compare mean `doubled_pawn` value in winning vs
   losing positions. If the feature is genuinely confounded (e.g., the engine often wins
   despite having doubled pawns because of a material/mobility advantage), the wrong sign
   is expected. If there's a clear asymmetry in the right direction, it's a training signal problem.

### Priority 3 — Stockfish Benchmark on run_id=18 (~30 min to launch, overnight)

Run `05_stockfish_benchmark.R` against run_id=18 at game 380 (same config as run_id=14:
10 games at ELO 1320). This gives a direct v1 vs v2 ELO comparison and answers
whether the feature redesigns improved playing strength.

Target: beat or match the run_id=14 result (~1428 estimated ELO).

### Priority 4 — Longer v2 Self-Play Run (~1 session to launch, overnight)

Start run_id=20: epsilon-greedy self-play seeded from run_id=18 game 380,
targeting 1000 games. This serves two purposes:
- Determine whether wrong-sign features correct with more data at v2
- Generate a longer trajectory for `pawn_advancement` to see if it stabilizes positive

If signs improve substantially by game 1000, the architecture is ready.
If they stay wrong, document the limitation and decide: accept it or redesign once more.

### Priority 5 — Documentation Update (~30–45 min)

- Update `ChessLearningDocumentation.md`: current weight state, open questions, feature status
- Update `engine.ipynb` to v2 architecture
- Append session 15 log entry to `ChessLearningLog.md`

### Priority 6 — Decision Point: Begin 30-Run Experimental Design

**Trigger conditions** (all should be true before starting):
- Bug fix confirmed
- Benchmark complete (ELO known for v2)
- Longer run shows stable or improving sign correctness
- Architecture decision made on the 3 wrong-sign features (accept limitation or fix)

**If we proceed:** run_id=20 through run_id=49 (30 independent runs from zero,
same config, no resume). Each run targets 400–600 games. The primary analysis
output is a concept emergence timeline: for each feature and run, the first
checkpoint where the weight exceeds a threshold and holds.

**If we redesign one more time:** candidate changes:
- `doubled_pawn`: inspect feature computation; may be overcounting in certain structures
- `pawn_advancement`: the v2 design (total rank advancement ÷ 20) may be mixing
  endgame and middlegame signals in a way that confounds the gradient
- `knight_pst`: may need a larger PST range or different normalization

---

## Suggested Session Sequence

| Session | Primary Goal |
|---------|-------------|
| 15 | ✅ Bug fix + wrong-sign diagnostics + benchmark launch |
| 16 | ✅ Assess benchmark + wrong-sign diagnosis + `knight_outpost` redesign + launch run_id=21 |
| 17 | Assess run_id=21 at ~500 games; architecture decision (proceed to 30-run design or continue redesign) |
| 18 | Launch 30-run experimental design (hardcoded material baseline) and/or continue learnable material exploration |
| 19+ | Monitor first few experimental runs; build cross-run analysis in `03_visualize.qmd` |

---

## Longer-Horizon Items (Not Blocking)

- **Public app deployment** (`public_app/`): built but never deployed to shinyapps.io; Supabase backend configured; blocked only by the deployment step
- **`engine.ipynb` v2 update**: documentation only; doesn't affect training
- **30-run analysis tooling**: `03_visualize.qmd` will need new sections for cross-run emergence timelines and concept ordering consistency metrics; build as first runs accumulate
- **Softmax variant**: run_id=13 and 15 showed softmax is worse for decisive rate; could revisit after the 30-run self-play baseline is established
