# 06_search_mechanics.R
# Step-by-step walkthrough of board evaluation and negamax move selection.
#
# Run section by section (Ctrl+Enter) to trace how a move score is built
# from raw features through the full depth-3 search tree.
#
# Assumes the session already has engine, chess_m, weights_py loaded
# (source 05_move_scorer.R first, or run the setup block below).

library(reticulate)
library(DBI)
library(RSQLite)

# ---- Setup (skip if already loaded) ----

DB_PATH     <- if (file.exists("../../data/chess_learning.db")) "../../data/chess_learning.db" else "data/chess_learning.db"

ENGINE_PATH <- if (file.exists("../python/engine.py")) "../python" else "scripts/python"

if (!exists("engine"))     engine  <- import_from_path("engine", path = ENGINE_PATH)
if (!exists("chess_m"))    chess_m <- import("chess")
if (!exists("weights_py")) {
    con     <- dbConnect(SQLite(), DB_PATH)
    wt_rows <- dbGetQuery(con, "
        SELECT feature_name, weight_value FROM weights
        WHERE run_id = 12
          AND game_number = (SELECT MAX(game_number) FROM weights WHERE run_id = 12)")
    dbDisconnect(con)
    weights_py <- as.list(setNames(wt_rows$weight_value, wt_rows$feature_name))
}

# We'll work from this position throughout: White has just played 1. Nf3
board <- chess_m$Board()
board$push(chess_m$Move$from_uci("g1f3"))
cat("Position: after 1. Nf3 — Black to move\n")
board


# ===========================================================================
# PART 1: Board Evaluation
# How does evaluate(board, weights) turn a position into a number?
# ===========================================================================

# ---- 1a. Extract raw feature values ----

# extract_features() returns a dict of named chess concept scores.
# Each value is the net score for White (White's count minus Black's count).

raw_features <- engine$extract_features(board)

features_df  <- tibble(
    feature = names(raw_features),
    value   = as.numeric(unlist(raw_features))
)
cat("\nRaw feature values (White minus Black, after 1. Nf3):\n")
print(features_df, n = Inf)

# From black's point of view, 
# 1. Nf3 does the following: 
        # attacks two central squares (-2 for black)
        # develops a piece (-1 for black)
        # increases piece mobility (-.2 for black)



# ---- 1b. Join features with learned weights ----

weights_df <- tibble(
    feature = names(weights_py),
    weight  = as.numeric(unlist(weights_py))
)

eval_df <- left_join(features_df, weights_df, by = "feature") |>
    mutate(contribution = value * weight) |>
    arrange(desc(abs(contribution)))

cat("\nFeature × weight contributions:\n")
eval_df |>
    select(Feature = feature, Value = value, Weight = weight, Contribution = contribution) |>
    flextable() |>
    colformat_double(j = c("Value", "Weight", "Contribution"), digits = 4) |>
    bg(i = ~ Contribution > 0, j = "Contribution", bg = "#d4edda") |>
    bg(i = ~ Contribution < 0, j = "Contribution", bg = "#f8d7da") |>
    autofit()


# ---- 1c. Sum contributions → evaluate() ----

# evaluate() also adds hard-coded material balance (not a learnable weight).
# The positional dot product alone:
positional_score <- sum(eval_df$contribution)
cat("\nPositional dot product (features · weights):", round(positional_score, 5), "\n")

# Full evaluate() — includes material balance
# Returned from the perspective of the side TO MOVE (here: Black)
eval_score <- engine$evaluate(board, weights_py)
cat("engine$evaluate()  (Black's perspective):  ", round(eval_score, 5), "\n")
cat("Material + positional together explains the full score.\n")
cat("Negated (White's perspective):             ", round(-eval_score, 5), "\n\n")


# ===========================================================================
# PART 2: Negamax at Depth 0
# No search — evaluate() IS the score.
# ===========================================================================

cat("─── depth = 0 ───────────────────────────────────────────────────────\n")
cat("No moves searched. Score = evaluate(board, weights) for side to move.\n")
cat("After 1. Nf3, it's Black to move.\n")
cat("evaluate() =", round(engine$evaluate(board, weights_py), 4),
    "(Black's perspective)\n\n")


# ===========================================================================
# PART 3: Negamax at Depth 1
# Black makes every legal move, evaluate() is called on each result.
# Black picks the move with the highest score (from Black's perspective).
# ===========================================================================

cat("─── depth = 1 ───────────────────────────────────────────────────────\n")
cat("Black makes a move, then evaluate() is called immediately.\n")
cat("No White response is considered yet.\n\n")

black_moves <- reticulate::iterate(board$legal_moves)

d1 <- purrr::map(black_moves, function(bm) {
    san <- board$san(bm)
    board$push(bm)
    # After bm, White is to move. evaluate() returns score for White.
    # Negate → score from Black's perspective.
    s <- -as.numeric(engine$evaluate(board, weights_py))
    board$pop()
    tibble(black_move = san, score_for_black = round(s, 4))
}) |> purrr::list_rbind() |> arrange(desc(score_for_black))

cat("All Black responses scored (depth = 1):\n")
print(d1, n = Inf)

best_black_d1 <- d1$black_move[1]
best_score_d1 <- d1$score_for_black[1]
cat("\nBlack picks:", best_black_d1, "→ score for Black:", best_score_d1, "\n")
cat("Negated → score for White after 1. Nf3:", -best_score_d1, "\n\n")
cat("This is negamax(depth=1). Compare to negamax(depth=1) direct call:\n")
cat("engine$negamax(depth=1):",
    round(engine$negamax(board, 1L, -1e6, 1e6, weights_py), 4), "\n\n")


# ===========================================================================
# PART 4: Negamax at Depth 2 — one branch in detail
# Take Black's best depth-1 move (above), then let White respond.
# White picks the reply that maximises White's score.
# Black's score for that move = -(White's best reply score).
# ===========================================================================

cat("─── depth = 2, one branch ───────────────────────────────────────────\n")
cat("Branch: 1. Nf3", best_black_d1, "\n")
cat("Now White searches one level deeper before evaluating.\n\n")

# Push Black's best move to examine White's replies
board$push(chess_m$Move$from_uci(
    d1$black_move[1] |>
        (\(san) {
            moves <- reticulate::iterate(board$legal_moves)
            m_match <- Filter(\(mv) board$san(mv) == san, moves)[[1]]
            m_match$uci()
        })()
))

white_moves <- reticulate::iterate(board$legal_moves)

d2_branch <- purrr::map(white_moves, function(wm) {
    san <- board$san(wm)
    board$push(wm)
    # After wm, Black is to move. evaluate() = score for Black.
    # Negate → score for White.
    s <- -as.numeric(engine$evaluate(board, weights_py))
    board$pop()
    tibble(white_reply = san, score_for_white = round(s, 4))
}) |> purrr::list_rbind() |> arrange(desc(score_for_white))

cat("White's replies after 1. Nf3", best_black_d1, "(top 5 and bottom 3):\n")
print(bind_rows(head(d2_branch, 5), tail(d2_branch, 3)))

best_white_d2  <- d2_branch$white_reply[1]
best_score_d2  <- d2_branch$score_for_white[1]
cat("\nWhite picks:", best_white_d2, "→ score for White:", best_score_d2, "\n")
cat("Black's score for playing", best_black_d1, "= -(White's best) =",
    -best_score_d2, "\n")

board$pop()   # undo Black's move, back to after 1. Nf3

cat("\n")
cat("Compare to engine$negamax(depth=2) called on the Nf3 position:\n")
cat("engine$negamax(depth=2):",
    round(engine$negamax(board, 2L, -1e6, 1e6, weights_py), 4), "\n")
cat("(May differ slightly — depth=2 searches ALL of Black's moves,\n")
cat(" not just the depth-1 best. Alpha-beta may also re-order results.)\n\n")


# ===========================================================================
# PART 5: Full depth = 3 — what score_moves() computes
# The actual score reported for Nf3 in 05_move_scorer.R.
# ===========================================================================

cat("─── depth = 3 (full) ────────────────────────────────────────────────\n")
cat("engine$negamax(depth=3) from the starting position, after pushing Nf3:\n")
cat("1. Nf3 → Black responds (depth=2 search) → White replies (depth=1)\n")
cat("   → evaluate() called at each leaf.\n\n")

# Board is currently after 1. Nf3 (Black to move)
full_score_black <- engine$negamax(board, 3L, -1e6, 1e6, weights_py)
cat("negamax(depth=3) from Black's perspective:", round(full_score_black, 4), "\n")
cat("Negated → White's score for playing Nf3: ", round(-full_score_black, 4), "\n")
cat("\nThis is the number that appears in the scored table.\n")

board$pop()   # restore to starting position
cat("\nBoard restored to starting position.\n")


# ===========================================================================
# PART 6: Depth-0 evaluation of all legal moves from the starting position
# A flat table showing what evaluate() returns immediately after each move,
# before any search. Compare to the depth-3 scores in 05_move_scorer.R.
# ===========================================================================

cat("─── depth = 0: all 20 starting moves with feature breakdown ─────────\n")

board      <- chess_m$Board()
moves      <- reticulate::iterate(board$legal_moves)
feat_names <- names(engine$extract_features(board))   # canonical feature order

# One row per move: score + every feature value after that move
move_features <- purrr::map(moves, function(mv) {
    san <- board$san(mv)
    board$push(mv)
    s     <- as.numeric(engine$evaluate(board, weights_py))
    feats <- as.numeric(unlist(engine$extract_features(board)))
    board$pop()
    bind_cols(
        tibble(move = san, score = round(-s, 4)),
        setNames(as.list(round(feats, 3)), feat_names) |> as_tibble()
    )
}) |>
    purrr::list_rbind() |>
    arrange(desc(score))

# Weights row prepended so value x weight is easy to compute by eye
weights_row <- bind_cols(
    tibble(move = "weight", score = NA_real_),
    setNames(
        as.list(round(as.numeric(unlist(weights_py[feat_names])), 4)),
        feat_names
    ) |> as_tibble()
)

raw_eval_tbl <- bind_rows(weights_row, move_features) |>
    select(move, score, mobility, center_control, piece_development, pawn_advancement,
           king_safety, everything()) 

raw_eval_tbl |> 
    flextable() |>
    colformat_double(digits = 4) |>
    bg(i = ~ is.na(score),    j = c("move", "score", feat_names), bg = "#e9ecef") |>
    color(i = ~ is.na(score), j = c("move", "score", feat_names), color = "#555555") |>
    flextable::italic(i = ~ is.na(score), j = c("move", "score", feat_names)) |>
    bg(i = ~ !is.na(score) & score > 0,  j = "score", bg = "#d4edda") |>
    bg(i = ~ !is.na(score) & score < 0,  j = "score", bg = "#f8d7da") |>
    bg(i = ~ !is.na(score) & score == 0, j = "score", bg = "#fff3cd") |>
    flextable::set_caption("Depth-0 evaluation: all 20 starting moves. Row 1 = weights.") |>
    autofit()

