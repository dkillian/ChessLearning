# 05_move_scorer.R
# Score all legal moves from a given board position using the trained engine.
#
# Configure RUN_ID, DEPTH, and FEN at the top, then source the file.
# FEN = NULL uses the starting position.
# FEN = a FEN string jumps to any position.
#
# Run from project root or scripts/r/.

library(DBI)
library(RSQLite)
library(reticulate)

# ---- Configuration ----

RUN_ID <- 12
DEPTH  <- 3
FEN    <- NULL   # NULL = starting position
                 # e.g. "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"

# ---- Setup ----

DB_PATH     <- if (file.exists("../../data/chess_learning.db")) "../../data/chess_learning.db" else "data/chess_learning.db"

ENGINE_PATH <- if (file.exists("../python/engine.py")) "../python" else "scripts/python"

engine  <- import_from_path("engine", path = ENGINE_PATH)
chess_m <- import("chess")

# ---- Load weights ----

con <- dbConnect(SQLite(), DB_PATH)

wt_rows <- dbGetQuery(con, sprintf("
    SELECT feature_name, weight_value
    FROM weights
    WHERE run_id = %d
      AND game_number = (SELECT MAX(game_number) FROM weights WHERE run_id = %d)
", RUN_ID, RUN_ID))

dbDisconnect(con)

if (nrow(wt_rows) == 0) stop("No weights found for run_id = ", RUN_ID)

weights_py <- as.list(setNames(wt_rows$weight_value, wt_rows$feature_name))

cat(sprintf("Weights loaded: run_id=%d, %d features\n", RUN_ID, nrow(wt_rows)))

# ---- Set up board ----

board <- if (is.null(FEN)) chess_m$Board() else chess_m$Board(FEN)
cat("Position FEN:", board$fen(), "\n")
cat("Side to move:", if (board$turn == chess_m$WHITE) "White" else "Black", "\n\n")

# ---- Score all legal moves ----

score_moves <- function(board, weights_py, depth) {
    moves <- reticulate::iterate(board$legal_moves)
    if (length(moves) == 0) return(tibble(san = character(), uci = character(), score = numeric()))

    purrr::map(moves, function(m) {
        san <- board$san(m)
        uci <- m$uci()
        board$push(m)
        score <- -engine$negamax(board, as.integer(depth - 1), -1e6, 1e6, weights_py)
        board$pop()
        tibble(san = san, uci = uci, score = score)
    }) |>
        purrr::list_rbind() |>
        mutate(score = as.numeric(score)) |>
        arrange(desc(score)) |>
        mutate(rank = row_number())
}

cat(sprintf("Scoring %d legal moves at depth %d...\n",
            length(reticulate::iterate(board$legal_moves)), DEPTH))

scored <- score_moves(board, weights_py, DEPTH)

# ---- Display ----

cat(sprintf("\nBest move:  %s  (score = %+.4f)\n", scored$san[1], scored$score[1]))
cat(sprintf("Worst move: %s  (score = %+.4f)\n", scored$san[nrow(scored)], scored$score[nrow(scored)]))
cat(sprintf("Score spread: %.4f\n\n", scored$score[1] - scored$score[nrow(scored)]))

scored |>
    select(Rank = rank, Move = san, UCI = uci, Score = score) |>
    flextable() |>
    colformat_double(j = "Score", digits = 4) |>
    bg(i = ~ Rank == 1, bg = "#d4edda") |>
    flextable::set_caption(sprintf("Move scores — run_id=%d, depth=%d", RUN_ID, DEPTH)) |>
    autofit()

# specific move ---- 

m <- chess_m$Move$from_uci("g1f3")

# What does the board look like before the move?
cat("Side to move:", if (board$turn == chess_m$WHITE) "White" else "Black", "\n")
cat("Move (UCI):", m$uci(), "\n")
cat("Move (SAN):", board$san(m), "\n")


# Step 2: Push the move onto the board
board$push(m)

cat("Side to move after Nf3:", if (board$turn == chess_m$WHITE) "White" else "Black", "\n")
cat("It's now Black's turn — negamax will score from Black's perspective\n")
board

# Step 3: Call negamax — searching depth-1 levels deeper from this position
# DEPTH=3, so we pass depth-1 = 2, meaning negamax looks 2 more plies ahead
# (Black's response, then White's reply)

raw_score <- engine$negamax(board, as.integer(DEPTH - 1), -1e6, 1e6, weights_py)
cat("Raw negamax score (Black's perspective):", raw_score, "\n")

# Step 4: Negate — flip back to White's perspective

# "A score of -0.078 for Black means +0.078 for White"
final_score <- -raw_score
cat("Final score (White's perspective):", final_score, "\n")
cat("This matches the scored table:", scored$score[scored$san == "Nf3"], "\n")


# Step 5: Pop the move — restore board for the next move's evaluation
board$pop()
cat("Side to move after pop:", if (board$turn == chess_m$WHITE) "White" else "Black", "\n")
cat("Board restored to starting position\n")

board$move_stack   # should be empty
board              # should show starting position




