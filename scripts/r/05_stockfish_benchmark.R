# 05_stockfish_benchmark.R
# Run ELO benchmark matches between the trained engine and Stockfish.
# Results are logged to the stockfish_evals table in the DB.
#
# HOW TO RUN (keeps your R console free):
#   RStudio → Jobs pane → Start Job → select this file
#   Or: rstudioapi::jobRunScript("scripts/r/05_stockfish_benchmark.R")
#
# Configure the four parameters below, then run.

library(DBI)
library(RSQLite)
library(reticulate)

# ---- Configuration ----

RUN_ID      <- 14          # which run's weights to use
CHECKPOINT  <- NULL        # NULL = latest checkpoint; or set an integer game number
ELO_LEVELS  <- 1320        # Stockfish ELO levels to test (minimum is 1320)
N_GAMES     <- 10          # games per ELO level (even number = balanced colours)
DEPTH       <- 3L          # search depth for our engine

# ---- Setup ----

DB_PATH <- if (file.exists("../../data/chess_learning.db")) {
    "../../data/chess_learning.db"
} else {
    "data/chess_learning.db"
}

SF_PATH <- local({
    hits <- list.files(
        file.path(Sys.getenv("LOCALAPPDATA"), "Microsoft", "WinGet", "Packages"),
        pattern = "stockfish.*\\.exe$", recursive = TRUE, full.names = TRUE
    )
    if (length(hits)) hits[1] else stop(
        "Stockfish not found. Install via: winget install Stockfish.Stockfish"
    )
})

ENGINE_PATH <- if (file.exists("../python/engine.py")) "../python" else "scripts/python"

chess_mod    <- import("chess")
chess_pgn    <- import("chess.pgn")
chess_engine <- import("chess.engine")
io_mod       <- import("io")
engine_mod   <- import_from_path("engine", path = ENGINE_PATH)

# ---- Load weights ----

con <- dbConnect(SQLite(), DB_PATH)

if (is.null(CHECKPOINT)) {
    CHECKPOINT <- dbGetQuery(con,
        sprintf("SELECT MAX(game_number) as gn FROM weights WHERE run_id = %d",
                as.integer(RUN_ID)))$gn
}

wt_rows <- dbGetQuery(con, sprintf(
    "SELECT feature_name, weight_value FROM weights
     WHERE run_id = %d AND game_number = %d",
    as.integer(RUN_ID), as.integer(CHECKPOINT)))

dbDisconnect(con)

if (nrow(wt_rows) == 0) stop("No weights found for run_id=", RUN_ID, ", checkpoint=", CHECKPOINT)
weights_py <- as.list(setNames(wt_rows$weight_value, wt_rows$feature_name))

cat(sprintf("Weights loaded: run_id=%d, checkpoint=%d, %d features\n",
            RUN_ID, CHECKPOINT, nrow(wt_rows)))

# ---- Create stockfish_games table if needed ----

con_setup <- dbConnect(SQLite(), DB_PATH)
dbExecute(con_setup, "
    CREATE TABLE IF NOT EXISTS stockfish_games (
        game_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id       INTEGER,
        game_number  INTEGER,
        sf_elo       INTEGER,
        game_index   INTEGER,
        engine_color TEXT,
        result       TEXT,
        engine_won   INTEGER,
        n_half       INTEGER,
        pgn          TEXT,
        played_at    TEXT DEFAULT (datetime('now'))
    )")
dbDisconnect(con_setup)

# ---- Helpers ----

build_pgn_str <- function(sans, result, engine_color, sf_elo) {
    white_str <- if (engine_color == chess_mod$WHITE)
        sprintf("Our Engine (run_id=%d, game=%d)", RUN_ID, CHECKPOINT) else
        sprintf("Stockfish %d", sf_elo)
    black_str <- if (engine_color == chess_mod$BLACK)
        sprintf("Our Engine (run_id=%d, game=%d)", RUN_ID, CHECKPOINT) else
        sprintf("Stockfish %d", sf_elo)
    header <- sprintf(
        '[Event "Stockfish Benchmark"]\n[White "%s"]\n[Black "%s"]\n[Result "%s"]\n',
        white_str, black_str, result)
    if (length(sans) == 0) return(paste0(header, "\n", result))
    pairs     <- split(sans, ceiling(seq_along(sans) / 2))
    moves_str <- paste(mapply(
        function(p, i) paste0(i, ". ", paste(p, collapse = " ")),
        pairs, seq_along(pairs)), collapse = " ")
    paste0(header, "\n", moves_str, " ", result)
}

# ---- Game function ----

play_one_game <- function(sf_inst, engine_color, move_limit = 80L) {
    board  <- chess_mod$Board()
    n_half <- 0L
    sans   <- character(0)   # collect SAN for PGN

    while (!board$is_game_over() && n_half < move_limit) {
        mv <- if (board$turn == engine_color) {
            engine_mod$get_best_move(board, weights_py, DEPTH, FALSE)[[1]]
        } else {
            sf_inst$play(board, chess_engine$Limit(time = 0.5), ponder = FALSE)$move
        }
        if (is.null(mv)) break
        sans  <- c(sans, board$san(mv))   # capture SAN before push
        board$push(mv)
        n_half <- n_half + 1L
    }

    res <- board$result()
    list(
        result     = res,
        n_half     = n_half,
        sans       = sans,
        engine_won = switch(res,
            "1-0" = engine_color == chess_mod$WHITE,
            "0-1" = engine_color == chess_mod$BLACK,
            NA
        )
    )
}

# ---- Run benchmark ----

cat(sprintf("\nBenchmark: run_id=%d | checkpoint=%d | depth=%d | %d ELO levels | %d games each\n",
            RUN_ID, CHECKPOINT, DEPTH, length(ELO_LEVELS), N_GAMES))
cat(paste(rep("-", 60), collapse = ""), "\n")

t_total <- proc.time()[["elapsed"]]

for (elo in ELO_LEVELS) {
    cat(sprintf("\nELO %d — opening Stockfish...\n", elo))

    sf_inst <- tryCatch(
        chess_engine$SimpleEngine$popen_uci(SF_PATH),
        error = function(e) stop("Failed to open Stockfish: ", conditionMessage(e))
    )
    sf_inst$configure(list("UCI_LimitStrength" = TRUE, "UCI_Elo" = as.integer(elo)))

    wins <- losses <- draws <- 0L
    t_elo <- proc.time()[["elapsed"]]

    for (i in seq_len(N_GAMES)) {
        engine_color <- if (i %% 2 == 1L) chess_mod$WHITE else chess_mod$BLACK
        color_label  <- if (engine_color == chess_mod$WHITE) "White" else "Black"
        t_game <- proc.time()[["elapsed"]]

        result <- tryCatch(
            play_one_game(sf_inst, engine_color),
            error = function(e) {
                cat(sprintf("  Game %d ERROR: %s\n", i, conditionMessage(e)))
                list(result = "1/2-1/2", n_half = 0L, engine_won = NA)
            }
        )

        if (isTRUE(result$engine_won))        wins   <- wins   + 1L
        else if (isFALSE(result$engine_won))  losses <- losses + 1L
        else                                   draws  <- draws  + 1L

        outcome_label <- if (isTRUE(result$engine_won)) "WIN"
                         else if (isFALSE(result$engine_won)) "LOSS" else "DRAW"

        # Log individual game with PGN
        pgn_str <- build_pgn_str(result$sans, result$result, engine_color, elo)
        con_g   <- dbConnect(SQLite(), DB_PATH)
        dbExecute(con_g,
            "INSERT INTO stockfish_games
             (run_id, game_number, sf_elo, game_index, engine_color,
              result, engine_won, n_half, pgn)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(as.integer(RUN_ID), as.integer(CHECKPOINT), as.integer(elo),
                 as.integer(i), color_label,
                 result$result,
                 if (isTRUE(result$engine_won)) 1L else if (isFALSE(result$engine_won)) 0L else NA,
                 result$n_half, pgn_str))
        dbDisconnect(con_g)

        cat(sprintf("  Game %d/%d (%s) — %s in %d half-moves  [%.0fs]\n",
                    i, N_GAMES, color_label, outcome_label,
                    result$n_half, proc.time()[["elapsed"]] - t_game))
        flush.console()
    }

    sf_inst$quit()

    win_rate <- wins / N_GAMES
    cat(sprintf("  ELO %d result: %dW / %dL / %dD  win_rate=%.0f%%  [%.0fs total]\n",
                elo, wins, losses, draws, win_rate * 100,
                proc.time()[["elapsed"]] - t_elo))

    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO stockfish_evals
         (run_id, game_number, sf_elo, n_games, wins, losses, draws, win_rate, source)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        list(as.integer(RUN_ID), as.integer(CHECKPOINT), as.integer(elo),
             as.integer(N_GAMES), as.integer(wins), as.integer(losses),
             as.integer(draws), win_rate, "benchmark"))
    dbDisconnect(con)
    cat(sprintf("  Logged to DB.\n"))
}

cat(sprintf("\nBenchmark complete. Total time: %.1f min\n",
            (proc.time()[["elapsed"]] - t_total) / 60))
