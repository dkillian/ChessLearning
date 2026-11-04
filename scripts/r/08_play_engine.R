# 08_play_engine.R
# Play a chosen engine in a Shiny GUI. Each finished game makes one TD(0)
# update to the learnable-material evaluation and is logged to the database.
#
# Engines:
#   mate_pursuit  mate-in-1 and mate-in-2 threats first; the TD-learned
#                 evaluation chooses the move when no threat exists
#   td_greedy     1-ply greedy on the TD-learned evaluation (baseline)
#
# Run: shiny::runApp("scripts/r/08_play_engine.R")

library(shiny)
library(bslib)
library(DBI)
library(RSQLite)
library(tidyverse)
library(reticulate)

# ---- Configuration ----

PY_TD_PATH <- if (file.exists("../python/Temporal Difference/engine_learnable_material.py")) {
    "../python/Temporal Difference"
} else {
    "scripts/python/Temporal Difference"
}
PY_MP_PATH <- if (file.exists("../python/Mate Pursuit/mate_pursuit.py")) {
    "../python/Mate Pursuit"
} else {
    "scripts/python/Mate Pursuit"
}
DB_PATH <- if (file.exists("../../data/chess_learning.db")) {
    "../../data/chess_learning.db"
} else {
    "data/chess_learning.db"
}
DEFAULT_LR      <- 0.001
DRAW_LR_SCALES  <- list(king_safety = 0.05, center_control = 0.2)
ENGINE_VERSION  <- "mate_pursuit_v1"

# ---- Python modules ----

py_run_string("import sys; [sys.modules.pop(m, None) for m in ['engine_learnable_material', 'mate_pursuit']]")

chess_mod  <- import("chess")
chess_svg  <- import("chess.svg")
random_mod <- import("random")
td_engine  <- import_from_path("engine_learnable_material", path = PY_TD_PATH)
mp_engine  <- import_from_path("mate_pursuit", path = PY_MP_PATH)

# ---- Helpers: database ----

`%||%` <- function(a, b) if (!is.null(a)) a else b

db_connect <- function() dbConnect(SQLite(), DB_PATH)

# Linear TD runs with learnable-material weights. Excludes human-play runs and
# neural-net or MCTS runs, whose weights are not linear TD weights.
get_source_runs <- function() {
    con <- db_connect()
    on.exit(dbDisconnect(con))
    dbGetQuery(con,
        "SELECT DISTINCT w.run_id, r.notes FROM weights w
         JOIN runs r ON r.run_id = w.run_id
         WHERE w.feature_name = 'material_queen'
           AND r.notes LIKE '%learnable_material%'
           AND r.notes NOT LIKE 'Human play%'
         ORDER BY w.run_id DESC")
}

get_checkpoints <- function(run_id) {
    con <- db_connect()
    on.exit(dbDisconnect(con))
    dbGetQuery(con, sprintf(
        "SELECT DISTINCT game_number FROM weights WHERE run_id = %d ORDER BY game_number DESC",
        as.integer(run_id)))$game_number
}

load_weights_vec <- function(run_id, game_number) {
    con <- db_connect()
    on.exit(dbDisconnect(con))
    rows <- dbGetQuery(con, sprintf(
        "SELECT feature_name, weight_value FROM weights WHERE run_id = %d AND game_number = %d",
        as.integer(run_id), as.integer(game_number)))
    setNames(rows$weight_value, rows$feature_name)
}

# New run seeded from a checkpoint; notes start with "Human play" so the
# existing TD pickers in 07_human_play.R skip it
create_run <- function(source_run_id, source_checkpoint, wv) {
    con <- db_connect()
    on.exit(dbDisconnect(con))
    notes <- sprintf("Human play - mate pursuit engine, seeded from run_id=%d, game=%d",
                     source_run_id, source_checkpoint)
    dbExecute(con,
        "INSERT INTO runs (started_at, depth, learning_rate, move_limit, notes, engine_version)
         SELECT datetime('now'), depth, learning_rate, move_limit, ?, ?
         FROM runs WHERE run_id = ? ORDER BY run_id LIMIT 1",
        list(notes, ENGINE_VERSION, source_run_id))
    new_id <- as.integer(dbGetQuery(con, "SELECT last_insert_rowid()")[[1]])
    for (fname in names(wv)) {
        dbExecute(con,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES (?, 0, ?, ?)",
            list(new_id, fname, wv[[fname]]))
    }
    new_id
}

log_game <- function(run_id, game_number, outcome, n_half, elapsed, td_result,
                     pgn_str, human_color, seed_run_id, seed_game_number) {
    con <- db_connect()
    on.exit(dbDisconnect(con))
    dbExecute(con,
        "INSERT INTO games (run_id, game_number, outcome, n_halfmoves, duration_s,
                            terminated_by, human_color, seed_run_id, seed_game_number)
         VALUES (?, ?, ?, ?, ?, 'natural', ?, ?, ?)",
        list(run_id, game_number, outcome, n_half, elapsed, human_color,
             seed_run_id, seed_game_number))
    dbExecute(con,
        "INSERT INTO game_records (run_id, game_number, pgn, td_error) VALUES (?, ?, ?, ?)",
        list(run_id, game_number, pgn_str, td_result$td_error))
    for (fname in names(td_result$deltas)) {
        dbExecute(con,
            "INSERT INTO weight_deltas (run_id, game_number, feature_name, delta)
             VALUES (?, ?, ?, ?)",
            list(run_id, game_number, fname, td_result$deltas[[fname]]))
    }
    for (fname in names(td_result$weights)) {
        dbExecute(con,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES (?, ?, ?, ?)",
            list(run_id, game_number, fname, td_result$weights[[fname]]))
    }
}

# ---- Helpers: game logic ----

build_pgn <- function(san) {
    n <- length(san)
    if (n == 0) return("")
    nums  <- ceiling(seq_len(n) / 2)
    parts <- ifelse(seq_len(n) %% 2 == 1, paste0(nums, ". ", san), san)
    paste(parts, collapse = " ")
}

outcome_for_color <- function(outcome, is_white) {
    if      (outcome == "1-0") { if (is_white) 1.0 else -1.0 }
    else if (outcome == "0-1") { if (!is_white) 1.0 else -1.0 }
    else 0.0
}

record_position <- function(board) {
    feats <- td_engine$extract_features(board)
    fv    <- setNames(as.numeric(unlist(feats)), names(feats))
    list(features = fv,
         names    = names(fv),
         is_white = board$turn == chess_mod$WHITE)
}

run_td_update <- function(positions, outcome, weights_vec,
                          lr = DEFAULT_LR, draw_lr_scales = DRAW_LR_SCALES) {
    n       <- length(positions)
    is_draw <- outcome == "1/2-1/2"

    values <- sapply(positions, function(p) sum(p$features * weights_vec[p$names]))
    deltas <- setNames(numeric(length(weights_vec)), names(weights_vec))
    errors <- numeric(n)

    for (t in seq_len(n)) {
        p      <- positions[[t]]
        target <- if (t < n) -values[t + 1L] else outcome_for_color(outcome, p$is_white)
        err    <- target - values[t]
        errors[t] <- err
        for (fname in p$names) {
            eff_lr <- lr * (if (is_draw) (draw_lr_scales[[fname]] %||% 1.0) else 1.0)
            deltas[[fname]] <- deltas[[fname]] + eff_lr * err * p$features[[fname]]
        }
    }

    new_weights <- weights_vec
    for (fname in names(deltas)) {
        new_weights[[fname]] <- max(-50, min(50, weights_vec[[fname]] + deltas[[fname]]))
    }

    list(weights = new_weights, deltas = deltas,
         td_error = mean(abs(errors)), errors = errors)
}

make_eval_fn <- function(wv) {
    wd <- r_to_py(as.list(wv))
    function(b) as.numeric(td_engine$evaluate(b, wd))
}

# Describes why the mate-pursuit engine chose a move (call before the move is played)
move_kind <- function(board, mv) {
    board$push(mv)
    mated <- board$is_checkmate()
    board$pop()
    if (mated) return("mate in 1")
    s <- mp_engine$threat_score(board, mv)
    if (s > 0) sprintf("threat; %.2f of replies keep mate in 1", s) else "evaluation fallback"
}

choose_engine_move <- function(board, engine_key, wv, rng, state = character()) {
    ev <- make_eval_fn(wv)
    new_state <- character()
    if (engine_key == "pursuit") {
        res       <- mp_engine$choose_pursuit(board, rng, as.list(state))
        mv        <- res[[1]]
        new_state <- unlist(res[[2]]) %||% character()
        kind      <- res[[3]]
    } else if (engine_key == "optimistic") {
        res  <- mp_engine$pursuit_choice(board, rng)
        mv   <- res[[1]]
        n    <- as.integer(res[[2]])
        kind <- if (n > 0) sprintf("optimistic mate, %d plies", n) else "random, no mate found"
    } else if (engine_key == "robust") {
        res  <- mp_engine$robust_choice(board, rng)
        mv   <- res[[1]]
        s    <- as.numeric(res[[2]])
        kind <- if (s > 0) sprintf("robust mate threat, %.2f of replies keep it", s) else "random, no mate found"
    } else if (engine_key == "mate_pursuit") {
        mv   <- mp_engine$choose_move(board, rng, ev)
        kind <- move_kind(board, mv)
    } else {
        mv   <- mp_engine$greedy_move(board, rng, ev)
        kind <- "1-ply evaluation"
    }
    list(move = mv, san = board$san(mv), kind = kind, state = new_state)
}

# Keep the known mate line only if the human followed it
advance_pursuit <- function(state, human_uci) {
    unlist(mp_engine$advance_state(as.list(state), human_uci)) %||% character()
}

# Engine chat lines, chosen from the engine's own state
narrate_engine <- function(ch, engine_key) {
    kind  <- ch$kind
    state <- ch$state
    if (engine_key == "pursuit") {
        if (startsWith(kind, "pursuing")) {
            if (length(state) == 0) return("And that's the mate. Don't take it personally.")
            n <- ceiling(length(state) / 2)
            return(sprintf("Pursuing known mate line, %d %s left. Will you cooperate?",
                           n, if (n == 1) "move" else "moves"))
        }
        if (startsWith(kind, "new mate line")) {
            n <- (length(state) + 2) / 2
            return(sprintf("Mate spotted in %d moves. Please don't defend it.", n))
        }
        return(sample(c("No mate in sight, so I'm improvising.",
                        "Nothing forced here. Let's see what happens.",
                        "Out of tricks, so a random move it is."), 1))
    }
    if (engine_key == "optimistic") {
        if (startsWith(kind, "optimistic mate")) {
            return(sprintf("Optimistic search found a %s. Cooperation required.",
                           sub("optimistic mate, ", "mate in ", kind)))
        }
        return(sample(c("The optimistic search came up empty. Random it is."), 1))
    }
    if (engine_key == "robust") {
        if (startsWith(kind, "robust mate threat")) {
            return(sprintf("Robust threat: %s. Let's see if it holds.",
                           sub("robust mate threat, ", "", kind)))
        }
        return(sample(c("No robust threat here. Random move."), 1))
    }
    sprintf("My evaluation likes this one. I trust my weights, mostly.")
}

# ---- JavaScript: chessboard.js drag-and-drop ----

board_js <- HTML("
$(document).ready(function () {
  var chess = new Chess();
  var board = null;
  window.humanTurn = false;

  function onDragStart(source, piece) {
    if (!window.humanTurn) return false;
    if (piece.charAt(0) !== window.humanColor) return false;
    return true;
  }

  function onDrop(source, target) {
    if (!window.humanTurn) return 'snapback';
    var move = chess.move({ from: source, to: target, promotion: 'q' });
    if (move === null) return 'snapback';
    window.humanTurn = false;
    Shiny.setInputValue('human_move_drop',
      { uci: source + target + (move.promotion || ''), nonce: Date.now() },
      { priority: 'event' });
  }

  function onSnapEnd() { board.position(chess.fen()); }

  board = Chessboard('chessboard', {
    draggable: true,
    position: 'start',
    onDragStart: onDragStart,
    onDrop: onDrop,
    onSnapEnd: onSnapEnd,
    pieceTheme: 'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
    snapbackSpeed: 300,
    snapSpeed: 100
  });

  Shiny.addCustomMessageHandler('update_board', function (msg) {
    chess.load(msg.fen);
    board.position(msg.fen, false);
    board.orientation(msg.orientation);
    window.humanTurn = msg.human_turn;
    window.humanColor = msg.human_color;

    window.prevHL = window.prevHL || [];
    window.prevHL.forEach(function (sq) {
      $('#chessboard .square-' + sq).css('background', '');
    });
    window.prevHL = [];
    [msg.lm_from, msg.lm_to].forEach(function (sq) {
      if (sq) {
        $('#chessboard .square-' + sq).css('background', 'rgba(255,214,0,0.4)');
        window.prevHL.push(sq);
      }
    });
    if (msg.check_sq) {
      $('#chessboard .square-' + msg.check_sq).css('background', 'rgba(220,0,0,0.45)');
      window.prevHL.push(msg.check_sq);
    }
  });

  $(window).resize(function () { board.resize(); });
});
")

# ---- UI ----

source_runs    <- get_source_runs()
source_choices <- setNames(
    as.character(source_runs$run_id),
    sprintf("run %d: %s", source_runs$run_id, substr(source_runs$notes, 1, 60))
)

ui <- page_sidebar(
    title = "Play an engine",
    theme = bs_theme(preset = "cosmo"),
    sidebar = sidebar(
        width = 300,
        radioButtons("engine", "Engine", selected = "pursuit",
            choices = c("Known mate-line pursuit (random fallback)" = "pursuit",
                        "Robust mate search (random fallback)"     = "robust",
                        "Optimistic mate search (random fallback)" = "optimistic",
                        "Mate pursuit (TD fallback)"               = "mate_pursuit",
                        "TD greedy (1-ply)"                        = "td_greedy")),
        selectInput("source_run", "Source run", choices = source_choices),
        uiOutput("checkpoint_ui"),
        radioButtons("human_color", "You play",
                     choices = c("White" = "white", "Black" = "black"), inline = TRUE),
        actionButton("start", "Start new game", class = "btn-primary"),
        hr(),
        p("Each finished game makes one TD update and is logged to chess_learning.db.")
    ),
    tags$head(
        tags$link(rel = "stylesheet",
                  href = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"),
        tags$script(src = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"),
        tags$script(src = "https://unpkg.com/chess.js@0.10.3/chess.js"),
        tags$script(board_js)
    ),

    layout_columns(
        col_widths = c(6, 6),
        card(
            card_header("Board (drag a piece, or use the move list)"),
            div(id = "chessboard", style = "width:420px; max-width:100%;")
        ),
        card(
            card_header("Your move"),
            selectInput("move_san", NULL, choices = character(0)),
            actionButton("play", "Play move"),
            textOutput("status"),
            textOutput("engine_last"),
            hr(),
            strong("Moves"),
            textOutput("move_list"),
            hr(),
            textOutput("td_summary"),
            hr(),
            strong("Engine chat"),
            uiOutput("chat"),
            hr(),
            strong("Engine chat"),
            uiOutput("chat")
        )
    )
)

# ---- Server ----

server <- function(input, output, session) {
    rng <- random_mod$Random(as.integer(6841))

    rv <- reactiveValues(
        board        = NULL,
        active       = FALSE,
        outcome      = NULL,
        positions    = list(),
        sans         = character(),
        weights      = NULL,
        run_id       = NULL,
        run_source   = NULL,
        seed_run     = NULL,
        seed_checkpoint = NULL,
        game_number  = 0L,
        human_color  = NULL,
        start_time   = NULL,
        last_engine  = "",
        pursuit      = character(),
        chat         = character(),
        td_summary   = ""
    )

    post_chat <- function(msg) {
        rv$chat <- c(rv$chat, msg)
    }

    # Note a human move: report a broken mate line, then update the line
    after_human <- function(uci) {
        if (length(rv$pursuit) > 0 && rv$pursuit[[1]] != uci) {
            post_chat("Mate line lost, searching for another.")
        }
        rv$pursuit <- advance_pursuit(rv$pursuit, uci)
    }

    finish_game <- function(outcome) {
        rv$active  <- FALSE
        rv$outcome <- outcome
        gn  <- rv$game_number + 1L
        res <- run_td_update(rv$positions, outcome, rv$weights)
        rv$weights     <- res$weights
        rv$game_number <- gn
        hc <- if (rv$human_color == chess_mod$BLACK) "black" else "white"
        elapsed <- as.numeric(difftime(Sys.time(), rv$start_time, units = "secs"))
        log_game(rv$run_id, gn, outcome, length(rv$sans), elapsed, res,
                 build_pgn(rv$sans), hc, rv$seed_run, rv$seed_checkpoint)
        engine_is_white <- rv$human_color == chess_mod$BLACK
        post_chat(if (outcome == "1/2-1/2") {
            "A draw. I'll take it, reluctantly."
        } else if ((outcome == "1-0") == engine_is_white) {
            "Checkmate. Good game?"
        } else {
            "You got me. Rematch?"
        })
        rv$td_summary <- sprintf("Game %d logged (%s). Mean |TD error| %.3f.",
                                 gn, outcome, res$td_error)
    }

    engine_turn <- function() {
        b <- rv$board
        rv$positions <- c(rv$positions, list(record_position(b)))
        ch <- choose_engine_move(b, input$engine, rv$weights, rng, rv$pursuit)
        rv$pursuit <- ch$state
        nb <- b$copy()
        nb$push(ch$move)
        rv$board       <- nb
        rv$sans        <- c(rv$sans, ch$san)
        rv$last_engine <- paste0(ch$san, " (", ch$kind, ")")
        post_chat(narrate_engine(ch, input$engine))
        if (nb$is_game_over(claim_draw = TRUE)) {
            finish_game(nb$result(claim_draw = TRUE))
        }
    }

    output$checkpoint_ui <- renderUI({
        req(input$source_run)
        cps <- get_checkpoints(as.integer(input$source_run))
        selectInput("checkpoint", "Checkpoint (game number)",
                    choices = cps, selected = max(cps))
    })

    observeEvent(input$start, {
        req(input$source_run, input$checkpoint)
        src <- as.integer(input$source_run)
        cp  <- as.integer(input$checkpoint)
        if (is.null(rv$run_id) || !identical(rv$run_source, c(src, cp))) {
            wv <- load_weights_vec(src, cp)
            rv$weights         <- wv
            rv$run_id          <- create_run(src, cp, wv)
            rv$run_source      <- c(src, cp)
            rv$seed_run        <- src
            rv$seed_checkpoint <- cp
            rv$game_number     <- 0L
        }
        rv$human_color <- if (input$human_color == "white") chess_mod$WHITE else chess_mod$BLACK
        rv$board       <- chess_mod$Board()
        rv$positions   <- list()
        rv$sans        <- character()
        rv$active      <- TRUE
        rv$outcome     <- NULL
        rv$last_engine <- ""
        rv$start_time  <- Sys.time()
        rv$pursuit     <- character()
        rv$chat        <- "Game on. Let's see what you're made of."
        if (rv$board$turn != rv$human_color) engine_turn()
    })

    observe({
        b <- rv$board
        req(b, rv$active)
        if (isTRUE(b$turn == rv$human_color)) {
            sans <- vapply(reticulate::iterate(b$legal_moves), \(m) b$san(m), character(1))
            updateSelectInput(session, "move_san", choices = sans)
        }
    })

    observeEvent(input$play, {
        req(rv$active, rv$board)
        b <- rv$board
        req(isTRUE(b$turn == rv$human_color))
        mv  <- b$parse_san(input$move_san)
        after_human(mv$uci())
        rv$positions <- c(rv$positions, list(record_position(b)))
        san <- b$san(mv)
        nb  <- b$copy()
        nb$push(mv)
        rv$board       <- nb
        rv$sans        <- c(rv$sans, san)
        rv$last_engine <- ""
        if (nb$is_game_over(claim_draw = TRUE)) {
            finish_game(nb$result(claim_draw = TRUE))
        } else {
            engine_turn()
        }
    })

    # Push the current position and highlights to the chessboard.js board
    send_board <- function() {
        b <- rv$board
        if (is.null(b)) return(invisible(NULL))
        last_uci <- if (length(b$move_stack) > 0) b$peek()$uci() else NULL
        check_sq <- if (b$is_check()) chess_mod$square_name(b$king(b$turn)) else NULL
        session$sendCustomMessage("update_board", list(
            fen         = b$fen(),
            orientation = if (isTRUE(rv$human_color == chess_mod$BLACK)) "black" else "white",
            human_turn  = isTRUE(rv$active) && isTRUE(b$turn == rv$human_color),
            human_color = if (isTRUE(rv$human_color == chess_mod$BLACK)) "b" else "w",
            lm_from     = if (is.null(last_uci)) NULL else substr(last_uci, 1, 2),
            lm_to       = if (is.null(last_uci)) NULL else substr(last_uci, 3, 4),
            check_sq    = check_sq
        ))
        invisible(NULL)
    }

    observe({
        req(rv$board)
        send_board()
    })

    observeEvent(input$human_move_drop, {
        req(rv$active, rv$board)
        b  <- rv$board
        mv <- tryCatch(chess_mod$Move$from_uci(input$human_move_drop$uci),
                       error = function(e) NULL)
        if (is.null(mv) || !isTRUE(b$turn == rv$human_color) || !isTRUE(b$is_legal(mv))) {
            send_board()
            return()
        }
        after_human(mv$uci())
        rv$positions   <- c(rv$positions, list(record_position(b)))
        san            <- b$san(mv)
        nb             <- b$copy()
        nb$push(mv)
        rv$board       <- nb
        rv$sans        <- c(rv$sans, san)
        rv$last_engine <- ""
        if (nb$is_game_over(claim_draw = TRUE)) {
            finish_game(nb$result(claim_draw = TRUE))
        } else {
            engine_turn()
        }
    })

    output$status <- renderText({
        if (!is.null(rv$outcome) && !rv$active) return(paste("Game over:", rv$outcome))
        if (is.null(rv$board)) return("Choose a source run and start a game.")
        if (isTRUE(rv$board$turn == rv$human_color)) "Your move." else "Engine thinking."
    })

    output$engine_last <- renderText({
        if (nzchar(rv$last_engine)) paste("Engine played:", rv$last_engine) else ""
    })

    output$move_list <- renderText(build_pgn(rv$sans))

    output$td_summary <- renderText(rv$td_summary)

    output$chat <- renderUI({
        if (length(rv$chat) == 0) return(p("Start a game to hear from the engine."))
        div(style = "max-height:260px; overflow-y:auto;",
            lapply(rv$chat, function(m) p(m)))
    })
}

shinyApp(ui, server)
