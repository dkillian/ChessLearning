# 07_human_play.R
# Human (or external agent) plays our trained engine.
# Each game contributes a real TD(0) weight update to a new run seeded
# from a chosen source checkpoint.
#
# Board: interactive chessboard.js (drag-and-drop move entry).
# Right panel shows live diagnostics during play, then post-game learning
# analysis (weight deltas, learning moments, comparison to self-play).
#
# Run: shiny::runApp("scripts/r/07_human_play.R")

library(shiny)
library(bslib)
library(bsicons)
library(DBI)
library(RSQLite)
library(tidyverse)
library(reticulate)

# ---- Python modules ----

# ---- Configuration ----

ENGINE_PATH      <- if (file.exists("../python/Temporal Difference/engine.py")) "../python/Temporal Difference" else "scripts/python/Temporal Difference"
DEFAULT_SEED_RUN <- 18L   # fallback seed run shown in selector on first launch

# ---- Python modules ----

# Clear both cached engine modules so the app always loads latest from disk
py_run_string("import sys; [sys.modules.pop(m, None) for m in ['engine', 'engine_learnable_material', 'engine_nn', 'engine_nn_v2']]")

chess_mod <- import("chess")
chess_svg <- import("chess.svg")

ENGINES <- list(
    engine                    = import_from_path("engine", path = ENGINE_PATH),
    engine_learnable_material = tryCatch(
        import_from_path("engine_learnable_material", path = ENGINE_PATH),
        error = function(e) NULL
    ),
    engine_nn                 = tryCatch(
        import_from_path("engine_nn", path = ENGINE_PATH),
        error = function(e) NULL
    )
)

# ---- Paths ----

DB_PATH <- if (file.exists("../../data/chess_learning.db")) {
    "../../data/chess_learning.db"
} else {
    "data/chess_learning.db"
}

MODEL_DIR <- if (file.exists("../../models")) "../../models" else "models"

# ---- Helpers ----

`%||%` <- function(a, b) if (!is.null(a)) a else b

get_run_ids <- function() {
    con <- dbConnect(SQLite(), DB_PATH)
    ids <- dbGetQuery(con, "SELECT DISTINCT run_id FROM weights ORDER BY run_id DESC")$run_id
    dbDisconnect(con)
    ids
}

# Returns run_ids appropriate for the given engine type, excluding human-play runs
get_runs_for_engine <- function(engine_type) {
    con <- dbConnect(SQLite(), DB_PATH)
    if (engine_type == "engine_nn") {
        ids <- dbGetQuery(con,
            "SELECT DISTINCT r.run_id FROM runs r
             WHERE r.notes LIKE '%neural_net%'
               AND r.notes NOT LIKE 'Human play%'
             ORDER BY r.run_id DESC")$run_id
    } else if (engine_type == "engine_learnable_material") {
        ids <- dbGetQuery(con,
            "SELECT DISTINCT w.run_id FROM weights w
             JOIN runs r ON r.run_id = w.run_id
             WHERE w.feature_name = 'material_queen'
               AND r.notes NOT LIKE 'Human play%'
               AND r.notes NOT LIKE '%neural_net%'
             ORDER BY w.run_id DESC")$run_id
    } else {
        ids <- dbGetQuery(con,
            "SELECT DISTINCT w.run_id FROM weights w
             JOIN runs r ON r.run_id = w.run_id
             WHERE w.run_id NOT IN (
                 SELECT DISTINCT run_id FROM weights WHERE feature_name = 'material_queen'
             )
               AND r.notes NOT LIKE 'Human play%'
               AND r.notes NOT LIKE '%neural_net%'
             ORDER BY w.run_id DESC")$run_id
    }
    dbDisconnect(con)
    ids
}

# Returns run_id of the canonical v2 human-play run (oldest), or NULL if none exists
get_active_human_run <- function() {
    con  <- dbConnect(SQLite(), DB_PATH)
    rows <- dbGetQuery(con,
        "SELECT run_id FROM runs
         WHERE notes LIKE 'Human play%' AND engine_version = 'v2'
         ORDER BY run_id ASC LIMIT 1")
    dbDisconnect(con)
    if (nrow(rows) == 0) NULL else rows$run_id[1]
}

get_latest_game_number <- function(run_id) {
    con <- dbConnect(SQLite(), DB_PATH)
    gn  <- dbGetQuery(con, sprintf(
        "SELECT COALESCE(MAX(game_number), 0) AS gn FROM games WHERE run_id = %d",
        as.integer(run_id)))$gn
    dbDisconnect(con)
    as.integer(gn)
}

# Parse source run_id from notes: "Human play — seeded from run_id=X, game=Y"
parse_source_run_id <- function(run_id) {
    con   <- dbConnect(SQLite(), DB_PATH)
    notes <- dbGetQuery(con, sprintf(
        "SELECT notes FROM runs WHERE run_id = %d", as.integer(run_id)))$notes
    dbDisconnect(con)
    m <- regmatches(notes[1], regexpr("run_id=\\d+", notes[1]))
    if (length(m) == 0) return(run_id)
    as.integer(sub("run_id=", "", m))
}

get_checkpoints <- function(run_id) {
    con  <- dbConnect(SQLite(), DB_PATH)
    cpts <- dbGetQuery(con, sprintf(
        "SELECT DISTINCT game_number FROM weights WHERE run_id=%d ORDER BY game_number DESC",
        as.integer(run_id)))$game_number
    dbDisconnect(con)
    cpts
}

load_weights_vec <- function(run_id, game_number) {
    con  <- dbConnect(SQLite(), DB_PATH)
    rows <- dbGetQuery(con, sprintf(
        "SELECT feature_name, weight_value FROM weights WHERE run_id=%d AND game_number=%d",
        as.integer(run_id), as.integer(game_number)))
    dbDisconnect(con)
    setNames(rows$weight_value, rows$feature_name)
}

# Used for mini-boards in post-game learning moments
board_svg_html <- function(board, last_move = NULL, flipped = FALSE, size = 160L) {
    args <- list(board = board, size = size, flipped = flipped)
    if (!is.null(last_move)) args$lastmove <- last_move
    if (board$is_check()) {
        sq <- board$king(board$turn)
        if (!is.null(sq)) args$check <- sq
    }
    HTML(paste0('<div style="text-align:center;line-height:0;">',
                do.call(chess_svg$board, args), "</div>"))
}

outcome_for_color <- function(outcome, is_white) {
    if      (outcome == "1-0") { if (is_white) 1.0 else -1.0 }
    else if (outcome == "0-1") { if (!is_white) 1.0 else -1.0 }
    else 0.0
}

run_td_update <- function(positions, outcome, weights_vec,
                           lr = 0.001,
                           draw_lr_scales = list(king_safety    = 0.05,
                                                 center_control = 0.2)) {
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

create_new_run <- function(source_run_id, source_checkpoint, weights_vec) {
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO runs (started_at, depth, learning_rate, move_limit, notes, engine_version)
         SELECT datetime('now'), depth, learning_rate, move_limit,
                printf('Human play \u2014 seeded from run_id=%d, game=%d', run_id, ?),
                'v2'
         FROM runs WHERE run_id = ? ORDER BY run_id LIMIT 1",
        list(source_checkpoint, source_run_id))
    new_run_id <- as.integer(dbGetQuery(con, "SELECT last_insert_rowid()")[[1]])
    for (fname in names(weights_vec)) {
        dbExecute(con,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES (?, 0, ?, ?)",
            list(new_run_id, fname, weights_vec[[fname]]))
    }
    dbDisconnect(con)
    new_run_id
}

log_human_game <- function(run_id, game_number, outcome, n_half,
                            elapsed, td_result, pgn_str, human_color,
                            seed_run_id, seed_game_number) {
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO games (run_id,game_number,outcome,n_halfmoves,duration_s,terminated_by,human_color,seed_run_id,seed_game_number)
         VALUES (?,?,?,?,?,'natural',?,?,?)",
        list(run_id, game_number, outcome, n_half, elapsed, human_color, seed_run_id, seed_game_number))
    dbExecute(con,
        "INSERT INTO game_records (run_id,game_number,pgn,td_error)
         VALUES (?,?,?,?)",
        list(run_id, game_number, pgn_str, td_result$td_error))
    for (fname in names(td_result$deltas)) {
        dbExecute(con,
            "INSERT INTO weight_deltas (run_id,game_number,feature_name,delta)
             VALUES (?,?,?,?)",
            list(run_id, game_number, fname, td_result$deltas[[fname]]))
    }
    for (fname in names(td_result$weights)) {
        dbExecute(con,
            "INSERT INTO weights (run_id,game_number,feature_name,weight_value)
             VALUES (?,?,?,?)",
            list(run_id, game_number, fname, td_result$weights[[fname]]))
    }
    dbDisconnect(con)
}

# Returns run_id of existing NN human play run seeded from source_run_id, or NULL
get_active_nn_human_run <- function(source_run_id) {
    con  <- dbConnect(SQLite(), DB_PATH)
    rows <- dbGetQuery(con, sprintf(
        "SELECT run_id FROM runs
         WHERE notes LIKE 'Human play (neural_net)%%'
           AND notes LIKE '%%run_id=%d,%%'
         ORDER BY run_id ASC LIMIT 1",
        as.integer(source_run_id)))
    dbDisconnect(con)
    if (nrow(rows) == 0) NULL else rows$run_id[1]
}

create_new_nn_run <- function(source_run_id, source_checkpoint, attr_weights) {
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO runs (started_at, depth, learning_rate, move_limit, notes, engine_version)
         SELECT datetime('now'), depth, learning_rate, move_limit,
                printf('Human play (neural_net) — seeded from run_id=%d, game=%d', ?, ?),
                'v2'
         FROM runs WHERE run_id = ? ORDER BY run_id LIMIT 1",
        list(source_run_id, source_checkpoint, source_run_id))
    new_run_id <- as.integer(dbGetQuery(con, "SELECT last_insert_rowid()")[[1]])
    # Log initial attribution weights (compatible with R analysis scripts)
    for (fname in names(attr_weights)) {
        dbExecute(con,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES (?, 0, ?, ?)",
            list(new_run_id, fname, attr_weights[[fname]]))
    }
    dbDisconnect(con)
    new_run_id
}

log_human_game_nn <- function(run_id, game_number, outcome, n_half,
                               elapsed, nn_result, pgn_str, human_color,
                               model, model_dir,
                               seed_run_id, seed_game_number) {
    # Convert Python dicts to R named vectors
    deltas  <- setNames(as.numeric(unlist(nn_result$deltas)),
                        names(nn_result$deltas))
    attr_w  <- setNames(as.numeric(unlist(nn_result$attr_weights)),
                        names(nn_result$attr_weights))
    td_err  <- as.numeric(nn_result$td_error)

    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO games (run_id,game_number,outcome,n_halfmoves,duration_s,terminated_by,human_color,seed_run_id,seed_game_number)
         VALUES (?,?,?,?,?,'natural',?,?,?)",
        list(run_id, game_number, outcome, n_half, elapsed, human_color, seed_run_id, seed_game_number))
    dbExecute(con,
        "INSERT INTO game_records (run_id,game_number,pgn,td_error)
         VALUES (?,?,?,?)",
        list(run_id, game_number, pgn_str, td_err))
    for (fname in names(deltas)) {
        dbExecute(con,
            "INSERT INTO weight_deltas (run_id,game_number,feature_name,delta)
             VALUES (?,?,?,?)",
            list(run_id, game_number, fname, deltas[[fname]]))
    }
    # Attribution weights logged as "weights" — compatible with R analysis scripts
    for (fname in names(attr_w)) {
        dbExecute(con,
            "INSERT INTO weights (run_id,game_number,feature_name,weight_value)
             VALUES (?,?,?,?)",
            list(run_id, game_number, fname, attr_w[[fname]]))
    }
    dbDisconnect(con)
    # Save model checkpoint (.pt)
    model_path <- file.path(model_dir,
        sprintf("nn_run_%d_game_%d.pt", as.integer(run_id), as.integer(game_number)))
    ENGINES[["engine_nn"]]$save_model(model, model_path)
}

build_pgn <- function(san_moves) {
    pairs <- split(san_moves, ceiling(seq_along(san_moves) / 2))
    paste(mapply(function(p, i) paste0(i, ". ", paste(p, collapse = " ")),
                 pairs, seq_along(pairs)), collapse = " ")
}

# ---- Static data (none required; run is auto-detected on startup) ----

# ---- JavaScript: chessboard.js interactive board ----

board_js <- HTML("
$(document).ready(function () {
  var chess = new Chess();
  var board  = null;
  window.humanTurn  = false;
  window.humanColor = 'w';
  window.prevHL     = [];   // squares highlighted in previous update

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
    var uci = source + target + (move.promotion || '');
    Shiny.setInputValue('human_move_drop',
      { uci: uci, nonce: Date.now() }, { priority: 'event' });
  }

  function onSnapEnd() { board.position(chess.fen()); }

  board = Chessboard('chessboard', {
    draggable:     true,
    position:      'start',
    onDragStart:   onDragStart,
    onDrop:        onDrop,
    onSnapEnd:     onSnapEnd,
    pieceTheme:    'https://chessboardjs.com/img/chesspieces/wikipedia/{piece}.png',
    snapbackSpeed: 300,
    snapSpeed:     100
  });

  Shiny.addCustomMessageHandler('update_board', function (msg) {
    chess.load(msg.fen);
    board.position(msg.fen, false);
    board.orientation(msg.orientation);
    window.humanTurn  = msg.human_turn;
    window.humanColor = msg.human_color;

    // Clear previous square highlights
    window.prevHL.forEach(function (sq) {
      $('#chessboard .square-' + sq).css('background', '');
    });
    window.prevHL = [];

    // Last-move highlight (yellow)
    if (msg.lm_from) {
      $('#chessboard .square-' + msg.lm_from).css('background', 'rgba(255,214,0,0.4)');
      window.prevHL.push(msg.lm_from);
    }
    if (msg.lm_to) {
      $('#chessboard .square-' + msg.lm_to).css('background', 'rgba(255,214,0,0.4)');
      window.prevHL.push(msg.lm_to);
    }

    // Check highlight (red)
    if (msg.check_sq) {
      $('#chessboard .square-' + msg.check_sq).css('background', 'rgba(220,0,0,0.45)');
      window.prevHL.push(msg.check_sq);
    }
  });

  $(window).resize(function () { board.resize(); });
});
")

# ---- UI ----

ui <- page_sidebar(
    title = "Human vs Engine",
    theme = bs_theme(version = 5, bootswatch = "cosmo"),

    tags$head(
        tags$link(rel  = "stylesheet",
                  href = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"),
        tags$script(src = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"),
        tags$script(src = "https://unpkg.com/chess.js@0.10.3/chess.js"),
        tags$script(board_js)
    ),

    sidebar = sidebar(
        width = 280,

        card(
            card_header("Engine"),
            selectInput("ui_engine", "Engine type",
                choices = c(
                    "Hardcoded material" = "engine",
                    "Learnable material" = "engine_learnable_material",
                    "Neural net (MLP)"   = "engine_nn"
                ),
                selected = "engine"),
            selectInput("ui_seed_run",    "Seed run",   choices = NULL),
            selectInput("ui_checkpoint",  "Checkpoint", choices = NULL)
        ),

        card(
            card_header("Game"),
            uiOutput("session_info"),
            radioButtons("human_color", "I play as",
                choices = c("White", "Black"), inline = TRUE),
            sliderInput("depth", "Engine depth",
                min = 1, max = 3, value = 3, step = 1, width = "100%"),
            actionButton("btn_new_game", "New Game",
                class = "btn-success w-100 mb-2"),
            actionButton("btn_resign", "Resign",
                class = "btn-outline-danger btn-sm w-100")
        )
    ),

    layout_column_wrap(
        width = 1/3, fill = FALSE,
        value_box("Status",        uiOutput("vb_status"),
                  theme = "primary",   showcase = bs_icon("circle-fill")),
        value_box("Engine eval",   uiOutput("vb_eval"),
                  theme = "secondary", showcase = bs_icon("speedometer")),
        value_box("Session games", uiOutput("vb_game_count"),
                  theme = "light",     showcase = bs_icon("list-ol"))
    ),

    layout_columns(
        col_widths = c(5, 7),

        # ── Board card ──────────────────────────────────────────────────────
        card(
            full_screen = TRUE,
            card_header(uiOutput("board_header")),
            div(style = "display:flex; justify-content:center; padding:12px 8px 4px;",
                div(id = "chessboard", style = "width:380px;")
            ),
            uiOutput("move_list"),
            div(style = "padding:4px 8px 8px;",
                actionButton("btn_undo", "\u21a9 Undo",
                    class = "btn-outline-secondary btn-sm w-100")
            ),
            card_footer(
                style = "font-size:0.75em; color:#666; word-break:break-all;",
                textOutput("fen_display")
            )
        ),

        # ── Diagnostics card ────────────────────────────────────────────────
        card(
            full_screen = TRUE,
            card_header(uiOutput("diag_header")),
            uiOutput("diagnostics_panel")
        )
    )
)

# ---- Server ----

server <- function(input, output, session) {

    # ── Session state ──────────────────────────────────────────────────────

    session_run_id       <- reactiveVal(NULL)
    session_weights      <- reactiveVal(NULL)   # named numeric vec; linear engines only
    session_nn_model     <- reactiveVal(NULL)   # ChessNet object; engine_nn only
    session_nn_optimizer <- reactiveVal(NULL)   # Adam optimizer; engine_nn only
    game_number_rv       <- reactiveVal(0L)
    src_td_errors        <- reactiveVal(numeric(0))
    source_run_info      <- reactiveVal(list(run_id = NULL, checkpoint = NULL))
    active_engine_mod    <- reactiveVal(ENGINES[["engine"]])  # locked in on New Game

    # TRUE when the active session is using the neural net engine
    is_nn_engine <- function() !is.null(session_nn_model())

    # Returns the second argument for evaluate() / get_best_move():
    # the ChessNet model for NN sessions, or a named list for linear sessions.
    get_engine_params <- function() {
        if (is_nn_engine()) session_nn_model() else as.list(session_weights())
    }

    # NN-specific session state
    session_nn_attrs     <- reactiveVal(NULL)       # latest attribution weights (named vec)
    session_nn_param_norms <- reactiveVal(numeric(0)) # Frobenius norm per human game
    session_nn_sp_baseline <- reactiveVal(NA_real_)   # avg per-game norm from seed self-play

    # Pre-populate selectors on startup using auto-detected active run as hint
    .active_run <- get_active_human_run()
    .hint_src   <- if (!is.null(.active_run)) parse_source_run_id(.active_run) else DEFAULT_SEED_RUN
    .hint_eng   <- "engine"   # default; hardcoded material runs are the v2 baseline

    .init_runs <- get_runs_for_engine(.hint_eng)
    .init_cpts <- if (.hint_src %in% .init_runs) get_checkpoints(.hint_src) else
                  if (length(.init_runs) > 0) get_checkpoints(.init_runs[1]) else integer(0)
    .init_run_sel <- if (.hint_src %in% .init_runs) .hint_src else
                     if (length(.init_runs) > 0) .init_runs[1] else NULL

    updateSelectInput(session, "ui_seed_run",
        choices  = setNames(.init_runs, paste("Run", .init_runs)),
        selected = .init_run_sel)
    updateSelectInput(session, "ui_checkpoint",
        choices  = setNames(.init_cpts, paste("Game", .init_cpts)),
        selected = .init_cpts[1])

    # ── Selector reactives ─────────────────────────────────────────────────

    # Update seed run list when engine type changes
    observeEvent(input$ui_engine, {
        runs <- get_runs_for_engine(input$ui_engine)
        updateSelectInput(session, "ui_seed_run",
            choices  = setNames(runs, paste("Run", runs)),
            selected = runs[1])
    }, ignoreInit = TRUE)

    # Update checkpoint list when seed run changes
    observeEvent(input$ui_seed_run, {
        req(input$ui_seed_run)
        cpts <- get_checkpoints(as.integer(input$ui_seed_run))
        updateSelectInput(session, "ui_checkpoint",
            choices  = setNames(cpts, paste("Game", cpts)),
            selected = cpts[1])
    }, ignoreInit = TRUE)

    output$session_info <- renderUI({
        rid <- session_run_id()
        tags$small(style = "color:#888; display:block; margin-bottom:8px;",
            if (is.null(rid)) "No session — click New Game to start"
            else paste0("Run ", rid, " \u00b7 ", game_number_rv(), " games played"))
    })

    # ── Game state ─────────────────────────────────────────────────────────
    move_history <- reactiveVal(character(0))
    move_san_log <- reactiveVal(character(0))
    last_move_rv <- reactiveVal(NULL)
    human_color  <- reactiveVal(NULL)
    game_active  <- reactiveVal(FALSE)
    positions_rv <- reactiveVal(list())
    eval_history <- reactiveVal(numeric(0))
    game_result  <- reactiveVal(NULL)
    post_game_rv <- reactiveVal(NULL)
    game_start_t <- reactiveVal(NULL)

    current_board <- reactive({
        b <- chess_mod$Board()
        for (uci in move_history()) b$push(chess_mod$Move$from_uci(uci))
        b
    })

    output$vb_game_count <- renderUI({ game_number_rv() })

    # ── Board update helper ────────────────────────────────────────────────

    send_board_update <- function(is_human_turn) {
        b   <- current_board()
        hc  <- human_color()
        lm  <- last_move_rv()
        ori <- if (!is.null(hc) && hc == chess_mod$BLACK) "black" else "white"
        hcc <- if (!is.null(hc) && hc == chess_mod$BLACK) "b"     else "w"

        lm_from  <- if (!is.null(lm)) chess_mod$square_name(lm$from_square) else NULL
        lm_to    <- if (!is.null(lm)) chess_mod$square_name(lm$to_square)   else NULL
        check_sq <- if (b$is_check()) chess_mod$square_name(b$king(b$turn)) else NULL

        session$sendCustomMessage("update_board", list(
            fen        = b$fen(),
            human_turn = is_human_turn,
            orientation = ori,
            human_color = hcc,
            lm_from    = lm_from,
            lm_to      = lm_to,
            check_sq   = check_sq
        ))
    }

    # ── New game ──────────────────────────────────────────────────────────

    observeEvent(input$btn_new_game, {
        req(input$ui_seed_run, input$ui_checkpoint)

        # If this is the first game or a config change, initialise/find the run
        seed_run <- as.integer(input$ui_seed_run)
        seed_ckpt <- as.integer(input$ui_checkpoint)
        eng_key   <- input$ui_engine

        if (is.null(session_run_id())) {

            if (eng_key == "engine_nn") {
                # ── Neural net path ──────────────────────────────────────────
                nn_mod     <- ENGINES[["engine_nn"]]
                model_path <- file.path(MODEL_DIR,
                    sprintf("nn_run_%d_game_%d.pt", seed_run, seed_ckpt))
                if (!file.exists(model_path)) {
                    showNotification(
                        paste("Model checkpoint not found:", model_path),
                        type = "error", duration = 8)
                    return()
                }
                nn_model <- nn_mod$load_model(model_path)

                # Check for existing NN human play run seeded from this config
                .existing_nn <- get_active_nn_human_run(seed_run)
                if (!is.null(.existing_nn)) {
                    # Reload latest .pt from that run
                    .latest_gn  <- get_latest_game_number(.existing_nn)
                    .latest_path <- file.path(MODEL_DIR,
                        sprintf("nn_run_%d_game_%d.pt", .existing_nn, .latest_gn))
                    if (file.exists(.latest_path))
                        nn_model <- nn_mod$load_model(.latest_path)
                    session_run_id(.existing_nn)
                    game_number_rv(get_latest_game_number(.existing_nn))
                } else {
                    # Create new NN human play run seeded from attribution weights
                    attr_r <- setNames(
                        as.numeric(unlist(nn_mod$attribution_weights(nn_model))),
                        nn_mod$FEATURE_NAMES)
                    .new_run <- create_new_nn_run(seed_run, seed_ckpt, attr_r)
                    session_run_id(.new_run)
                    game_number_rv(0L)
                }
                session_nn_model(nn_model)
                session_nn_optimizer(nn_mod$make_optimizer(nn_model))
                session_weights(NULL)
                session_nn_attrs(NULL)
                session_nn_param_norms(numeric(0))
                source_run_info(list(run_id = seed_run, checkpoint = seed_ckpt))
                active_engine_mod(ENGINES[["engine_nn"]])

                # Populate src_td_errors from seed run for TD histogram
                .con_td <- dbConnect(SQLite(), DB_PATH)
                .sp_errs <- dbGetQuery(.con_td, sprintf(
                    "SELECT td_error FROM game_records WHERE run_id=%d AND td_error IS NOT NULL",
                    as.integer(seed_run)))$td_error
                dbDisconnect(.con_td)
                src_td_errors(.sp_errs)

                # Compute self-play baseline norm from seed run's last checkpoint interval
                .prev_ckpt <- seed_ckpt - 10L
                .path_a <- file.path(MODEL_DIR, sprintf("nn_run_%d_game_%d.pt", seed_run, .prev_ckpt))
                .path_b <- file.path(MODEL_DIR, sprintf("nn_run_%d_game_%d.pt", seed_run, seed_ckpt))
                if (file.exists(.path_a) && file.exists(.path_b)) {
                    .baseline <- tryCatch(
                        as.numeric(nn_mod$compute_baseline_norm(.path_a, .path_b, 10L)),
                        error = function(e) NA_real_)
                    session_nn_sp_baseline(.baseline)
                } else {
                    session_nn_sp_baseline(NA_real_)
                }

            } else {
                # ── Linear path (unchanged) ───────────────────────────────────
                .existing <- get_active_human_run()
                if (!is.null(.existing) && parse_source_run_id(.existing) == seed_run) {
                    .latest  <- get_checkpoints(.existing)[1]
                    .wv      <- load_weights_vec(.existing, .latest)
                    .gn      <- get_latest_game_number(.existing)
                    .src_id  <- seed_run
                    session_run_id(.existing)
                } else {
                    .wv      <- load_weights_vec(seed_run, seed_ckpt)
                    .new_run <- create_new_run(seed_run, seed_ckpt, .wv)
                    .gn      <- 0L
                    .src_id  <- seed_run
                    session_run_id(.new_run)
                }
                session_weights(.wv)
                session_nn_model(NULL)
                session_nn_optimizer(NULL)
                game_number_rv(.gn)
                source_run_info(list(run_id = .src_id, checkpoint = seed_ckpt))
                .con    <- dbConnect(SQLite(), DB_PATH)
                .errors <- dbGetQuery(.con, sprintf(
                    "SELECT td_error FROM game_records WHERE run_id=%d AND td_error IS NOT NULL",
                    as.integer(.src_id)))$td_error
                dbDisconnect(.con)
                src_td_errors(.errors)
                active_engine_mod(ENGINES[[eng_key]])
            }
        }

        col <- if (input$human_color == "White") chess_mod$WHITE else chess_mod$BLACK
        human_color(col)
        move_history(character(0))
        move_san_log(character(0))
        last_move_rv(NULL)
        positions_rv(list())
        eval_history(numeric(0))
        game_result(NULL)
        post_game_rv(NULL)
        game_active(TRUE)
        game_start_t(proc.time()[["elapsed"]])

        send_board_update(is_human_turn = col == chess_mod$WHITE)
        if (col == chess_mod$BLACK) play_engine_move()
    })

    # ── Resign ────────────────────────────────────────────────────────────

    observeEvent(input$btn_resign, {
        req(game_active())
        outcome <- if (human_color() == chess_mod$WHITE) "0-1" else "1-0"
        end_game(outcome)
    })

    # ── Position & eval recording ──────────────────────────────────────────

    record_position <- function(board) {
        feats_py <- active_engine_mod()$extract_features(board)
        feat_vec <- setNames(as.numeric(unlist(feats_py)), names(feats_py))
        pos_entry <- list(
            features = feat_vec,
            names    = names(feat_vec),
            is_white = board$turn == chess_mod$WHITE
        )
        # For NN: also store the raw Python feature dict and color for nn_td_update()
        if (is_nn_engine()) {
            pos_entry$py_feat  <- feats_py
            pos_entry$py_color <- board$turn
        }
        positions_rv(c(positions_rv(), list(pos_entry)))
    }

    update_eval_history <- function(board_after) {
        s <- as.numeric(active_engine_mod()$evaluate(board_after, get_engine_params()))
        white_eval <- if (board_after$turn == chess_mod$WHITE) s else -s
        eval_history(c(eval_history(), white_eval))
    }

    # ── Human move (drag-and-drop) ─────────────────────────────────────────

    observeEvent(input$human_move_drop, {
        req(game_active())
        uci <- input$human_move_drop$uci
        b   <- current_board()
        req(b$turn == human_color())

        # Server-side legality check
        legal_ucis <- sapply(reticulate::iterate(b$legal_moves), \(m) m$uci())
        req(uci %in% legal_ucis)

        mv  <- chess_mod$Move$from_uci(uci)
        san <- b$san(mv)
        record_position(b)
        move_history(c(move_history(), uci))
        move_san_log(c(move_san_log(), san))
        last_move_rv(mv)
        update_eval_history(current_board())

        if (current_board()$is_game_over(claim_draw = TRUE)) {
            end_game(current_board()$result(claim_draw = TRUE))
        } else {
            play_engine_move()
        }
    })

    # ── Undo ──────────────────────────────────────────────────────────────

    observeEvent(input$btn_undo, {
        req(game_active())
        h     <- move_history()
        n_pop <- min(2L, length(h))
        if (n_pop == 0) return()
        move_history(head(h, -n_pop))
        move_san_log(head(move_san_log(), -n_pop))
        last_move_rv(NULL)
        positions_rv(head(positions_rv(), -n_pop))
        eval_history(head(eval_history(), -n_pop))
        send_board_update(is_human_turn = TRUE)
    })

    # ── Engine move ────────────────────────────────────────────────────────

    play_engine_move <- function() {
        b  <- current_board()
        record_position(b)

        tryCatch({
            mv  <- active_engine_mod()$get_best_move(b, get_engine_params(), as.integer(input$depth), FALSE)[[1]]
            req(!is.null(mv))
            san <- b$san(mv)
            move_history(c(move_history(), mv$uci()))
            move_san_log(c(move_san_log(), san))
            last_move_rv(mv)
            update_eval_history(current_board())

            if (current_board()$is_game_over(claim_draw = TRUE)) {
                end_game(current_board()$result(claim_draw = TRUE))
            } else {
                send_board_update(is_human_turn = TRUE)
            }
        }, error = function(e) {
            showNotification(paste("Engine error:", conditionMessage(e)), type = "error")
        })
    }

    # ── End game ──────────────────────────────────────────────────────────

    end_game <- function(outcome) {
        game_active(FALSE)
        game_result(outcome)
        send_board_update(is_human_turn = FALSE)

        elapsed <- proc.time()[["elapsed"]] - (game_start_t() %||% 0)
        pos <- positions_rv()
        if (length(pos) == 0) return()
        gn  <- game_number_rv() + 1L
        game_number_rv(gn)
        hc  <- if (human_color() == chess_mod$BLACK) "black" else "white"

        if (is_nn_engine()) {
            # ── Neural net TD update ──────────────────────────────────────────
            model     <- session_nn_model()
            optimizer <- session_nn_optimizer()
            feat_dicts <- lapply(pos, function(p) p$py_feat)
            colors     <- lapply(pos, function(p) p$py_color)

            nn_result <- ENGINES[["engine_nn"]]$nn_td_update(
                model, optimizer, feat_dicts, colors, outcome, 0.1)

            session_nn_model(model)  # model mutated in-place; reassign for reactivity
            log_human_game_nn(session_run_id(), gn, outcome,
                              length(move_history()), elapsed,
                              nn_result, build_pgn(move_san_log()), hc,
                              model, MODEL_DIR,
                              source_run_info()$run_id, source_run_info()$checkpoint)

            # Extract and store NN-specific diagnostics
            .deltas  <- setNames(as.numeric(unlist(nn_result$deltas)),          names(nn_result$deltas))
            .attrs   <- setNames(as.numeric(unlist(nn_result$attr_weights)),     names(nn_result$attr_weights))
            .pnorm   <- as.numeric(nn_result$param_norm)
            .perrors <- as.numeric(unlist(nn_result$position_errors))

            session_nn_attrs(.attrs)
            session_nn_param_norms(c(session_nn_param_norms(), .pnorm))

            result  <- list(
                weights  = .attrs,
                deltas   = .deltas,
                td_error = as.numeric(nn_result$td_error),
                errors   = .perrors,
                param_norm = .pnorm
            )
        } else {
            # ── Linear TD update (unchanged) ─────────────────────────────────
            wv     <- session_weights()
            result <- run_td_update(pos, outcome, wv,
                lr             = 0.001,
                draw_lr_scales = list(king_safety = 0.05, center_control = 0.2))
            session_weights(result$weights)
            log_human_game(session_run_id(), gn, outcome,
                           length(move_history()), elapsed, result, build_pgn(move_san_log()), hc,
                           source_run_info()$run_id, source_run_info()$checkpoint)
        }

        post_game_rv(list(
            result     = result,
            positions  = pos,
            outcome    = outcome,
            n_half     = length(pos),
            moves      = move_history(),
            td_error   = result$td_error,
            param_norm = result$param_norm %||% NA_real_
        ))
    }

    # ── Board card outputs ─────────────────────────────────────────────────

    output$board_header <- renderUI({
        sans <- move_san_log()
        n    <- length(sans)
        gr   <- game_result()
        if (!is.null(gr)) {
            label <- switch(gr, "1-0" = "White wins", "0-1" = "Black wins", "Draw")
            return(paste0(label, if (n > 0) paste0(" \u2014 ", n, " moves") else ""))
        }
        if (n == 0) return("Starting position \u2014 your move")
        side <- if (current_board()$turn == chess_mod$WHITE) "White" else "Black"
        paste0(side, " to move \u2014 move ", ceiling(n / 2) + (n %% 2))
    })

    output$move_list <- renderUI({
        sans <- move_san_log()
        if (length(sans) == 0) return(NULL)
        n    <- length(sans)
        rows <- lapply(seq(1, n, by = 2), function(i) {
            mn  <- (i + 1L) %/% 2L
            w   <- sans[i]
            blk <- if (i + 1 <= n) sans[i + 1] else ""
            tags$tr(
                tags$td(style = "color:#999; padding:1px 4px; min-width:20px; text-align:right;",
                        paste0(mn, ".")),
                tags$td(style = "padding:1px 8px 1px 2px; font-family:monospace;", w),
                tags$td(style = "padding:1px 6px 1px 0; font-family:monospace;",   blk)
            )
        })
        div(style = paste("max-height:100px; overflow-y:auto;",
                          "font-size:0.82em; border-top:1px solid #dee2e6;",
                          "padding:4px 8px;"),
            tags$table(style = "border-collapse:collapse;", do.call(tagList, rows))
        )
    })

    output$fen_display <- renderText({ current_board()$fen() })

    # ── Value box outputs ──────────────────────────────────────────────────

    output$vb_status <- renderUI({
        gr <- game_result()
        if (!is.null(gr)) {
            switch(gr, "1-0" = "White wins", "0-1" = "Black wins", "Draw")
        } else if (game_active()) {
            if (current_board()$turn == human_color()) "Your turn" else "Engine thinking\u2026"
        } else if (!is.null(session_run_id())) "Ready" else "No session"
    })

    output$vb_eval <- renderUI({
        req(length(eval_history()) > 0)
        sprintf("%+.3f", tail(eval_history(), 1))
    })

    # ── Diagnostics panel ──────────────────────────────────────────────────

    output$diag_header <- renderUI({
        if (!is.null(post_game_rv()))  "Post-game analysis"
        else if (game_active())        "Live diagnostics"
        else                           "Diagnostics"
    })

    output$diagnostics_panel <- renderUI({
        pg <- post_game_rv()
        if (!is.null(pg))   return(uiOutput("post_game_panel"))
        if (game_active())  return(uiOutput("live_panel"))
        if (!is.null(session_run_id())) {
            return(p(style = "color:#999; padding:16px;",
                     "Start a new game to see live diagnostics."))
        }
        p(style = "color:#999; padding:16px;", "Start a session to begin.")
    })

    # Live panel ----
    output$live_panel <- renderUI({
        tagList(
            card(
                card_header("Position features \u00d7 weights"),
                div(style = "max-height:240px; overflow-y:auto;",
                    tableOutput("feature_table"))
            ),
            card(
                card_header("Evaluation trajectory (White\u2019s perspective)"),
                plotOutput("eval_trajectory", height = "200px")
            )
        )
    })

    output$feature_table <- renderTable({
        req(game_active())
        feats_py <- active_engine_mod()$extract_features(current_board())
        feat_vec <- setNames(as.numeric(unlist(feats_py)), names(feats_py))

        if (is_nn_engine()) {
            # Attribution weights (dV/dx_i) used in place of linear weights
            attrs <- session_nn_attrs()
            if (is.null(attrs)) attrs <- setNames(rep(0, length(feat_vec)), names(feat_vec))
            tibble(
                Feature     = names(feat_vec),
                Value       = feat_vec,
                Attribution = attrs[names(feat_vec)],
                Sensitivity = feat_vec * attrs[names(feat_vec)]
            ) |>
                arrange(desc(abs(Sensitivity))) |>
                mutate(across(where(is.numeric), \(x) round(x, 4)))
        } else {
            wp <- session_weights()
            tibble(
                Feature      = names(feat_vec),
                Value        = feat_vec,
                Weight       = wp[names(feat_vec)],
                Contribution = feat_vec * wp[names(feat_vec)]
            ) |>
                arrange(desc(abs(Contribution))) |>
                mutate(across(where(is.numeric), \(x) round(x, 4)))
        }
    }, striped = TRUE, bordered = FALSE, small = TRUE)

    output$eval_trajectory <- renderPlot({
        eh <- eval_history()
        req(length(eh) > 0)
        tibble(ply = seq_along(eh), eval = eh) |>
            ggplot(aes(x = ply, y = eval)) +
            geom_hline(yintercept = 0, color = "grey70", linewidth = 0.4) +
            geom_line(linewidth = 0.8) +
            geom_point(size = 1.5) +
            scale_y_continuous(limits = c(-1, 1)) +
            labs(x = "Half-move", y = "Eval (White)") +
            theme_minimal(base_size = 12)
    })

    # Post-game panel ----
    output$post_game_panel <- renderUI({
        req(!is.null(post_game_rv()))
        nn <- is_nn_engine()
        tagList(
            uiOutput("pg_summary_boxes"),
            card(
                card_header(if (nn) "Attribution changes from this game" else "Weight changes from this game"),
                plotOutput("pg_delta_chart", height = "240px")
            ),
            card(
                card_header("Top learning moments"),
                uiOutput("pg_moments")
            ),
            if (nn) card(
                card_header("Parameter update magnitude — this session"),
                plotOutput("pg_norm_history", height = "160px")
            ),
            card(
                card_header("TD error vs self-play distribution"),
                plotOutput("pg_histogram", height = "200px")
            )
        )
    })

    output$pg_norm_history <- renderPlot({
        req(is_nn_engine())
        norms    <- session_nn_param_norms()
        baseline <- session_nn_sp_baseline()
        req(length(norms) > 0)
        df <- tibble(game = seq_along(norms), norm = norms)
        p  <- ggplot(df, aes(x = game, y = norm)) +
            geom_line(linewidth = 0.8) +
            geom_point(size = 2.5) +
            labs(x = "Human game (this session)", y = "Frobenius norm") +
            theme_minimal(base_size = 12)
        if (!is.na(baseline)) {
            p <- p +
                geom_hline(yintercept = baseline, linetype = "dashed",
                           color = "steelblue", linewidth = 0.7) +
                annotate("text", x = Inf, y = baseline, hjust = 1.1, vjust = -0.4,
                         label = sprintf("Self-play avg (%.4f)", baseline),
                         color = "steelblue", size = 3.2)
        }
        p
    })

    output$pg_summary_boxes <- renderUI({
        pg          <- post_game_rv()
        req(!is.null(pg))
        sp_errors   <- src_td_errors()
        sp_mean     <- if (length(sp_errors) > 0) mean(sp_errors) else NA_real_
        total_delta <- sum(abs(pg$result$deltas))

        if (is_nn_engine()) {
            # NN boxes: TD error | Param Δ (Frobenius) | vs self-play norm
            pnorm    <- pg$param_norm
            baseline <- session_nn_sp_baseline()
            vs_label <- if (!is.na(pnorm) && !is.na(baseline) && baseline > 0) {
                sprintf("%.1f\u00d7 self-play", pnorm / baseline)
            } else "\u2014"
            pnorm_label <- if (!is.na(pnorm)) sprintf("%.4f", pnorm) else "\u2014"
            layout_column_wrap(
                width = 1/3, fill = FALSE,
                value_box("TD error",     sprintf("%.4f", pg$td_error),
                          theme = "primary",   showcase = bs_icon("lightning")),
                value_box("Param \u0394 (Frob.)", pnorm_label,
                          theme = "info",      showcase = bs_icon("arrows-move")),
                value_box("vs self-play", vs_label,
                          theme = "secondary", showcase = bs_icon("bar-chart"))
            )
        } else {
            rel_label <- if (!is.na(sp_mean)) {
                sprintf("%+.0f%% vs avg", (pg$td_error / sp_mean - 1) * 100)
            } else "\u2014"
            layout_column_wrap(
                width = 1/3, fill = FALSE,
                value_box("TD error",     sprintf("%.4f", pg$td_error),
                          theme = "primary",   showcase = bs_icon("lightning")),
                value_box("vs self-play", rel_label,
                          theme = "secondary", showcase = bs_icon("bar-chart")),
                value_box("Total \u0394w", sprintf("%.4f", total_delta),
                          theme = "light",     showcase = bs_icon("arrow-left-right"))
            )
        }
    })

    output$pg_delta_chart <- renderPlot({
        pg <- post_game_rv()
        req(!is.null(pg))
        tibble(feature = names(pg$result$deltas),
               delta   = as.numeric(pg$result$deltas)) |>
            mutate(feature = fct_reorder(feature, abs(delta)),
                   sign    = ifelse(delta >= 0, "pos", "neg")) |>
            ggplot(aes(x = delta, y = feature, fill = sign)) +
            geom_col(width = 0.6) +
            geom_vline(xintercept = 0, linewidth = 0.4, color = "grey40") +
            scale_fill_manual(values = c(pos = "#d4edda", neg = "#f8d7da"), guide = "none") +
            labs(x = "\u0394w (weight change)", y = NULL) +
            theme_minimal(base_size = 12) +
            theme(panel.grid.major.y = element_blank())
    })

    output$pg_moments <- renderUI({
        pg <- post_game_rv()
        req(!is.null(pg))
        errs     <- pg$result$errors
        top3_idx <- order(abs(errs), decreasing = TRUE)[seq_len(min(3, length(errs)))]
        moves_uci <- pg$moves
        wv        <- session_weights()

        cards <- lapply(top3_idx, function(t) {
            b_t <- chess_mod$Board()
            for (uci in head(moves_uci, t - 1L)) b_t$push(chess_mod$Move$from_uci(uci))
            svg <- board_svg_html(b_t)

            pred <- sum(pg$positions[[t]]$features * wv[pg$positions[[t]]$names])
            tgt  <- if (t < length(errs)) {
                -sum(pg$positions[[t + 1L]]$features * wv[pg$positions[[t + 1L]]$names])
            } else {
                outcome_for_color(pg$outcome, pg$positions[[t]]$is_white)
            }

            div(style = "display:inline-block; text-align:center; margin:8px; vertical-align:top;",
                svg,
                tags$small(
                    tags$b(sprintf("Ply %d \u2014 |error| = %.3f", t, abs(errs[t]))),
                    tags$br(),
                    sprintf("Predicted: %+.3f \u2192 Target: %+.3f", pred, tgt)
                )
            )
        })
        div(style = "white-space:nowrap; overflow-x:auto;", do.call(tagList, cards))
    })

    output$pg_histogram <- renderPlot({
        pg <- post_game_rv()
        req(!is.null(pg))
        sp <- src_td_errors()

        if (length(sp) == 0) {
            return(ggplot() + theme_void() +
                annotate("text", x = 0, y = 0, label = "No self-play data available",
                         color = "grey60"))
        }

        ggplot(tibble(td_error = sp), aes(x = td_error)) +
            geom_histogram(bins = 30, fill = "#adc8f7", color = "white") +
            geom_vline(xintercept = pg$td_error, color = "#0d6efd", linewidth = 1.2) +
            annotate("text", x = pg$td_error, y = Inf, vjust = 1.5, hjust = -0.1,
                     label = sprintf("This game\n%.4f", pg$td_error),
                     color = "#0d6efd", size = 3.5) +
            labs(x = "TD error", y = "Self-play games",
                 title = sprintf("Run %d self-play distribution",
                                 source_run_info()$run_id %||% 0)) +
            theme_minimal(base_size = 12)
    })
}

shinyApp(ui, server)
