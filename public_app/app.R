# public_app/app.R
# Crowdsourced human vs engine — public shinyapps.io deployment.
# All human games contribute TD(0) weight updates to a shared Supabase run.
# Weights evolve collectively across all players.

library(shiny)
library(bslib)
library(bsicons)
library(DBI)
library(RPostgres)
library(tidyverse)
library(reticulate)

# ---- Python environment ----
# Install python-chess into a virtualenv on first cold start

if (!virtualenv_exists("chess-public")) {
    virtualenv_create("chess-public")
    virtualenv_install("chess-public", "chess", ignore_installed = FALSE)
}
use_virtualenv("chess-public", required = TRUE)

chess_mod  <- import("chess")
chess_svg  <- import("chess.svg")
engine_mod <- import_from_path("engine", path = getwd())

# ---- Database ----

PG_CREDS <- list(
    host     = "db.betniftrhjpeacfzjoym.supabase.co",
    port     = 5432L,
    dbname   = "postgres",
    user     = "postgres",
    password = Sys.getenv("SUPABASE_PASSWORD"),
    sslmode  = "require"
)

PUBLIC_RUN_ID <- 1L

pg_connect <- function() do.call(dbConnect, c(list(Postgres()), PG_CREDS))

load_latest_weights <- function() {
    pg  <- pg_connect()
    gn  <- dbGetQuery(pg,
        "SELECT MAX(game_number) FROM weights WHERE run_id = $1",
        list(PUBLIC_RUN_ID))[[1]]
    rows <- dbGetQuery(pg,
        "SELECT feature_name, weight_value FROM weights
         WHERE run_id = $1 AND game_number = $2",
        list(PUBLIC_RUN_ID, gn))
    dbDisconnect(pg)
    setNames(rows$weight_value, rows$feature_name)
}

log_public_game <- function(game_number, outcome, n_half, elapsed, td_result, pgn_str) {
    pg <- pg_connect()
    dbExecute(pg,
        "INSERT INTO games (run_id, game_number, outcome, n_halfmoves, duration_s, terminated_by)
         VALUES ($1, $2, $3, $4, $5, 'natural')",
        list(PUBLIC_RUN_ID, game_number, outcome, n_half, elapsed))
    dbExecute(pg,
        "INSERT INTO game_records (run_id, game_number, pgn, td_error)
         VALUES ($1, $2, $3, $4)",
        list(PUBLIC_RUN_ID, game_number, pgn_str, td_result$td_error))
    for (fname in names(td_result$deltas)) {
        dbExecute(pg,
            "INSERT INTO weight_deltas (run_id, game_number, feature_name, delta)
             VALUES ($1, $2, $3, $4)",
            list(PUBLIC_RUN_ID, game_number, fname, td_result$deltas[[fname]]))
    }
    for (fname in names(td_result$weights)) {
        dbExecute(pg,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES ($1, $2, $3, $4)",
            list(PUBLIC_RUN_ID, game_number, fname, td_result$weights[[fname]]))
    }
    dbDisconnect(pg)
}

get_game_count <- function() {
    pg  <- pg_connect()
    n   <- dbGetQuery(pg,
        "SELECT COUNT(*) FROM games WHERE run_id = $1",
        list(PUBLIC_RUN_ID))[[1]]
    dbDisconnect(pg)
    n
}

# ---- Helpers ----

`%||%` <- function(a, b) if (!is.null(a)) a else b

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
    values  <- sapply(positions, function(p) sum(p$features * weights_vec[p$names]))
    deltas  <- setNames(numeric(length(weights_vec)), names(weights_vec))
    errors  <- numeric(n)

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
    for (fname in names(deltas))
        new_weights[[fname]] <- max(-50, min(50, weights_vec[[fname]] + deltas[[fname]]))

    list(weights = new_weights, deltas = deltas,
         td_error = mean(abs(errors)), errors = errors)
}

build_pgn <- function(san_moves) {
    pairs <- split(san_moves, ceiling(seq_along(san_moves) / 2))
    paste(mapply(function(p, i) paste0(i, ". ", paste(p, collapse = " ")),
                 pairs, seq_along(pairs)), collapse = " ")
}

# ---- JavaScript: chessboard.js ----

board_js <- HTML("
$(document).ready(function () {
  var chess = new Chess();
  var board  = null;
  window.humanTurn  = false;
  window.humanColor = 'w';
  window.prevHL     = [];

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

    window.prevHL.forEach(function (sq) {
      $('#chessboard .square-' + sq).css('background', '');
    });
    window.prevHL = [];

    if (msg.lm_from) {
      $('#chessboard .square-' + msg.lm_from).css('background', 'rgba(255,214,0,0.4)');
      window.prevHL.push(msg.lm_from);
    }
    if (msg.lm_to) {
      $('#chessboard .square-' + msg.lm_to).css('background', 'rgba(255,214,0,0.4)');
      window.prevHL.push(msg.lm_to);
    }
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
    title = "Teach the Chess Engine",
    theme = bs_theme(version = 5, bootswatch = "cosmo"),

    tags$head(
        tags$link(rel  = "stylesheet",
                  href = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"),
        tags$script(src = "https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"),
        tags$script(src = "https://unpkg.com/chess.js@0.10.3/chess.js"),
        tags$script(board_js),
        tags$style(HTML("
          .about-text { font-size: 0.85em; color: #555; line-height: 1.5; }
        "))
    ),

    sidebar = sidebar(
        width = 260,

        card(
            card_header("New Game"),
            radioButtons("human_color", "Play as",
                choices = c("White", "Black"), inline = TRUE),
            sliderInput("depth", "Engine depth",
                min = 1, max = 3, value = 2, step = 1, width = "100%"),
            actionButton("btn_new_game", "Start Game",
                class = "btn-success w-100 mb-2"),
            actionButton("btn_resign", "Resign",
                class = "btn-outline-danger btn-sm w-100")
        ),

        card(
            card_header("About"),
            p(class = "about-text",
              "Every game you play teaches the engine. ",
              "It uses ", tags$b("temporal difference learning"), " to update its ",
              "positional weights after each game. ",
              "All players share the same evolving engine.")
        )
    ),

    layout_column_wrap(
        width = 1/3, fill = FALSE,
        value_box("Status",       uiOutput("vb_status"),
                  theme = "primary",   showcase = bs_icon("circle-fill")),
        value_box("Engine eval",  uiOutput("vb_eval"),
                  theme = "secondary", showcase = bs_icon("speedometer")),
        value_box("Games played", uiOutput("vb_game_count"),
                  theme = "light",     showcase = bs_icon("people"))
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

    # Load latest shared weights at session start
    session_weights <- reactiveVal(load_latest_weights())
    game_count_rv   <- reactiveVal(get_game_count())

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

    output$vb_game_count <- renderUI({ game_count_rv() })

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
        # Reload latest shared weights so this game builds on all prior learning
        session_weights(load_latest_weights())

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
        feats_py <- engine_mod$extract_features(board)
        feat_vec <- setNames(as.numeric(unlist(feats_py)), names(feats_py))
        positions_rv(c(positions_rv(), list(list(
            features = feat_vec,
            names    = names(feat_vec),
            is_white = board$turn == chess_mod$WHITE
        ))))
    }

    update_eval_history <- function(board_after) {
        wp <- as.list(session_weights())
        s  <- as.numeric(engine_mod$evaluate(board_after, wp))
        white_eval <- if (board_after$turn == chess_mod$WHITE) s else -s
        eval_history(c(eval_history(), white_eval))
    }

    # ── Human move ────────────────────────────────────────────────────────

    observeEvent(input$human_move_drop, {
        req(game_active())
        uci <- input$human_move_drop$uci
        b   <- current_board()
        req(b$turn == human_color())

        legal_ucis <- sapply(reticulate::iterate(b$legal_moves), \(m) m$uci())
        req(uci %in% legal_ucis)

        mv  <- chess_mod$Move$from_uci(uci)
        san <- b$san(mv)
        record_position(b)
        move_history(c(move_history(), uci))
        move_san_log(c(move_san_log(), san))
        last_move_rv(mv)
        update_eval_history(current_board())

        if (current_board()$is_game_over()) {
            end_game(current_board()$result())
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
        wp <- as.list(session_weights())
        record_position(b)

        tryCatch({
            mv  <- engine_mod$get_best_move(b, wp, as.integer(input$depth), FALSE)[[1]]
            req(!is.null(mv))
            san <- b$san(mv)
            move_history(c(move_history(), mv$uci()))
            move_san_log(c(move_san_log(), san))
            last_move_rv(mv)
            update_eval_history(current_board())

            if (current_board()$is_game_over()) {
                end_game(current_board()$result())
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
        wv  <- session_weights()
        pos <- positions_rv()
        if (length(pos) == 0) return()

        result <- run_td_update(pos, outcome, wv,
            lr             = 0.001,
            draw_lr_scales = list(king_safety = 0.05, center_control = 0.2))

        session_weights(result$weights)

        pg <- pg_connect()
        gn <- dbGetQuery(pg,
            "SELECT COALESCE(MAX(game_number), 0) + 1 FROM games WHERE run_id = $1",
            list(PUBLIC_RUN_ID))[[1]]
        dbDisconnect(pg)

        log_public_game(gn, outcome, length(move_history()), elapsed,
                        result, build_pgn(move_san_log()))

        game_count_rv(game_count_rv() + 1L)

        post_game_rv(list(
            result   = result,
            outcome  = outcome,
            n_half   = length(pos),
            moves    = move_history(),
            td_error = result$td_error
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
        if (n == 0) return("Your move \u2014 drag a piece to play")
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

    # ── Value boxes ───────────────────────────────────────────────────────

    output$vb_status <- renderUI({
        gr <- game_result()
        if (!is.null(gr)) {
            switch(gr, "1-0" = "White wins", "0-1" = "Black wins", "Draw")
        } else if (game_active()) {
            if (current_board()$turn == human_color()) "Your turn" else "Engine thinking\u2026"
        } else "Start a game"
    })

    output$vb_eval <- renderUI({
        req(length(eval_history()) > 0)
        sprintf("%+.3f", tail(eval_history(), 1))
    })

    # ── Diagnostics panel ──────────────────────────────────────────────────

    output$diag_header <- renderUI({
        if (!is.null(post_game_rv()))  "What the engine learned"
        else if (game_active())        "Live evaluation"
        else                           "Engine weights"
    })

    output$diagnostics_panel <- renderUI({
        pg <- post_game_rv()
        if (!is.null(pg))   return(uiOutput("post_game_panel"))
        if (game_active())  return(uiOutput("live_panel"))
        uiOutput("weights_panel")
    })

    # Weights panel (shown before first game)
    output$weights_panel <- renderUI({
        card(
            card_header("Current positional weights"),
            tableOutput("weights_table")
        )
    })

    output$weights_table <- renderTable({
        wv <- session_weights()
        tibble(Feature = names(wv), Weight = unname(wv)) |>
            arrange(desc(abs(Weight))) |>
            mutate(Weight = round(Weight, 4))
    }, striped = TRUE, bordered = FALSE, small = TRUE)

    # Live panel
    output$live_panel <- renderUI({
        tagList(
            card(
                card_header("Position features \u00d7 weights"),
                div(style = "max-height:240px; overflow-y:auto;",
                    tableOutput("feature_table"))
            ),
            card(
                card_header("Evaluation (White\u2019s perspective)"),
                plotOutput("eval_trajectory", height = "180px")
            )
        )
    })

    output$feature_table <- renderTable({
        req(game_active())
        wp       <- session_weights()
        feats_py <- engine_mod$extract_features(current_board())
        feat_vec <- setNames(as.numeric(unlist(feats_py)), names(feats_py))
        tibble(
            Feature      = names(feat_vec),
            Value        = feat_vec,
            Weight       = wp[names(feat_vec)],
            Contribution = feat_vec * wp[names(feat_vec)]
        ) |>
            arrange(desc(abs(Contribution))) |>
            mutate(across(where(is.numeric), \(x) round(x, 4)))
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

    # Post-game panel
    output$post_game_panel <- renderUI({
        req(!is.null(post_game_rv()))
        tagList(
            uiOutput("pg_summary_boxes"),
            card(
                card_header("What changed"),
                plotOutput("pg_delta_chart", height = "220px")
            )
        )
    })

    output$pg_summary_boxes <- renderUI({
        pg <- post_game_rv()
        req(!is.null(pg))
        outcome_label <- switch(pg$outcome,
            "1-0" = "White won", "0-1" = "Black won", "Draw")
        total_delta <- sum(abs(pg$result$deltas))

        layout_column_wrap(
            width = 1/2, fill = FALSE,
            value_box("Result",    outcome_label,
                      theme = "primary",   showcase = bs_icon("trophy")),
            value_box("Total \u0394w", sprintf("%.4f", total_delta),
                      theme = "secondary", showcase = bs_icon("arrow-left-right"))
        )
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
            labs(x = "\u0394w (weight change this game)", y = NULL) +
            theme_minimal(base_size = 12) +
            theme(panel.grid.major.y = element_blank())
    })
}

shinyApp(ui, server)
