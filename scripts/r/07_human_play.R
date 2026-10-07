# 07_human_play.R
# Human (or external agent) plays our trained engine.
# Each game contributes a real TD(0) weight update to a new run seeded
# from a chosen source checkpoint.
#
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

chess_mod  <- import("chess")
chess_svg  <- import("chess.svg")
engine_mod <- import_from_path("engine",
    path = if (file.exists("../python/engine.py")) "../python" else "scripts/python")

# ---- Paths ----

DB_PATH <- if (file.exists("../../data/chess_learning.db")) {
    "../../data/chess_learning.db"
} else {
    "data/chess_learning.db"
}

# ---- Helpers ----

`%||%` <- function(a, b) if (!is.null(a)) a else b

get_run_ids <- function() {
    con <- dbConnect(SQLite(), DB_PATH)
    ids <- dbGetQuery(con, "SELECT DISTINCT run_id FROM weights ORDER BY run_id DESC")$run_id
    dbDisconnect(con)
    ids
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

board_svg_html <- function(board, last_move = NULL, flipped = FALSE, size = 380L) {
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

# Re-implementation of td_update() matching engine notebook exactly
run_td_update <- function(positions, outcome, weights_vec,
                           lr = 0.001,
                           draw_lr_scales = list(king_safety = 0.05)) {
    n       <- length(positions)
    is_draw <- outcome == "1/2-1/2"

    values <- sapply(positions, function(p) {
        sum(p$features * weights_vec[p$names])
    })

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

    list(
        weights  = new_weights,
        deltas   = deltas,
        td_error = mean(abs(errors)),
        errors   = errors
    )
}

create_new_run <- function(source_run_id, source_checkpoint, weights_vec) {
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO runs (started_at, depth, learning_rate, move_limit, epsilon,
                           weight_init, notes)
         SELECT datetime('now'), depth, learning_rate, move_limit, epsilon,
                weight_init,
                printf('Human play — seeded from run_id=%d, game=%d', run_id, ?)
         FROM runs WHERE run_id = ? ORDER BY run_id LIMIT 1",
        list(source_checkpoint, source_run_id))
    new_run_id <- as.integer(dbGetQuery(con, "SELECT last_insert_rowid()")[[1]])

    # Log source weights as checkpoint 0
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
                             elapsed, td_result, pgn_str) {
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con,
        "INSERT INTO games (run_id,game_number,outcome,n_half_moves,elapsed_sec,terminated_by)
         VALUES (?,?,?,?,?,'natural')",
        list(run_id, game_number, outcome, n_half, elapsed))
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

build_pgn <- function(san_moves) {
    pairs <- split(san_moves, ceiling(seq_along(san_moves) / 2))
    paste(mapply(function(p, i) paste0(i, ". ", paste(p, collapse=" ")),
                 pairs, seq_along(pairs)), collapse=" ")
}

# ---- Static data ----

available_runs <- get_run_ids()

# ---- UI ----

ui <- page_sidebar(
    title = "Human vs Engine",
    theme = bs_theme(version = 5, bootswatch = "cosmo"),

    sidebar = sidebar(
        width = 280,

        card(
            card_header("Session"),
            selectInput("src_run_id", "Source run",
                choices = available_runs, selected = available_runs[1], width = "100%"),
            selectInput("src_checkpoint", "Checkpoint",
                choices = character(0), width = "100%"),
            actionButton("btn_start_session", "Start Session",
                class = "btn-primary w-100"),
            uiOutput("session_info")
        ),

        card(
            card_header("Game"),
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
        value_box("Status",       uiOutput("vb_status"),
                  theme = "primary",   showcase = bs_icon("circle-fill")),
        value_box("Engine eval",  uiOutput("vb_eval"),
                  theme = "secondary", showcase = bs_icon("speedometer")),
        value_box("Session games", uiOutput("vb_game_count"),
                  theme = "light",    showcase = bs_icon("list-ol"))
    ),

    layout_columns(
        col_widths = c(5, 7),

        # ── Board card ──────────────────────────────────────────────────────
        card(
            full_screen = TRUE,
            card_header(uiOutput("board_header")),
            uiOutput("board_svg"),
            uiOutput("move_controls"),
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
    session_run_id  <- reactiveVal(NULL)
    session_weights <- reactiveVal(NULL)   # named numeric vector
    game_number_rv  <- reactiveVal(0L)
    src_td_errors   <- reactiveVal(numeric(0))  # self-play td_errors for comparison
    source_run_info <- reactiveVal(NULL)

    # Checkpoint selector
    observeEvent(input$src_run_id, {
        cpts <- get_checkpoints(as.integer(input$src_run_id))
        updateSelectInput(session, "src_checkpoint", choices = cpts, selected = cpts[1])
    }, ignoreNULL = FALSE)

    observeEvent(input$btn_start_session, {
        run_id  <- as.integer(input$src_run_id)
        gn      <- as.integer(input$src_checkpoint)
        wv      <- load_weights_vec(run_id, gn)
        new_id  <- create_new_run(run_id, gn, wv)

        session_run_id(new_id)
        session_weights(wv)
        game_number_rv(0L)
        source_run_info(list(run_id = run_id, checkpoint = gn))

        # Load self-play td_errors for comparison histogram
        con     <- dbConnect(SQLite(), DB_PATH)
        errors  <- dbGetQuery(con, sprintf(
            "SELECT td_error FROM game_records WHERE run_id=%d AND td_error IS NOT NULL",
            run_id))$td_error
        dbDisconnect(con)
        src_td_errors(errors)

        showNotification(paste0("Session started — run_id = ", new_id), type = "message")
    })

    output$session_info <- renderUI({
        rid <- session_run_id()
        if (is.null(rid)) return(NULL)
        tags$small(style = "color:#666;",
            paste0("Session run: ", rid,
                   " | Games: ", game_number_rv()))
    })

    # ── Game state ─────────────────────────────────────────────────────────
    move_history  <- reactiveVal(character(0))
    move_san_log  <- reactiveVal(character(0))
    last_move_rv  <- reactiveVal(NULL)
    human_color   <- reactiveVal(NULL)
    game_active   <- reactiveVal(FALSE)
    positions_rv  <- reactiveVal(list())    # list of list(features, names, is_white)
    eval_history  <- reactiveVal(numeric(0))
    game_result   <- reactiveVal(NULL)
    post_game_rv  <- reactiveVal(NULL)
    game_start_t  <- reactiveVal(NULL)

    current_board <- reactive({
        b <- chess_mod$Board()
        for (uci in move_history()) b$push(chess_mod$Move$from_uci(uci))
        b
    })

    output$vb_game_count <- renderUI({ game_number_rv() })

    # New game
    observeEvent(input$btn_new_game, {
        req(!is.null(session_run_id()))
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

        # Engine moves first if human is Black
        if (col == chess_mod$BLACK) play_engine_move()
    })

    # Resign
    observeEvent(input$btn_resign, {
        req(game_active())
        outcome <- if (human_color() == chess_mod$WHITE) "0-1" else "1-0"
        end_game(outcome, resigned = TRUE)
    })

    # ── Position & eval recording ──────────────────────────────────────────

    record_position <- function(board) {
        feats_py  <- engine_mod$extract_features(board)
        feat_vec  <- setNames(as.numeric(unlist(feats_py)), names(feats_py))
        positions_rv(c(positions_rv(), list(list(
            features = feat_vec,
            names    = names(feat_vec),
            is_white = board$turn == chess_mod$WHITE
        ))))
    }

    update_eval_history <- function(board_after) {
        wp <- as.list(session_weights())
        s  <- as.numeric(engine_mod$evaluate(board_after, wp))
        # From White's perspective: negate if White is to move (White just played, Black to move)
        white_eval <- if (board_after$turn == chess_mod$WHITE) s else -s
        eval_history(c(eval_history(), white_eval))
    }

    # ── Human move ────────────────────────────────────────────────────────

    observeEvent(input$btn_play, {
        req(game_active(), nzchar(input$move_select %||% ""))
        uci <- input$move_select
        b   <- current_board()
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

    # Populate human move selector
    observe({
        b <- current_board()
        req(game_active(), b$turn == human_color())
        legal <- reticulate::iterate(b$legal_moves)
        if (length(legal) == 0) return()
        choices <- setNames(
            sapply(legal, \(m) m$uci()),
            sapply(legal, \(m) b$san(m))
        )
        choices <- choices[order(names(choices))]
        updateSelectInput(session, "move_select", choices = choices, selected = choices[1])
    })

    # Undo (removes last human + engine move pair)
    observeEvent(input$btn_undo, {
        req(game_active())
        h <- move_history()
        n_pop <- min(2L, length(h))
        if (n_pop == 0) return()
        move_history(head(h, -n_pop))
        move_san_log(head(move_san_log(), -n_pop))
        last_move_rv(NULL)
        positions_rv(head(positions_rv(), -n_pop))
        eval_history(head(eval_history(), -n_pop))
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

            if (current_board()$is_game_over()) end_game(current_board()$result())
        }, error = function(e) {
            showNotification(paste("Engine error:", conditionMessage(e)), type = "error")
        })
    }

    # ── End game ──────────────────────────────────────────────────────────

    end_game <- function(outcome, resigned = FALSE) {
        game_active(FALSE)
        game_result(outcome)

        elapsed <- proc.time()[["elapsed"]] - (game_start_t() %||% 0)
        wv      <- session_weights()
        pos     <- positions_rv()

        if (length(pos) == 0) return()

        # Run TD update
        result <- run_td_update(pos, outcome, wv,
            lr             = 0.001,
            draw_lr_scales = list(king_safety = 0.05))

        # Update session weights
        session_weights(result$weights)

        # Increment game counter and log
        gn <- game_number_rv() + 1L
        game_number_rv(gn)
        pgn <- build_pgn(move_san_log())

        log_human_game(session_run_id(), gn, outcome,
                       length(move_history()), elapsed, result, pgn)

        post_game_rv(list(
            result     = result,
            positions  = pos,
            outcome    = outcome,
            n_half     = length(pos),
            moves      = move_history(),
            td_error   = result$td_error
        ))
    }

    # ── Board outputs ──────────────────────────────────────────────────────

    output$board_header <- renderUI({
        sans <- move_san_log()
        n    <- length(sans)
        gr   <- game_result()
        if (!is.null(gr)) {
            label <- switch(gr, "1-0"="White wins", "0-1"="Black wins", "Draw")
            return(paste0(label, if (n > 0) paste0(" — after ", n, " moves") else ""))
        }
        if (n == 0) return("Starting position — your move")
        side <- if (current_board()$turn == chess_mod$WHITE) "White" else "Black"
        paste0(side, " to move — after ", ceiling(n/2), " moves")
    })

    output$board_svg <- renderUI({
        col <- human_color()
        board_svg_html(current_board(), last_move = last_move_rv(),
                       flipped = !is.null(col) && col == chess_mod$BLACK)
    })

    output$fen_display <- renderText({ current_board()$fen() })

    output$move_controls <- renderUI({
        req(game_active())
        b <- current_board()
        if (b$turn != human_color()) {
            return(p(style = "text-align:center; color:#999; padding:8px;",
                     "Engine is thinking\u2026"))
        }
        div(style = "padding:8px;",
            layout_columns(
                col_widths = c(7, 2, 3),
                selectInput("move_select", NULL, choices = character(0), width = "100%"),
                actionButton("btn_play", "\u25b6",  class = "btn-primary  btn-sm w-100"),
                actionButton("btn_undo", "\u21a9",  class = "btn-outline-secondary btn-sm w-100")
            )
        )
    })

    # ── Value box outputs ──────────────────────────────────────────────────

    output$vb_status <- renderUI({
        gr <- game_result()
        if (!is.null(gr)) {
            switch(gr, "1-0"="White wins", "0-1"="Black wins", "Draw")
        } else if (game_active()) {
            if (current_board()$turn == human_color()) "Your turn" else "Engine thinking\u2026"
        } else if (!is.null(session_run_id())) "Ready" else "No session"
    })

    output$vb_eval <- renderUI({
        req(length(eval_history()) > 0)
        val <- tail(eval_history(), 1)
        sprintf("%+.3f", val)
    })

    # ── Diagnostics panel ──────────────────────────────────────────────────

    output$diag_header <- renderUI({
        if (!is.null(post_game_rv()))   "Post-game analysis"
        else if (game_active())         "Live diagnostics"
        else                            "Diagnostics"
    })

    output$diagnostics_panel <- renderUI({
        pg <- post_game_rv()
        if (!is.null(pg)) return(uiOutput("post_game_panel"))
        if (game_active())  return(uiOutput("live_panel"))
        if (!is.null(session_run_id())) {
            return(p(style="color:#999; padding:16px;",
                     "Start a new game to see live diagnostics."))
        }
        p(style="color:#999; padding:16px;",
          "Start a session to begin.")
    })

    # Live panel
    output$live_panel <- renderUI({
        tagList(
            card(
                card_header("Position features \u00d7 weights"),
                div(style="max-height:240px; overflow-y:auto;",
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
        wp  <- session_weights()
        b   <- current_board()
        feats_py <- engine_mod$extract_features(b)
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
            labs(x = "Half-move", y = "Eval (White)", title = NULL) +
            theme_minimal(base_size = 12)
    })

    # Post-game panel
    output$post_game_panel <- renderUI({
        req(!is.null(post_game_rv()))
        tagList(
            uiOutput("pg_summary_boxes"),
            card(
                card_header("Weight changes from this game"),
                plotOutput("pg_delta_chart", height = "240px")
            ),
            card(
                card_header("Top learning moments"),
                uiOutput("pg_moments")
            ),
            card(
                card_header("This game vs self-play"),
                plotOutput("pg_histogram", height = "200px")
            )
        )
    })

    output$pg_summary_boxes <- renderUI({
        pg <- post_game_rv()
        req(!is.null(pg))
        sp_errors <- src_td_errors()
        pct <- if (length(sp_errors) > 0) {
            round(mean(sp_errors < pg$td_error) * 100)
        } else NA_real_

        total_delta <- sum(abs(pg$result$deltas))
        sp_mean     <- if (length(sp_errors) > 0) mean(sp_errors) else NA_real_
        rel_label   <- if (!is.na(sp_mean)) {
            sprintf("%+.0f%% vs avg", (pg$td_error / sp_mean - 1) * 100)
        } else "\u2014"

        layout_column_wrap(
            width = 1/3, fill = FALSE,
            value_box("TD error",       sprintf("%.4f", pg$td_error),
                      theme = "primary",   showcase = bs_icon("lightning")),
            value_box("vs self-play",  rel_label,
                      theme = "secondary", showcase = bs_icon("bar-chart")),
            value_box("Total \u0394w",  sprintf("%.4f", total_delta),
                      theme = "light",     showcase = bs_icon("arrow-left-right"))
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
            labs(x = "\u0394w (weight change)", y = NULL) +
            theme_minimal(base_size = 12) +
            theme(panel.grid.major.y = element_blank())
    })

    output$pg_moments <- renderUI({
        pg <- post_game_rv()
        req(!is.null(pg))
        errs      <- pg$result$errors
        top3_idx  <- order(abs(errs), decreasing = TRUE)[seq_len(min(3, length(errs)))]
        moves_uci <- pg$moves

        cards <- lapply(top3_idx, function(t) {
            # Reconstruct board at ply t
            b_t <- chess_mod$Board()
            for (uci in head(moves_uci, t - 1L)) b_t$push(chess_mod$Move$from_uci(uci))
            svg <- board_svg_html(b_t, size = 160L)

            pred   <- sum(pg$positions[[t]]$features * session_weights()[pg$positions[[t]]$names])
            tgt    <- if (t < length(errs)) -sum(pg$positions[[t + 1L]]$features *
                                                    session_weights()[pg$positions[[t + 1L]]$names])
                      else outcome_for_color(pg$outcome, pg$positions[[t]]$is_white)

            div(style = "display:inline-block; text-align:center; margin:8px; vertical-align:top;",
                svg,
                tags$small(
                    tags$b(sprintf("Ply %d — |error| = %.3f", t, abs(errs[t]))), tags$br(),
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
                annotate("text", x=0, y=0, label="No self-play data available", color="grey60"))
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
