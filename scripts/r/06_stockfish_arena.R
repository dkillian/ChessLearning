# 06_stockfish_arena.R
# Two-tab Shiny app for playing and benchmarking our engine against Stockfish.
#
# Tab 1 — Live Game:  watch our engine play Stockfish move-by-move
# Tab 2 — ELO Benchmark: batch matches at multiple ELO levels, win-rate chart,
#          estimated ELO, and ELO trajectory across training checkpoints
#
# Requires Stockfish installed via: winget install Stockfish.Stockfish
# Run: shiny::runApp("scripts/r/06_stockfish_arena.R")

library(shiny)
library(bslib)
library(bsicons)
library(DBI)
library(RSQLite)
library(tidyverse)
library(reticulate)

# ---- Python modules ----

chess_mod    <- import("chess")
chess_svg    <- import("chess.svg")
chess_engine <- import("chess.engine")
engine_mod   <- import_from_path("engine",
    path = if (file.exists("../python/engine.py")) "../python" else "scripts/python")

# ---- Paths ----

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

# ---- DB setup ----

local({
    con <- dbConnect(SQLite(), DB_PATH)
    dbExecute(con, "
        CREATE TABLE IF NOT EXISTS stockfish_evals (
            eval_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER,
            game_number INTEGER,
            sf_elo      INTEGER,
            n_games     INTEGER,
            wins        INTEGER,
            losses      INTEGER,
            draws       INTEGER,
            win_rate    REAL,
            source      TEXT
        )")
    dbDisconnect(con)
})

# ---- Helpers ----

get_run_ids <- function() {
    con  <- dbConnect(SQLite(), DB_PATH)
    ids  <- dbGetQuery(con, "SELECT DISTINCT run_id FROM weights ORDER BY run_id DESC")$run_id
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

load_weights_fn <- function(run_id, game_number) {
    con  <- dbConnect(SQLite(), DB_PATH)
    rows <- dbGetQuery(con, sprintf(
        "SELECT feature_name, weight_value FROM weights WHERE run_id=%d AND game_number=%d",
        as.integer(run_id), as.integer(game_number)))
    dbDisconnect(con)
    as.list(setNames(rows$weight_value, rows$feature_name))
}

board_svg_html <- function(board, last_move = NULL, flipped = FALSE, size = 390L) {
    args <- list(board = board, size = size, flipped = flipped)
    if (!is.null(last_move)) args$lastmove <- last_move
    if (board$is_check()) {
        king_sq <- board$king(board$turn)
        if (!is.null(king_sq)) args$check <- king_sq
    }
    HTML(paste0('<div style="text-align:center;line-height:0;">',
                do.call(chess_svg$board, args), "</div>"))
}

estimate_elo <- function(df) {
    df <- df |> arrange(sf_elo)
    if (nrow(df) < 2) return("Need \u22652 ELO levels")
    if (all(df$win_rate >= 0.5)) return(paste0(">", max(df$sf_elo)))
    if (all(df$win_rate <  0.5)) return(paste0("<", min(df$sf_elo)))
    above <- df |> filter(win_rate >= 0.5) |> slice_min(sf_elo, n = 1, with_ties = FALSE)
    below <- df |> filter(win_rate <  0.5) |> slice_min(sf_elo, n = 1, with_ties = FALSE)
    x1 <- above$sf_elo; y1 <- above$win_rate
    x2 <- below$sf_elo; y2 <- below$win_rate
    if (abs(y1 - y2) < 1e-6) return(as.character(x1))
    paste0("~", round(x1 + (0.5 - y1) * (x2 - x1) / (y2 - y1)))
}

# ---- Static data ----

available_runs <- get_run_ids()

# ---- UI ----

ui <- page_navbar(
    title = "Stockfish Arena",
    theme = bs_theme(version = 5, bootswatch = "cosmo"),

    # ── Tab 1: Live Game ────────────────────────────────────────────────────
    nav_panel("\u2694 Live Game",
        layout_sidebar(
            sidebar = sidebar(width = 280,

                card(
                    card_header("Engine"),
                    selectInput("run_id", "Run",
                        choices = available_runs, selected = available_runs[1], width = "100%"),
                    selectInput("checkpoint", "Checkpoint",
                        choices = character(0), width = "100%")
                ),

                card(
                    card_header("Stockfish"),
                    sliderInput("sf_elo_live", "ELO",
                        min = 1320, max = 2500, value = 1320, step = 50, width = "100%")
                ),

                card(
                    card_header("Game"),
                    radioButtons("our_color", "Our colour",
                        choices = c("White", "Black", "Alternate"),
                        selected = "White", inline = TRUE),
                    actionButton("btn_new_game", "New Game",
                        class = "btn-primary w-100 mb-2"),
                    layout_columns(
                        col_widths = c(6, 6),
                        actionButton("btn_step", "Step \u2192",
                            class = "btn-outline-primary btn-sm w-100"),
                        input_switch("sw_auto", "Auto-play", value = FALSE)
                    ),
                    helpText("Depth\u20113. Engine compute (\u223c2s) sets minimum pace.")
                )
            ),

            layout_column_wrap(
                width = 1/4, fill = FALSE,
                value_box("Status",     uiOutput("vb_status"),    theme = "primary",
                          showcase = bs_icon("circle-fill")),
                value_box("Our engine", uiOutput("vb_engine"),    theme = "secondary",
                          showcase = bs_icon("cpu")),
                value_box("Stockfish",  uiOutput("vb_stockfish"), theme = "warning",
                          showcase = bs_icon("robot")),
                value_box("Moves",      uiOutput("vb_moves"),     theme = "light",
                          showcase = bs_icon("list-ol"))
            ),

            layout_columns(
                col_widths = c(5, 7),

                card(
                    full_screen = TRUE,
                    card_header(uiOutput("board_header")),
                    uiOutput("board_svg"),
                    card_footer(
                        style = "font-size:0.75em; color:#666; word-break:break-all;",
                        textOutput("fen_display")
                    )
                ),

                card(
                    full_screen = TRUE,
                    card_header("Move log"),
                    div(
                        style = "max-height:500px; overflow-y:auto;",
                        uiOutput("move_log")
                    )
                )
            )
        )
    ),

    # ── Tab 2: ELO Benchmark ────────────────────────────────────────────────
    nav_panel("\U0001f4ca ELO Benchmark",
        layout_sidebar(
            sidebar = sidebar(width = 280,

                card(
                    card_header("Run"),
                    selectInput("bm_run_id", "Run",
                        choices = available_runs, selected = available_runs[1], width = "100%"),
                    selectInput("bm_checkpoint", "Checkpoint",
                        choices = character(0), width = "100%")
                ),

                card(
                    card_header("Run a benchmark"),
                    p("Benchmarks run as a background job to keep your session free."),
                    p(tags$b("Configure and run:"),
                      tags$code("scripts/r/05_stockfish_benchmark.R")),
                    p("In RStudio: Jobs pane \u2192 Start Job \u2192 select the script."),
                    p("Results appear in the charts automatically once logged to the DB.",
                      style = "color:#666; font-size:0.85em;")
                )
            ),

            layout_column_wrap(
                width = 1/3, fill = FALSE,
                value_box("Estimated ELO", uiOutput("vb_est_elo"),
                    theme = "primary", showcase = bs_icon("bullseye"))
            ),

            layout_columns(
                col_widths = c(6, 6),

                card(
                    full_screen = TRUE,
                    card_header(uiOutput("elo_chart_header")),
                    plotOutput("elo_chart", height = "360px")
                ),

                card(
                    full_screen = TRUE,
                    card_header("ELO trajectory over training"),
                    plotOutput("elo_trajectory", height = "360px")
                )
            )
        )
    )
)

# ---- Server ----

server <- function(input, output, session) {

    # ── Shared: Stockfish process management ──────────────────────────────────

    sf_instance <- reactiveVal(NULL)

    open_sf <- function(elo) {
        old <- sf_instance()
        if (!is.null(old)) try(old$quit(), silent = TRUE)
        sf <- chess_engine$SimpleEngine$popen_uci(SF_PATH)
        sf$configure(list("UCI_LimitStrength" = TRUE, "UCI_Elo" = as.integer(elo)))
        sf_instance(sf)
    }

    close_sf <- function() {
        sf <- sf_instance()
        if (!is.null(sf)) try(sf$quit(), silent = TRUE)
        sf_instance(NULL)
    }

    onStop(close_sf)

    # ── Live Game: state ──────────────────────────────────────────────────────

    move_history <- reactiveVal(character(0))
    move_san_log <- reactiveVal(character(0))
    last_move_rv <- reactiveVal(NULL)
    our_color_rv <- reactiveVal(NULL)
    game_weights <- reactiveVal(NULL)   # frozen at game start
    game_active  <- reactiveVal(FALSE)
    auto_playing <- reactiveVal(FALSE)
    game_result  <- reactiveVal(NULL)
    game_count   <- reactiveVal(0L)

    current_board <- reactive({
        b <- chess_mod$Board()
        for (uci in move_history()) b$push(chess_mod$Move$from_uci(uci))
        b
    })

    # Populate checkpoint selector
    observeEvent(input$run_id, {
        cpts <- get_checkpoints(as.integer(input$run_id))
        updateSelectInput(session, "checkpoint", choices = cpts, selected = cpts[1])
    }, ignoreNULL = FALSE)

    # New game
    observeEvent(input$btn_new_game, {
        close_sf()

        n <- game_count() + 1L
        game_count(n)

        col <- switch(input$our_color,
            White     = chess_mod$WHITE,
            Black     = chess_mod$BLACK,
            Alternate = if (n %% 2 == 1L) chess_mod$WHITE else chess_mod$BLACK
        )
        our_color_rv(col)

        # Snapshot weights so changing selectors mid-game has no effect
        game_weights(load_weights_fn(
            as.integer(input$run_id),
            as.integer(input$checkpoint)
        ))

        move_history(character(0))
        move_san_log(character(0))
        last_move_rv(NULL)
        game_result(NULL)
        auto_playing(FALSE)

        open_sf(input$sf_elo_live)
        game_active(TRUE)

        # If Stockfish has first move (our engine is Black), advance once
        if (col == chess_mod$BLACK) advance_one_move()
    })

    # Manual step
    observeEvent(input$btn_step, {
        req(game_active())
        advance_one_move()
    })

    # Auto-play switch
    observeEvent(input$sw_auto, {
        auto_playing(as.logical(input$sw_auto))
    })

    # Auto-advance: reactiveTimer fires every 600ms; checks state before advancing
    auto_timer <- reactiveTimer(600)

    observe({
        auto_timer()
        req(auto_playing(), game_active())
        advance_one_move()
    })

    # Build the board from move_history() directly — avoids calling the
    # current_board reactive() expression from inside a non-output context
    make_board_now <- function() {
        b <- chess_mod$Board()
        for (uci in move_history()) b$push(chess_mod$Move$from_uci(uci))
        b
    }

    advance_one_move <- function() {
        b  <- make_board_now()
        wp <- game_weights()

        if (is.null(wp) || b$is_game_over() || !game_active()) {
            game_active(FALSE)
            auto_playing(FALSE)
            return()
        }

        tryCatch({
            mv <- if (b$turn == our_color_rv()) {
                engine_mod$get_best_move(b, wp, 3L, FALSE)[[1]]
            } else {
                sf <- sf_instance()
                if (is.null(sf)) return()
                sf$play(b, chess_engine$Limit(time = 0.5), ponder = FALSE)$move
            }

            if (is.null(mv)) return()

            san <- b$san(mv)
            move_history(c(move_history(), mv$uci()))
            move_san_log(c(move_san_log(), san))
            last_move_rv(mv)

            # Rebuild board to check game over — don't call current_board()
            b2 <- make_board_now()
            if (b2$is_game_over()) {
                res <- b2$result()
                game_result(res)
                game_active(FALSE)
                auto_playing(FALSE)
                log_live_result(res)
                close_sf()
            }
        }, error = function(e) {
            auto_playing(FALSE)
            game_active(FALSE)
            showNotification(paste("Move error:", conditionMessage(e)), type = "error")
        })
    }

    log_live_result <- function(result_str) {
        col        <- our_color_rv()
        engine_won <- switch(result_str,
            "1-0" = col == chess_mod$WHITE,
            "0-1" = col == chess_mod$BLACK,
            NA
        )
        wins   <- if (isTRUE(engine_won))  1L else 0L
        losses <- if (isFALSE(engine_won)) 1L else 0L
        draws  <- if (is.na(engine_won))   1L else 0L
        con    <- dbConnect(SQLite(), DB_PATH)
        dbExecute(con,
            "INSERT INTO stockfish_evals
             (run_id,game_number,sf_elo,n_games,wins,losses,draws,win_rate,source)
             VALUES (?,?,?,?,?,?,?,?,?)",
            list(as.integer(input$run_id), as.integer(input$checkpoint),
                 as.integer(input$sf_elo_live),
                 1L, wins, losses, draws, as.numeric(wins), "live"))
        dbDisconnect(con)
    }

    # ── Live Game: outputs ────────────────────────────────────────────────────

    output$board_header <- renderUI({
        sans <- move_san_log()
        n    <- length(sans)
        if (n == 0) return("Starting position")
        side <- if (n %% 2 == 1L) "White" else "Black"
        paste0("After ", side, "\u2009", ceiling(n / 2), ". ", sans[n])
    })

    output$board_svg <- renderUI({
        col <- our_color_rv()
        board_svg_html(current_board(),
            last_move = last_move_rv(),
            flipped   = !is.null(col) && col == chess_mod$BLACK)
    })

    output$fen_display <- renderText({ current_board()$fen() })

    output$move_log <- renderUI({
        sans <- move_san_log()
        if (length(sans) == 0) return(p("No moves yet.", style = "color:#999;"))

        rows <- lapply(seq(1, length(sans), by = 2), function(i) {
            move_no  <- ceiling(i / 2)
            white_mv <- sans[i]
            black_mv <- if ((i + 1L) <= length(sans)) sans[i + 1L] else ""
            tags$tr(
                tags$td(style = "color:#999;font-size:0.8em;padding:2px 6px 2px 2px;", paste0(move_no, ".")),
                tags$td(style = "font-family:monospace;padding:1px 4px;", white_mv),
                tags$td(style = "font-family:monospace;padding:1px 4px;", black_mv)
            )
        })

        # Append result banner if game over
        res <- game_result()
        banner <- if (!is.null(res)) {
            label <- switch(res, "1-0" = "White wins", "0-1" = "Black wins", "Draw")
            tags$tr(tags$td(colspan = "3",
                style = "text-align:center;padding:6px;font-weight:600;background:#e9ecef;border-radius:4px;",
                label))
        }

        tagList(
            tags$table(style = "width:100%;border-collapse:collapse;font-size:0.92em;",
                do.call(tags$tbody, c(rows, list(banner))))
        )
    })

    output$vb_status <- renderUI({
        res <- game_result()
        if (!is.null(res)) switch(res, "1-0"="White wins", "0-1"="Black wins", "Draw")
        else if (game_active()) {
            if (current_board()$turn == chess_mod$WHITE) "White to move" else "Black to move"
        } else "Ready"
    })

    output$vb_engine <- renderUI({
        col <- our_color_rv()
        if (is.null(col)) return("—")
        sprintf("Run %s (%s)", input$run_id,
                if (col == chess_mod$WHITE) "White" else "Black")
    })

    output$vb_stockfish <- renderUI({ paste0("ELO\u2009", input$sf_elo_live) })
    output$vb_moves     <- renderUI({ length(move_san_log()) })

    # ── ELO Benchmark ─────────────────────────────────────────────────────────
    # Results are written by 05_stockfish_benchmark.R (run as a background job).
    # This app reads from the DB and displays; it does not run benchmarks itself.

    refresh_bm <- reactiveVal(0L)

    observeEvent(input$bm_run_id, {
        cpts <- get_checkpoints(as.integer(input$bm_run_id))
        updateSelectInput(session, "bm_checkpoint", choices = cpts, selected = cpts[1])
    }, ignoreNULL = FALSE)

    # Poll for new results every 30 seconds so charts update while benchmark runs
    results_timer <- reactiveTimer(30000)

    bm_results <- reactive({
        results_timer()
        refresh_bm()
        req(input$bm_run_id)
        con <- dbConnect(SQLite(), DB_PATH)
        df  <- dbGetQuery(con, sprintf(
            "SELECT * FROM stockfish_evals WHERE run_id=%d AND source='benchmark' ORDER BY game_number, sf_elo",
            as.integer(input$bm_run_id)))
        dbDisconnect(con)
        df
    })

    bm_results_filtered <- reactive({
        req(input$bm_checkpoint)
        bm_results() |> filter(game_number == as.integer(input$bm_checkpoint))
    })

    output$elo_chart_header <- renderUI({
        paste0("Win rate vs Stockfish ELO \u2014 Run ", input$bm_run_id,
               ", Game ", input$bm_checkpoint)
    })

    output$vb_est_elo <- renderUI({
        df <- bm_results_filtered()
        if (nrow(df) == 0) "—" else estimate_elo(df)
    })

    output$elo_chart <- renderPlot({
        df <- bm_results_filtered()
        if (nrow(df) == 0) {
            return(ggplot() + theme_void() +
                annotate("text", x=0, y=0, label="No results yet", size=5, color="grey60"))
        }
        df |>
            mutate(fill_cat = case_when(
                win_rate > 0.6 ~ "high",
                win_rate < 0.4 ~ "low",
                TRUE           ~ "mid"
            )) |>
            ggplot(aes(x = win_rate, y = factor(sf_elo), fill = fill_cat)) +
            geom_col(width = 0.6) +
            geom_vline(xintercept = 0.5, linetype = "dashed", color = "grey40", linewidth = 0.6) +
            geom_text(aes(label = sprintf("%dW/%dL/%dD", wins, losses, draws)),
                      x = 0.02, hjust = 0, size = 3.5, color = "grey30") +
            scale_fill_manual(
                values = c(high = "#d4edda", mid = "#fff3cd", low = "#f8d7da"),
                guide  = "none") +
            scale_x_continuous(limits = c(0, 1), labels = scales::percent_format()) +
            labs(x = "Win rate", y = "Stockfish ELO") +
            theme_minimal(base_size = 13) +
            theme(panel.grid.major.y = element_blank())
    })

    output$elo_trajectory <- renderPlot({
        df <- bm_results()
        if (nrow(df) == 0) {
            return(ggplot() + theme_void() +
                annotate("text", x=0, y=0, label="Run a benchmark to see the trajectory",
                         size=4, color="grey60"))
        }

        traj <- df |>
            group_by(game_number) |>
            filter(n() >= 2) |>
            group_modify(~ tibble(est_elo_str = estimate_elo(.x))) |>
            filter(!grepl("[<>]", est_elo_str)) |>
            mutate(est_elo = as.numeric(str_remove(est_elo_str, "~")))

        if (nrow(traj) == 0) {
            return(ggplot() + theme_void() +
                annotate("text", x=0, y=0,
                    label="Need \u22652 checkpoints benchmarked to show trajectory",
                    size=4, color="grey60"))
        }

        ggplot(traj, aes(x = game_number, y = est_elo)) +
            geom_line(linewidth = 0.8) +
            geom_point(size = 2.5) +
            labs(x = "Training games (checkpoint)", y = "Estimated ELO",
                 title = paste0("Run ", input$bm_run_id)) +
            theme_minimal(base_size = 13)
    })
}

shinyApp(ui, server)
