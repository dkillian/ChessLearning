# 05_search_explorer.R
# Interactive search explorer: score all legal moves from any board position
# using the trained engine weights at a configurable search depth.
#
# Run from RStudio: open file and click Run App, or:
#   shiny::runApp("scripts/r/05_search_explorer.R")

library(shiny)
library(bslib)
library(bsicons)
library(DBI)
library(RSQLite)
library(tidyverse)
library(reticulate)

# ---- Python modules ----

chess_mod <- import("chess")
chess_svg  <- import("chess.svg")

DB_PATH     <- if (file.exists("../../data/chess_learning.db")) "../../data/chess_learning.db" else "data/chess_learning.db"
ENGINE_PATH <- if (file.exists("../python/engine.py")) "../python" else "scripts/python"

engine <- import_from_path("engine", path = ENGINE_PATH)

# ---- Helpers ----

get_run_ids <- function() {
    con     <- dbConnect(SQLite(), DB_PATH)
    ids     <- dbGetQuery(con, "SELECT DISTINCT run_id FROM weights ORDER BY run_id DESC")$run_id
    dbDisconnect(con)
    ids
}

load_weights <- function(run_id) {
    con  <- dbConnect(SQLite(), DB_PATH)
    rows <- dbGetQuery(con, sprintf("
        SELECT feature_name, weight_value FROM weights
        WHERE run_id = %d
          AND game_number = (SELECT MAX(game_number) FROM weights WHERE run_id = %d)
    ", run_id, run_id))
    dbDisconnect(con)
    as.list(setNames(rows$weight_value, rows$feature_name))
}

board_svg_html <- function(board, last_move = NULL, size = 380L) {
    args <- list(board = board, size = size)
    if (!is.null(last_move)) args$lastmove <- last_move
    if (board$is_check()) {
        king_sq <- board$king(board$turn)
        if (!is.null(king_sq)) args$check <- king_sq
    }
    svg_str <- do.call(chess_svg$board, args)
    HTML(paste0('<div style="text-align:center; line-height:0;">', svg_str, "</div>"))
}

make_board <- function(base_fen, history) {
    b <- chess_mod$Board(base_fen)
    for (uci in history) b$push(chess_mod$Move$from_uci(uci))
    b
}

score_moves <- function(board, weights_py, depth) {
    moves <- reticulate::iterate(board$legal_moves)
    if (length(moves) == 0) return(tibble(san = character(), uci = character(), score = numeric()))
    purrr::map(moves, function(m) {
        san <- board$san(m)
        uci <- m$uci()
        board$push(m)
        s <- -engine$negamax(board, as.integer(depth - 1), -1e6, 1e6, weights_py)
        board$pop()
        tibble(san = san, uci = uci, score = as.numeric(s))
    }) |>
        purrr::list_rbind() |>
        arrange(desc(score)) |>
        mutate(rank = row_number())
}

# ---- Static data ----

available_runs <- get_run_ids()
STARTING_FEN   <- chess_mod$Board()$fen()

# ---- UI ----

ui <- page_sidebar(
    title = "Search Explorer",
    theme = bs_theme(version = 5, bootswatch = "cosmo"),

    sidebar = sidebar(
        width = 280,

        card(
            card_header("Weights"),
            selectInput("run_id", "Run",
                choices  = available_runs,
                selected = available_runs[1],
                width    = "100%"
            )
        ),

        card(
            card_header("Search depth"),
            sliderInput("depth", NULL,
                min = 1, max = 3, value = 2, step = 1,
                ticks = TRUE, width = "100%"
            ),
            helpText("Depth 3 may take a few seconds.")
        ),

        card(
            card_header("Position"),
            textAreaInput("fen_input", "FEN (leave blank for start)",
                value = "", rows = 3, width = "100%"
            ),
            layout_columns(
                col_widths = c(6, 6),
                actionButton("btn_set_fen",  "Set FEN",  class = "btn-outline-primary  btn-sm w-100"),
                actionButton("btn_reset",    "Reset",    class = "btn-outline-secondary btn-sm w-100")
            )
        ),

        card(
            card_header("Play a move"),
            selectInput("move_select", NULL,
                choices  = character(0),
                selected = NULL,
                width    = "100%"
            ),
            layout_columns(
                col_widths = c(6, 6),
                actionButton("btn_play", "Play →", class = "btn-primary  btn-sm w-100"),
                actionButton("btn_undo", "← Undo", class = "btn-outline-secondary btn-sm w-100")
            )
        )
    ),

    layout_column_wrap(
        width  = 1/3,
        fill   = FALSE,
        value_box(
            title   = "Best move",
            value   = uiOutput("vb_best_move"),
            theme   = "primary",
            showcase = bs_icon("trophy")
        ),
        value_box(
            title   = "Best score",
            value   = uiOutput("vb_best_score"),
            theme   = "success",
            showcase = bs_icon("graph-up-arrow")
        ),
        value_box(
            title   = "Legal moves",
            value   = uiOutput("vb_n_moves"),
            theme   = "secondary",
            showcase = bs_icon("grid-3x3")
        )
    ),

    layout_columns(
        col_widths = c(5, 7),

        card(
            full_screen = TRUE,
            card_header(uiOutput("board_header")),
            uiOutput("board_svg"),
            card_footer(
                style = "font-size:0.78em; color:#666; word-break:break-all;",
                textOutput("fen_display")
            )
        ),

        card(
            full_screen = TRUE,
            card_header(uiOutput("scores_header")),
            plotOutput("scores_plot", height = "440px")
        )
    )
)

# ---- Server ----

server <- function(input, output, session) {

    # ── State ──────────────────────────────────────────────────────────────────

    base_fen     <- reactiveVal(STARTING_FEN)
    move_history <- reactiveVal(character(0))
    last_move_py <- reactiveVal(NULL)

    # ── Weights (reload when run changes) ──────────────────────────────────────

    weights_py <- reactive({
        load_weights(as.integer(input$run_id))
    })

    # ── Board state ────────────────────────────────────────────────────────────

    current_board <- reactive({
        make_board(base_fen(), move_history())
    })

    # ── Move scores (recompute on position or depth change) ────────────────────

    scored <- reactive({
        b <- current_board()
        if (b$is_game_over()) return(tibble(san = character(), uci = character(), score = numeric(), rank = integer()))
        score_moves(b, weights_py(), input$depth)
    }) |> bindEvent(current_board(), input$depth, input$run_id)

    # ── Set FEN ────────────────────────────────────────────────────────────────

    observeEvent(input$btn_set_fen, {
        fen_str <- trimws(input$fen_input)
        if (nchar(fen_str) == 0) return()
        tryCatch({
            chess_mod$Board(fen_str)   # validate
            base_fen(fen_str)
            move_history(character(0))
            last_move_py(NULL)
        }, error = function(e) {
            showNotification("Invalid FEN — could not parse position.", type = "error")
        })
    })

    observeEvent(input$btn_reset, {
        base_fen(STARTING_FEN)
        move_history(character(0))
        last_move_py(NULL)
    })

    # ── Play / Undo ────────────────────────────────────────────────────────────

    observeEvent(input$btn_play, {
        req(input$move_select)
        uci <- input$move_select
        if (!nzchar(uci)) return()
        b   <- current_board()
        m   <- chess_mod$Move$from_uci(uci)
        if (!(uci %in% sapply(reticulate::iterate(b$legal_moves), \(x) x$uci()))) {
            showNotification("Move is no longer legal.", type = "warning")
            return()
        }
        last_move_py(m)
        move_history(c(move_history(), uci))
    })

    observeEvent(input$btn_undo, {
        h <- move_history()
        if (length(h) == 0) return()
        move_history(h[-length(h)])
        last_move_py(if (length(h) > 1) chess_mod$Move$from_uci(h[length(h) - 1]) else NULL)
    })

    # ── Populate move selector from scored table ───────────────────────────────

    observe({
        df <- scored()
        if (nrow(df) == 0) {
            updateSelectInput(session, "move_select", choices = character(0))
            return()
        }
        labels  <- sprintf("%s  (%+.3f)", df$san, df$score)
        choices <- setNames(df$uci, labels)
        updateSelectInput(session, "move_select", choices = choices, selected = choices[1])
    })

    # ── Outputs ────────────────────────────────────────────────────────────────

    output$board_header <- renderUI({
        b  <- current_board()
        h  <- move_history()
        side <- if (b$turn == chess_mod$WHITE) "White to move" else "Black to move"
        if (length(h) == 0) side else paste0(side, " — after ", h[length(h)])
    })

    output$board_svg <- renderUI({
        board_svg_html(current_board(), last_move = last_move_py())
    })

    output$fen_display <- renderText({ current_board()$fen() })

    output$scores_header <- renderUI({
        df <- scored()
        if (nrow(df) == 0) return("Move scores")
        paste0("Move scores — depth ", input$depth, " — ", nrow(df), " moves")
    })

    output$vb_best_move <- renderUI({
        df <- scored()
        if (nrow(df) == 0) return("—")
        df$san[1]
    })

    output$vb_best_score <- renderUI({
        df <- scored()
        if (nrow(df) == 0) return("—")
        sprintf("%+.3f", df$score[1])
    })

    output$vb_n_moves <- renderUI({
        df <- scored()
        nrow(df)
    })

    output$scores_plot <- renderPlot({
        df <- scored()
        if (nrow(df) == 0) {
            ggplot() + annotate("text", x = 0, y = 0, label = "Game over", size = 6) +
                theme_void()
        } else {
            df |>
                mutate(
                    san  = fct_reorder(san, score),
                    best = rank == 1
                ) |>
                ggplot(aes(x = score, y = san, fill = best)) +
                geom_col() +
                geom_vline(xintercept = 0, linewidth = 0.4, color = "grey40") +
                geom_text(aes(label = sprintf("%+.3f", score),
                              x = score + sign(score) * 0.001),
                          hjust = ifelse(df$score >= 0, -0.1, 1.1),
                          size  = 3, color = "grey30") +
                scale_fill_manual(values = c("TRUE" = "#0d6efd", "FALSE" = "#adc8f7"),
                                  guide  = "none") +
                labs(
                    x = "Score (side to move's perspective)",
                    y = NULL,
                    title = NULL
                ) +
                theme_minimal(base_size = 13) +
                theme(
                    panel.grid.major.y = element_blank(),
                    panel.grid.minor   = element_blank()
                )
        }
    })
}

shinyApp(ui, server)
