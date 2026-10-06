# 04_game_viewer.R
# Stage 04: Interactive game viewer
# Select an impactful training game and step through it move by move.
#
# Requires: reticulate with python-chess installed in the active Python environment.
# Run from RStudio: open file and click Run App, or shiny::runApp("scripts/r/04_game_viewer.R")

library(shiny)
library(bslib)
library(DBI)
library(RSQLite)
library(tidyverse)
library(reticulate)

# Python chess modules — loaded once at startup
chess_mod <- import("chess")
chess_pgn <- import("chess.pgn")
chess_svg <- import("chess.svg")
io_mod    <- import("io")

DB_PATH <- if (file.exists("../../data/chess_learning.db")) {
  "../../data/chess_learning.db"
} else {
  "data/chess_learning.db"
}

# ── Helpers ───────────────────────────────────────────────────────────────────

load_impactful_games <- function(db_path, run_id_filter = NULL, n = 30) {
  con <- dbConnect(SQLite(), db_path)
  weight_deltas <- dbReadTable(con, "weight_deltas")
  game_records  <- dbReadTable(con, "game_records")
  games_tbl     <- dbReadTable(con, "games")
  dbDisconnect(con)

  if (!is.null(run_id_filter)) {
    weight_deltas <- weight_deltas |> filter(run_id == run_id_filter)
    games_tbl     <- games_tbl     |> filter(run_id == run_id_filter)
    game_records  <- game_records  |> filter(run_id == run_id_filter)
  }

  weight_deltas |>
    group_by(run_id, game_number) |>
    summarise(
      max_abs_delta = max(abs(delta)),
      top_feature   = feature_name[which.max(abs(delta))],
      .groups = "drop"
    ) |>
    left_join(
      games_tbl |> select(run_id, game_number, outcome, n_halfmoves),
      by = c("run_id", "game_number")
    ) |>
    left_join(
      game_records |> select(run_id, game_number, pgn, td_error),
      by = c("run_id", "game_number")
    ) |>
    arrange(desc(max_abs_delta)) |>
    head(n)
}

get_run_ids <- function(db_path) {
  con     <- dbConnect(SQLite(), db_path)
  run_ids <- dbGetQuery(con, "SELECT DISTINCT run_id FROM games ORDER BY run_id DESC")$run_id
  dbDisconnect(con)
  run_ids
}

parse_game <- function(pgn_str) {
  game  <- chess_pgn$read_game(io_mod$StringIO(pgn_str))
  board <- game$board()

  # Slot 1: starting position (no preceding move)
  fens  <- board$fen()
  sans  <- character(0)
  moves <- list(NULL)

  node <- game
  while (length(node$variations) > 0) {
    node  <- node$variations[[1]]
    move  <- node$move
    board$push(move)
    fens  <- c(fens,  board$fen())
    sans  <- c(sans,  node$san())
    moves <- c(moves, list(move))   # python Move objects for lastmove highlighting
  }

  list(fens = fens, sans = sans, moves = moves, n_moves = length(sans))
}

board_svg_html <- function(fen, last_move = NULL, flipped = FALSE, size = 390L) {
  board <- chess_mod$Board(fen)
  args  <- list(board = board, size = size, flipped = flipped)
  if (!is.null(last_move)) args$lastmove <- last_move
  if (board$is_check()) {
    king_sq <- board$king(board$turn)
    if (!is.null(king_sq)) args$check <- king_sq
  }
  svg_str <- do.call(chess_svg$board, args)
  HTML(paste0('<div style="text-align:center; line-height:0;">', svg_str, "</div>"))
}

# ── App data (loaded once at startup) ────────────────────────────────────────

available_runs <- get_run_ids(DB_PATH)

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- page_sidebar(
  title = "Chess Learning \u2014 Game Viewer",
  theme = bs_theme(version = 5, bootswatch = "cosmo"),

  sidebar = sidebar(
    width = 310,

    card(
      card_header("Select run"),
      selectInput(
        "run_id", NULL,
        choices  = available_runs,
        selected = available_runs[1],
        width    = "100%"
      )
    ),

    card(
      card_header("Select game"),
      selectInput(
        "game_idx", NULL,
        choices  = character(0),
        selected = NULL,
        width    = "100%"
      )
    ),

    card(
      card_header("Navigate"),
      layout_columns(
        col_widths = c(3, 3, 3, 3),
        actionButton("btn_start", "|<", class = "btn-outline-secondary btn-sm w-100"),
        actionButton("btn_prev",  "< ", class = "btn-outline-primary   btn-sm w-100"),
        actionButton("btn_next",  " >", class = "btn-outline-primary   btn-sm w-100"),
        actionButton("btn_end",   ">|", class = "btn-outline-secondary btn-sm w-100")
      ),
      sliderInput(
        "move_slider", NULL,
        min = 0, max = 1, value = 0, step = 1,
        ticks = FALSE, width = "100%"
      ),
      input_switch("flip_board", "Flip board", value = FALSE)
    ),

    card(
      card_header("Game info"),
      uiOutput("game_info")
    )
  ),

  layout_columns(
    col_widths = c(7, 5),

    card(
      full_screen = TRUE,
      card_header(uiOutput("board_header")),
      uiOutput("board_svg")
    ),

    card(
      full_screen = TRUE,
      card_header("Move list"),
      div(
        style = "max-height: 560px; overflow-y: auto;",
        uiOutput("move_list")
      )
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # Reload impactful games when run changes
  dat <- reactive({
    load_impactful_games(DB_PATH, run_id_filter = as.integer(input$run_id))
  })

  # Update game selector when run changes
  observeEvent(dat(), {
    d <- dat()
    labels <- sprintf(
      "Game %d | %s | \u0394=%.4f (%s)",
      d$game_number, d$outcome, d$max_abs_delta, d$top_feature
    )
    updateSelectInput(session, "game_idx",
                      choices  = setNames(seq_len(nrow(d)), labels),
                      selected = 1)
  })

  # Parsed game: list of fens, sans, moves
  parsed <- reactive({
    req(input$game_idx)
    parse_game(dat()$pgn[as.integer(input$game_idx)])
  })

  # Current move counter: 0 = starting position
  move_num <- reactiveVal(0L)

  # Reset on game change
  observeEvent(parsed(), {
    n <- parsed()$n_moves
    move_num(0L)
    updateSliderInput(session, "move_slider", max = n, value = 0L)
  })

  # Slider drives move counter
  observeEvent(input$move_slider, ignoreInit = TRUE, {
    move_num(as.integer(input$move_slider))
  })

  # Helper: set move and sync slider
  go_to <- function(val) {
    move_num(val)
    updateSliderInput(session, "move_slider", value = val)
  }

  observeEvent(input$btn_prev,  { go_to(max(0L,                  move_num() - 1L)) })
  observeEvent(input$btn_next,  { go_to(min(parsed()$n_moves,    move_num() + 1L)) })
  observeEvent(input$btn_start, { go_to(0L) })
  observeEvent(input$btn_end,   { go_to(parsed()$n_moves) })

  # ── Board ──────────────────────────────────────────────────────────────────

  output$board_header <- renderUI({
    mn <- move_num()
    g  <- parsed()
    if (mn == 0) {
      "Starting position"
    } else {
      color    <- if (mn %% 2 == 1) "White" else "Black"
      move_num_display <- ceiling(mn / 2)
      paste0("After ", color, " ", move_num_display, ". ", g$sans[mn])
    }
  })

  output$board_svg <- renderUI({
    g  <- parsed()
    mn <- move_num()
    board_svg_html(
      fen       = g$fens[mn + 1],
      last_move = g$moves[[mn + 1]],
      flipped   = input$flip_board
    )
  })

  # ── Move list ──────────────────────────────────────────────────────────────

  output$move_list <- renderUI({
    g  <- parsed()
    mn <- move_num()

    if (g$n_moves == 0) return(p("No moves recorded."))

    rows <- lapply(seq(1, g$n_moves, by = 2), function(i) {
      move_no  <- ceiling(i / 2)
      white_mv <- g$sans[i]
      black_mv <- if ((i + 1) <= g$n_moves) g$sans[i + 1] else ""

      hl <- function(ply) {
        if (mn == ply) {
          "background:#cce5ff; border-radius:3px; padding:1px 4px; font-weight:600;"
        } else {
          "padding:1px 4px;"
        }
      }

      tags$tr(
        tags$td(
          style = "color:#999; font-size:0.8em; padding:2px 6px 2px 2px; white-space:nowrap;",
          paste0(move_no, ".")
        ),
        tags$td(style = paste0("font-family:monospace; ", hl(i)),       white_mv),
        tags$td(style = paste0("font-family:monospace; ", hl(i + 1L)), black_mv)
      )
    })

    tagList(
      tags$table(
        style = "width:100%; border-collapse:collapse; font-size:0.92em;",
        do.call(tags$tbody, rows)
      )
    )
  })

  # ── Game info ──────────────────────────────────────────────────────────────

  output$game_info <- renderUI({
    idx <- as.integer(input$game_idx)
    row <- dat()[idx, ]

    items <- list(
      "Run"          = row$run_id,
      "Game"         = row$game_number,
      "Outcome"      = row$outcome,
      "Half-moves"   = row$n_halfmoves,
      "TD error"     = round(row$td_error, 4),
      "Top feature"  = row$top_feature,
      "Max |\u0394|" = round(row$max_abs_delta, 4)
    )

    rows <- lapply(names(items), function(nm) {
      tags$tr(
        tags$td(style = "color:#666; padding:2px 10px 2px 0; white-space:nowrap;", nm),
        tags$td(style = "font-weight:600;", as.character(items[[nm]]))
      )
    })

    tags$table(
      style = "width:100%; font-size:0.88em;",
      do.call(tags$tbody, rows)
    )
  })
}

shinyApp(ui, server)
