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
library(DiagrammeR)

# Python chess modules — loaded once at startup
chess_mod  <- import("chess")
chess_pgn  <- import("chess.pgn")
chess_svg  <- import("chess.svg")
io_mod     <- import("io")
engine_mod <- import_from_path("engine",
    path = if (file.exists("../python/engine.py")) "../python" else "scripts/python")

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

get_sf_run_ids <- function(db_path) {
  con <- dbConnect(SQLite(), db_path)
  ids <- tryCatch(
    dbGetQuery(con, "SELECT DISTINCT run_id FROM stockfish_games ORDER BY run_id DESC")$run_id,
    error = function(e) integer(0)
  )
  dbDisconnect(con)
  ids
}

get_sf_elo_levels <- function(db_path, run_id) {
  con  <- dbConnect(SQLite(), db_path)
  elos <- dbGetQuery(con, sprintf(
    "SELECT DISTINCT sf_elo FROM stockfish_games WHERE run_id=%d ORDER BY sf_elo",
    as.integer(run_id)))$sf_elo
  dbDisconnect(con)
  elos
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

load_weights_vec <- function(run_id, game_number) {
  con  <- dbConnect(SQLite(), DB_PATH)
  rows <- dbGetQuery(con, sprintf(
    "SELECT feature_name, weight_value FROM weights
     WHERE run_id=%d AND game_number=%d",
    as.integer(run_id), as.integer(game_number)))
  dbDisconnect(con)
  setNames(rows$weight_value, rows$feature_name)
}

get_sf_games <- function(db_path, run_id, sf_elo) {
  con <- dbConnect(SQLite(), db_path)
  df  <- dbGetQuery(con, sprintf(
    "SELECT game_id, game_number, sf_elo, game_index, engine_color,
            result, engine_won, n_half, pgn
     FROM stockfish_games
     WHERE run_id=%d AND sf_elo=%d
     ORDER BY game_id",
    as.integer(run_id), as.integer(sf_elo)))
  dbDisconnect(con)
  df
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

available_runs    <- get_run_ids(DB_PATH)
available_sf_runs <- get_sf_run_ids(DB_PATH)

# ── UI ────────────────────────────────────────────────────────────────────────

ui <- page_sidebar(
  title = "Chess Learning \u2014 Game Viewer",
  theme = bs_theme(version = 5, bootswatch = "cosmo"),

  sidebar = sidebar(
    width = 310,

    card(
      card_header("Source"),
      radioButtons("source_mode", NULL,
        choices  = c("Self-play", "Stockfish vs Engine"),
        selected = "Self-play",
        inline   = TRUE
      )
    ),

    # ── Self-play selectors ──────────────────────────────────────────────────
    conditionalPanel("input.source_mode == 'Self-play'",
      card(
        card_header("Select run"),
        selectInput("run_id", NULL,
          choices = available_runs, selected = available_runs[1], width = "100%")
      ),
      card(
        card_header("Select game"),
        selectInput("game_idx", NULL,
          choices = character(0), selected = NULL, width = "100%")
      )
    ),

    # ── Stockfish selectors ──────────────────────────────────────────────────
    conditionalPanel("input.source_mode == 'Stockfish vs Engine'",
      card(
        card_header("Select run"),
        selectInput("sf_run_id", NULL,
          choices  = if (length(available_sf_runs)) available_sf_runs else "None",
          selected = if (length(available_sf_runs)) available_sf_runs[1] else "None",
          width    = "100%")
      ),
      card(
        card_header("Stockfish ELO"),
        selectInput("sf_elo_select", NULL,
          choices = character(0), selected = NULL, width = "100%")
      ),
      card(
        card_header("Select game"),
        selectInput("sf_game_idx", NULL,
          choices = character(0), selected = NULL, width = "100%")
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
  ),

  # ── Analysis section ────────────────────────────────────────────────────────
  layout_columns(
    col_widths = c(5, 7),

    card(
      full_screen = TRUE,
      card_header("Why this move?"),
      layout_columns(
        col_widths = c(5, 3, 4),
        selectInput("analysis_depth", "Depth", choices = 1:3, selected = 2,
                    width = "100%"),
        actionButton("btn_analyze", "Analyze",
                     class = "btn-primary btn-sm w-100",
                     style = "margin-top:24px;"),
        selectInput("tree_candidate", "Show tree for",
                    choices = character(0), width = "100%")
      ),
      plotOutput("analysis_bar_chart", height = "300px"),
      hr(),
      strong("Feature breakdown \u2014 resulting position"),
      tableOutput("analysis_feature_table")
    ),

    card(
      full_screen = TRUE,
      card_header("Search tree \u2014 top 3 branches per level"),
      grVizOutput("search_tree_diagram", height = "420px")
    )
  )
)

# ── Server ────────────────────────────────────────────────────────────────────

server <- function(input, output, session) {

  # ── Self-play mode ───────────────────────────────────────────────────────────

  dat <- reactive({
    req(input$source_mode == "Self-play")
    load_impactful_games(DB_PATH, run_id_filter = as.integer(input$run_id))
  })

  observeEvent(dat(), {
    d <- dat()
    labels <- sprintf("Game %d | %s | \u0394=%.4f (%s)",
                      d$game_number, d$outcome, d$max_abs_delta, d$top_feature)
    updateSelectInput(session, "game_idx",
                      choices = setNames(seq_len(nrow(d)), labels), selected = 1)
  })

  # ── Stockfish mode ───────────────────────────────────────────────────────────

  observeEvent(input$sf_run_id, {
    elos <- get_sf_elo_levels(DB_PATH, as.integer(input$sf_run_id))
    updateSelectInput(session, "sf_elo_select", choices = elos, selected = elos[1])
  }, ignoreNULL = FALSE)

  sf_dat <- reactive({
    req(input$source_mode == "Stockfish vs Engine",
        input$sf_run_id, input$sf_elo_select)
    get_sf_games(DB_PATH, as.integer(input$sf_run_id), as.integer(input$sf_elo_select))
  })

  observeEvent(sf_dat(), {
    df <- sf_dat()
    if (nrow(df) == 0) return()
    labels <- sprintf("Game %d — Engine as %s — %s (%d plies)",
                      df$game_index, df$engine_color, df$result, df$n_half)
    updateSelectInput(session, "sf_game_idx",
                      choices = setNames(seq_len(nrow(df)), labels), selected = 1)
  })

  # ── Parsed game (shared by both modes) ───────────────────────────────────────

  parsed <- reactive({
    if (input$source_mode == "Self-play") {
      req(input$game_idx)
      parse_game(dat()$pgn[as.integer(input$game_idx)])
    } else {
      req(input$sf_game_idx)
      df <- sf_dat()
      req(nrow(df) > 0)
      parse_game(df$pgn[as.integer(input$sf_game_idx)])
    }
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
    if (input$source_mode == "Self-play") {
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
    } else {
      df  <- sf_dat()
      req(nrow(df) > 0)
      row <- df[as.integer(input$sf_game_idx), ]
      won_label <- if (is.na(row$engine_won)) "Draw"
                   else if (row$engine_won == 1L) "Win" else "Loss"
      items <- list(
        "Run"            = input$sf_run_id,
        "Checkpoint"     = row$game_number,
        "Stockfish ELO"  = row$sf_elo,
        "Engine played"  = row$engine_color,
        "Result"         = row$result,
        "Engine result"  = won_label,
        "Half-moves"     = row$n_half
      )
    }

    rows <- lapply(names(items), function(nm) {
      tags$tr(
        tags$td(style = "color:#666; padding:2px 10px 2px 0; white-space:nowrap;", nm),
        tags$td(style = "font-weight:600;", as.character(items[[nm]]))
      )
    })
    tags$table(style = "width:100%; font-size:0.88em;", do.call(tags$tbody, rows))
  })

  # ── Analysis: weights + helpers ─────────────────────────────────────────────

  # Load weights appropriate for the current game source and selection
  analysis_weights <- eventReactive(input$btn_analyze, {
    if (input$source_mode == "Self-play") {
      req(input$game_idx)
      row <- dat()[as.integer(input$game_idx), ]
      load_weights_vec(row$run_id, row$game_number)
    } else {
      req(input$sf_game_idx)
      df  <- sf_dat()
      req(nrow(df) > 0)
      row <- df[as.integer(input$sf_game_idx), ]
      load_weights_vec(as.integer(input$sf_run_id), row$game_number)
    }
  })

  # FEN before the current move (position to analyze)
  analysis_fen <- eventReactive(input$btn_analyze, {
    mn <- move_num()
    if (mn == 0) parsed()$fens[1] else parsed()$fens[mn]
  })

  # SAN of the move actually played (to highlight in the bar chart)
  analysis_played_san <- eventReactive(input$btn_analyze, {
    mn <- move_num()
    if (mn == 0 || mn > parsed()$n_moves) NA_character_ else parsed()$sans[mn]
  })

  # Score all legal moves at the analysis position
  analysis_scores <- eventReactive(input$btn_analyze, {
    fen      <- analysis_fen()
    wp       <- as.list(analysis_weights())
    depth    <- as.integer(input$analysis_depth)
    played   <- analysis_played_san()

    board <- chess_mod$Board(fen)
    moves <- reticulate::iterate(board$legal_moves)
    if (length(moves) == 0) return(tibble(san=character(), score=numeric(), played=logical()))

    purrr::map(moves, function(mv) {
      san <- board$san(mv)
      board$push(mv)
      s <- -as.numeric(engine_mod$negamax(board, as.integer(depth - 1L), -1e6, 1e6, wp))
      board$pop()
      tibble(san = san, uci = mv$uci(), score = s,
             played = !is.na(played) && san == played)
    }) |>
      purrr::list_rbind() |>
      arrange(desc(score)) |>
      mutate(rank = row_number())
  })

  # Populate candidate selector from scored moves
  observeEvent(analysis_scores(), {
    df      <- analysis_scores()
    labels  <- sprintf("%s (%+.3f)%s", df$san, df$score,
                       ifelse(df$played, " \u2605", ""))
    choices <- setNames(df$uci, labels)
    # Default to the played move if present
    played_uci <- df$uci[df$played]
    sel <- if (length(played_uci)) played_uci[1] else choices[1]
    updateSelectInput(session, "tree_candidate", choices = choices, selected = sel)
  })

  # ── Analysis outputs ─────────────────────────────────────────────────────────

  output$analysis_bar_chart <- renderPlot({
    df <- analysis_scores()
    req(nrow(df) > 0)
    df |>
      mutate(san = fct_reorder(san, score)) |>
      ggplot(aes(x = score, y = san, fill = played)) +
      geom_col(width = 0.7) +
      geom_vline(xintercept = 0, linewidth = 0.4, color = "grey40") +
      scale_fill_manual(
        values = c("TRUE" = "#0d6efd", "FALSE" = "#adc8f7"),
        guide  = "none") +
      labs(x = "Score (side to move\u2019s perspective)", y = NULL,
           title = sprintf("All legal moves \u2014 depth %s", input$analysis_depth)) +
      theme_minimal(base_size = 12) +
      theme(panel.grid.major.y = element_blank())
  })

  output$analysis_feature_table <- renderUI({
    df <- analysis_scores()
    req(nrow(df) > 0)
    wp       <- analysis_weights()
    top3     <- head(df, 3)
    base_fen <- analysis_fen()

    # For each top-3 move: extract feature values and compute contributions
    move_data <- purrr::map(seq_len(nrow(top3)), function(i) {
      b <- chess_mod$Board(base_fen)
      b$push(chess_mod$Move$from_uci(top3$uci[i]))
      feats_py <- engine_mod$extract_features(b)
      vals     <- setNames(as.numeric(unlist(feats_py)), names(feats_py))
      list(vals = vals, contrib = vals * wp[names(vals)])
    })

    feat_names_all <- names(move_data[[1]]$vals)
    # Sort by absolute contribution of the top move
    feat_order <- feat_names_all[order(-abs(move_data[[1]]$contrib))]

    mono <- "font-family:monospace; text-align:right; padding:2px 6px;"
    col_color <- function(val)
      if (val > 0) paste0(mono, "color:#155724;")
      else if (val < 0) paste0(mono, "color:#842029;")
      else paste0(mono, "color:#666;")
    th_style <- "border:1px solid #dee2e6; padding:4px 6px; text-align:right; background:#f8f9fa;"
    td_label <- "border:1px solid #dee2e6; padding:2px 6px; color:#555; font-size:0.85em;"

    # Top-level header: Feature (rowspan=2) | Weight (rowspan=2) | move names (colspan=2 each)
    move_headers <- purrr::map(seq_len(nrow(top3)), function(i) {
      label <- sprintf("%s (%+.3f)%s", top3$san[i], top3$score[i],
                       if (top3$played[i]) " \u2605" else "")
      col_style <- if (top3$played[i])
        "border:1px solid #dee2e6; padding:4px 6px; text-align:center; background:#cce5ff; font-weight:700;"
      else
        "border:1px solid #dee2e6; padding:4px 6px; text-align:center; background:#f8f9fa;"
      tags$th(colspan = "2", style = col_style, label)
    })

    # Sub-headers: Value | Contrib for each move (Weight is shown once in row 1)
    sub_headers <- purrr::map(seq_len(nrow(top3)), function(i) {
      list(tags$th(style = th_style, "Value"),
           tags$th(style = th_style, "Contrib"))
    }) |> purrr::list_flatten()

    # Data rows: feature | weight (once) | value | contrib for each move
    body_rows <- purrr::map(feat_order, function(fn) {
      w     <- wp[[fn]]
      cells <- purrr::map(seq_len(nrow(top3)), function(i) {
        v <- move_data[[i]]$vals[[fn]]
        c <- move_data[[i]]$contrib[[fn]]
        list(
          tags$td(style = paste0(mono, "color:#333;"), sprintf("%+.3f", v)),
          tags$td(style = col_color(c),                sprintf("%+.4f", c))
        )
      }) |> purrr::list_flatten()

      tags$tr(
        tags$td(style = td_label, fn),
        tags$td(style = paste0(mono, "color:#333;"), sprintf("%.4f", w)),
        do.call(tagList, cells)
      )
    })

    tags$table(
      style = "width:100%; border-collapse:collapse; font-size:0.85em;",
      tags$thead(
        tags$tr(
          tags$th(style = th_style, rowspan = "2", "Feature"),
          tags$th(style = th_style, rowspan = "2", "Weight"),
          do.call(tagList, move_headers)
        ),
        tags$tr(do.call(tagList, sub_headers))
      ),
      do.call(tags$tbody, body_rows)
    )
  })

  # ── Search tree helpers ──────────────────────────────────────────────────────

  # Build top-3 tree: returns a nested list suitable for Mermaid rendering
  build_search_tree <- function(board, wp, candidate_uci, played_san) {
    # Level 0: the candidate move
    mv0   <- chess_mod$Move$from_uci(candidate_uci)
    san0  <- board$san(mv0)
    score0 <- analysis_scores()$score[analysis_scores()$uci == candidate_uci]
    if (length(score0) == 0) score0 <- NA_real_

    board$push(mv0)

    # Level 1: opponent's top-3 responses (score at depth=0 = evaluate)
    opp_moves <- reticulate::iterate(board$legal_moves)
    if (length(opp_moves) == 0) { board$pop(); return(NULL) }

    opp_scored <- purrr::map(opp_moves, function(mv) {
      san <- board$san(mv)
      board$push(mv)
      s <- -as.numeric(engine_mod$evaluate(board, as.list(wp)))
      board$pop()
      tibble(san = san, uci = mv$uci(), score = s)
    }) |> purrr::list_rbind()

    top_opp <- opp_scored |> arrange(desc(score)) |> head(3)

    # Level 2: our top-3 replies for each opponent move
    l2_nodes <- purrr::map(seq_len(nrow(top_opp)), function(j) {
      mv1 <- chess_mod$Move$from_uci(top_opp$uci[j])
      board$push(mv1)

      our_moves <- reticulate::iterate(board$legal_moves)
      if (length(our_moves) == 0) {
        board$pop()
        return(list(opp_san = top_opp$san[j], opp_score = top_opp$score[j],
                    replies = tibble(san=character(), score=numeric())))
      }

      our_scored <- purrr::map(our_moves, function(mv) {
        san <- board$san(mv)
        board$push(mv)
        s <- -as.numeric(engine_mod$evaluate(board, as.list(wp)))
        board$pop()
        tibble(san = san, uci = mv$uci(), score = s)
      }) |> purrr::list_rbind()

      board$pop()
      list(
        opp_san   = top_opp$san[j],
        opp_score = top_opp$score[j],
        replies   = our_scored |> arrange(desc(score)) |> head(3)
      )
    })

    board$pop()  # undo candidate move

    list(
      san       = san0,
      uci       = candidate_uci,
      score     = score0,
      is_played = san0 == played_san,
      l2        = l2_nodes
    )
  }

  # Convert nested tree list to Mermaid graph code
  # Uses classDef for styling and simple labels to maximise DiagrammeR compatibility
  # Build Graphviz DOT string — more reliably rendered by DiagrammeR than Mermaid
  build_graphviz <- function(tree) {
    if (is.null(tree)) return(
      'digraph G { node [shape=box] A [label="No moves available"] }')

    safe_id <- function(s) paste0("N", gsub("[^A-Za-z0-9]", "_", s))

    played_tag <- if (isTRUE(tree$is_played)) " [played]" else ""
    score_str  <- if (!is.na(tree$score)) sprintf(" %+.3f", tree$score) else ""
    root_label <- sprintf("%s%s%s", tree$san, score_str, played_tag)

    nodes <- sprintf(
      '  M0 [label="%s", style=filled, fillcolor="#4472C4", fontcolor=white, shape=box]',
      root_label)
    edges <- character(0)

    for (j in seq_along(tree$l2)) {
      node   <- tree$l2[[j]]
      opp_id <- safe_id(paste0("R", j, node$opp_san))
      nodes  <- c(nodes, sprintf(
        '  %s [label="%s %+.3f", style=filled, fillcolor="#FFE0B2", shape=diamond]',
        opp_id, node$opp_san, node$opp_score))
      edges  <- c(edges, sprintf("  M0 -> %s", opp_id))

      for (k in seq_len(nrow(node$replies))) {
        rep_id <- safe_id(paste0("S", j, k, node$replies$san[k]))
        nodes  <- c(nodes, sprintf(
          '  %s [label="%s %+.3f", style=filled, fillcolor="#C8E6C9", shape=ellipse]',
          rep_id, node$replies$san[k], node$replies$score[k]))
        edges  <- c(edges, sprintf("  %s -> %s", opp_id, rep_id))
      }
    }

    paste0(
      'digraph G {\n',
      '  graph [rankdir=TB, bgcolor=transparent, splines=ortho]\n',
      '  node [fontsize=11, fontname=Helvetica]\n',
      paste(nodes, collapse = "\n"), "\n",
      paste(edges, collapse = "\n"), "\n}"
    )
  }

  # ── Search tree output ───────────────────────────────────────────────────────

  output$search_tree_diagram <- renderGrViz({
    req(nrow(analysis_scores()) > 0, nzchar(input$tree_candidate %||% ""))
    board  <- chess_mod$Board(analysis_fen())
    tree   <- build_search_tree(board, analysis_weights(),
                                input$tree_candidate, analysis_played_san())
    grViz(build_graphviz(tree))
  })
}

shinyApp(ui, server)
