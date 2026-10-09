library(DBI)
library(RSQLite)

con <- dbConnect(SQLite(), "data/chess_learning.db")
#dbDisconnect(con)

# Helpers ----

tranche_filter <- function(tbl, start = GAME_START, end = GAME_END) {
    if (!is.null(start)) tbl <- filter(tbl, game_number >= start)
    if (!is.null(end))   tbl <- filter(tbl, game_number <= end)
    tbl
}

# Survey ----

runs <- tbl(con, "runs") |> collect() |> print()

games <- tbl(con, "games") |>
    group_by(run_id) |>
    summarise(
        n_games    = n(),
        game_start = min(game_number),
        game_end   = max(game_number),
        .groups    = "drop"
    ) |>
    collect() |>
    print()

# Parameters ----

RUN_ID     <- 16
GAME_START <- NULL   # set to integer to restrict to a tranche (NULL = all)
GAME_END   <- NULL

# Outcomes ----

outcomes <- tbl(con, "games") |>
    filter(run_id == RUN_ID) |>
    tranche_filter() |>
    select(game_number, outcome, n_halfmoves, duration_s, terminated_by) |>
    collect() |>
    mutate(
        engine_color  = if_else(game_number %% 2 == 1, "White", "Black"),
        engine_result = case_when(
            outcome == "1-0" & engine_color == "White" ~ "Win",
            outcome == "0-1" & engine_color == "Black" ~ "Win",
            outcome == "1-0" & engine_color == "Black" ~ "Loss",
            outcome == "0-1" & engine_color == "White" ~ "Loss",
            TRUE ~ "Draw"
        )
    )

# Per-game view
outcomes |>
    select(game_number, engine_color, outcome, engine_result, n_halfmoves, terminated_by)

# Summary from engine's perspective
outcomes |>
    count(engine_result) |>
    mutate(pct = round(n / sum(n) * 100, 1)) |>
    arrange(factor(engine_result, levels = c("Win", "Draw", "Loss")))

# Weights ----

weights <- tbl(con, "weights") |>
    filter(run_id == RUN_ID) |>
    tranche_filter() |>
    collect() |>
    group_by(feature_name) |>
    slice_max(game_number, n = 1, with_ties = FALSE) |>
    ungroup() |>
    arrange(desc(weight_value))

weights |> select(feature_name, weight_value)

# Trajectory plot ----

wts_trj <- tbl(con, "weights") |>
    filter(run_id == RUN_ID) |>
    tranche_filter() |>
    collect()

ggplot(wts_trj, aes(x = game_number, y = weight_value)) +
    geom_line(color = "dodgerblue2") +
    stat_smooth(method = "lm") +
    facet_wrap(~ feature_name, scales = "free_y") +
    labs(title = paste("Weight trajectories — run", RUN_ID),
         x = "Game", y = "Weight") +
    faceted


# which features drive deltas ---- 

tbl(con, "weight_deltas") |>
    filter(run_id == RUN_ID) |>
    collect() |>
    left_join(
        tbl(con, "games") |> filter(run_id == 13) |>
            select(game_number, outcome) |> collect(),
        by = "game_number"
    ) |>
    filter(outcome == "1/2-1/2") |>
    group_by(feature_name) |>
    summarise(mean_abs_delta = mean(abs(delta)), .groups = "drop") |>
    arrange(desc(mean_abs_delta))

