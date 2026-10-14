# seed_supabase.R
# One-time script: seeds Supabase with the two new engine conditions.
#
# Run once after deploying the updated public_app/app.R.
# Safe to re-run — uses INSERT OR IGNORE / ON CONFLICT DO NOTHING.
#
# Prerequisites:
#   - Sys.getenv("SUPABASE_PASSWORD") must be set
#   - local chess_learning.db must be accessible
#   - models/nn_run_24_game_100.pt (or latest available) must exist
#
# What this seeds:
#   run_id=1  hardcoded material  — already in Supabase, not touched here
#   run_id=2  learnable material  — seeded from local run_id=21, game 1050
#   run_id=3  neural net (MLP)   — seeded from local run_id=24, latest .pt

library(DBI)
library(RSQLite)
library(RPostgres)
library(reticulate)
library(tidyverse)

# ---- Paths ---------------------------------------------------------------

DB_PATH   <- if (file.exists("../../data/chess_learning.db"))
    "../../data/chess_learning.db" else "data/chess_learning.db"

MODEL_DIR <- if (file.exists("../../models")) "../../models" else "models"

ENGINE_PATH <- if (file.exists("../../scripts/python"))
    "../../scripts/python" else "scripts/python"

# Source run IDs in local SQLite
LOCAL_LEARNABLE_RUN <- 21L   # learnable material — 17/19 correct signs at game 1050
LOCAL_NN_RUN        <- 24L   # neural net — game 100

# Public run IDs in Supabase
PUBLIC_LEARNABLE_RUN <- 2L
PUBLIC_NN_RUN        <- 3L

# ---- Connections ---------------------------------------------------------

local_con <- dbConnect(SQLite(), DB_PATH)

pg <- dbConnect(
    Postgres(),
    host     = "db.betniftrhjpeacfzjoym.supabase.co",
    port     = 5432L,
    dbname   = "postgres",
    user     = "postgres",
    password = Sys.getenv("SUPABASE_PASSWORD"),
    sslmode  = "require"
)

cat("Connected to local DB and Supabase.\n")

# ---- Step 1: Create nn_models table if it doesn't exist -----------------

cat("\n[1/5] Creating nn_models table in Supabase...\n")

dbExecute(pg, "
    CREATE TABLE IF NOT EXISTS nn_models (
        id          SERIAL PRIMARY KEY,
        run_id      INTEGER NOT NULL,
        game_number INTEGER NOT NULL,
        model_b64   TEXT    NOT NULL,
        updated_at  TIMESTAMPTZ DEFAULT NOW()
    );
")
dbExecute(pg, "
    CREATE INDEX IF NOT EXISTS nn_models_run_game
    ON nn_models (run_id, game_number);
")
cat("   nn_models table ready.\n")

# ---- Step 2: Seed run_id=2 (learnable material) -------------------------

cat("\n[2/5] Seeding run_id=2 (learnable material) in Supabase runs table...\n")

local_run2 <- dbGetQuery(local_con,
    paste0("SELECT * FROM runs WHERE run_id = ", LOCAL_LEARNABLE_RUN))

dbExecute(pg, "
    INSERT INTO runs (run_id, started_at, depth, learning_rate, move_limit, notes)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (run_id) DO NOTHING",
    list(PUBLIC_LEARNABLE_RUN,
         format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
         local_run2$depth,
         local_run2$learning_rate,
         local_run2$move_limit,
         "Learnable material — seeded from local run_id=21 game 1050; 19 features; 17/19 correct signs"))

cat("   Run row created (or already existed).\n")

# ---- Step 3: Seed learnable material weights ----------------------------

cat("\n[3/5] Seeding weights for run_id=2 (learnable material)...\n")

# Get the latest checkpoint from the local learnable material run
local_ckpt2 <- dbGetQuery(local_con,
    paste0("SELECT MAX(game_number) as gn FROM weights WHERE run_id = ", LOCAL_LEARNABLE_RUN))$gn

local_weights2 <- dbGetQuery(local_con,
    paste0("SELECT feature_name, weight_value FROM weights
            WHERE run_id = ", LOCAL_LEARNABLE_RUN, "
              AND game_number = ", local_ckpt2))

cat(sprintf("   Seeding %d features from local run_id=%d game %d.\n",
            nrow(local_weights2), LOCAL_LEARNABLE_RUN, local_ckpt2))

# Check if weights already exist for this public run
existing2 <- dbGetQuery(pg,
    "SELECT COUNT(*) as n FROM weights WHERE run_id = $1", list(PUBLIC_LEARNABLE_RUN))$n

if (existing2 == 0) {
    for (i in seq_len(nrow(local_weights2))) {
        dbExecute(pg,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES ($1, $2, $3, $4)",
            list(PUBLIC_LEARNABLE_RUN, 0L,
                 local_weights2$feature_name[i],
                 local_weights2$weight_value[i]))
    }
    cat(sprintf("   Inserted %d weight rows.\n", nrow(local_weights2)))
} else {
    cat(sprintf("   Weights already exist (%d rows) — skipping.\n", existing2))
}

# ---- Step 4: Seed run_id=3 (neural net) ---------------------------------

cat("\n[4/5] Seeding run_id=3 (neural net) in Supabase runs table...\n")

dbExecute(pg, "
    INSERT INTO runs (run_id, started_at, depth, learning_rate, move_limit, notes)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (run_id) DO NOTHING",
    list(PUBLIC_NN_RUN,
         format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
         3L, 0.001, 200L,
         "Neural net (MLP 19->32->1) — seeded from local run_id=24; Adam optimizer"))

cat("   Run row created (or already existed).\n")

# ---- Step 5: Seed NN model ----------------------------------------------

cat("\n[5/5] Serializing and seeding NN model for run_id=3...\n")

existing_model <- dbGetQuery(pg,
    "SELECT COUNT(*) as n FROM nn_models WHERE run_id = $1", list(PUBLIC_NN_RUN))$n

if (existing_model == 0) {
    # Find the latest available .pt checkpoint for the NN run
    pt_files <- list.files(MODEL_DIR,
        pattern  = paste0("nn_run_", LOCAL_NN_RUN, "_game_\\d+\\.pt"),
        full.names = TRUE)

    if (length(pt_files) == 0) {
        stop("No .pt checkpoint found for run_id=", LOCAL_NN_RUN,
             " in ", MODEL_DIR, ". Run selfplay_nn.ipynb first.")
    }

    # Pick the highest game-number checkpoint
    game_nums  <- as.integer(gsub(".*_game_(\\d+)\\.pt", "\\1", basename(pt_files)))
    latest_pt  <- pt_files[which.max(game_nums)]
    latest_game <- max(game_nums)

    cat(sprintf("   Loading %s (game %d)...\n", basename(latest_pt), latest_game))

    # Load Python engine and serialize model to base64
    # Force module reload in case a cached version is missing model_to_b64
    reticulate::py_run_string(paste0(
        "import importlib, sys; sys.path.insert(0, r'",
        normalizePath(ENGINE_PATH), "');\n",
        "importlib.reload(sys.modules['engine_nn']) if 'engine_nn' in sys.modules else None"
    ))
    nn_mod <- reticulate::import_from_path("engine_nn", path = ENGINE_PATH)
    model  <- nn_mod$load_model(latest_pt)
    b64    <- nn_mod$model_to_b64(model)

    cat(sprintf("   Model serialized: %d base64 characters.\n", nchar(b64)))

    # Also seed attribution weights for R analysis compatibility
    attr_w <- nn_mod$attribution_weights(model)
    attr_r <- setNames(as.numeric(unlist(attr_w)), names(attr_w))

    # Insert model blob
    dbExecute(pg,
        "INSERT INTO nn_models (run_id, game_number, model_b64, updated_at)
         VALUES ($1, $2, $3, NOW())",
        list(PUBLIC_NN_RUN, latest_game, b64))

    # Insert attribution weights (for R analysis scripts)
    for (fname in names(attr_r)) {
        dbExecute(pg,
            "INSERT INTO weights (run_id, game_number, feature_name, weight_value)
             VALUES ($1, $2, $3, $4)
             ON CONFLICT DO NOTHING",
            list(PUBLIC_NN_RUN, 0L, fname, attr_r[[fname]]))
    }

    cat(sprintf("   NN model (game %d) seeded to Supabase.\n", latest_game))
    cat(sprintf("   Attribution weights inserted: %d features.\n", length(attr_r)))

} else {
    cat(sprintf("   NN model already exists (%d row(s)) — skipping.\n", existing_model))
}

# ---- Done ---------------------------------------------------------------

cat("\n── Seeding complete ─────────────────────────────────────────────────\n")
cat("Supabase public run summary:\n")
print(dbGetQuery(pg, "SELECT run_id, notes FROM runs WHERE run_id IN (1, 2, 3) ORDER BY run_id"))
cat("\nWeight counts per run:\n")
print(dbGetQuery(pg,
    "SELECT run_id, COUNT(*) as n_weight_rows, MAX(game_number) as latest_game
     FROM weights WHERE run_id IN (1,2,3) GROUP BY run_id ORDER BY run_id"))
cat("\nNN model rows:\n")
print(dbGetQuery(pg,
    "SELECT run_id, game_number, LENGTH(model_b64) as b64_chars, updated_at
     FROM nn_models ORDER BY run_id, game_number"))

dbDisconnect(local_con)
dbDisconnect(pg)
cat("\nConnections closed. Done.\n")
