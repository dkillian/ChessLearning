"""
Add three AlphaZero-inspired improvements to both v2 notebooks:
  1. Step-decay LR schedule (MultiStepLR)
  2. Temperature-based move selection for first N moves  [SF teacher only]
  3. Resignation when engine value < -threshold for K consecutive moves
"""
import json, copy

def src(cell):
    return ''.join(cell['source'])

# ══════════════════════════════════════════════════════════════════════════════
# selfplay_sf_nn_v2.ipynb
# ══════════════════════════════════════════════════════════════════════════════
with open('selfplay_sf_nn_v2.ipynb', encoding='utf-8') as f:
    nb_sf = json.load(f)

for i, cell in enumerate(nb_sf['cells']):
    s = src(cell)

    # ── 1. Header ──────────────────────────────────────────────────────────────
    if 'Stage 02 (Stockfish Teacher, Neural Net, v2)' in s:
        s = s.replace(
            '**v2 change:** Replaced `torch.optim.Adam` with `torch.optim.AdamW`'
            ' (adds L2 weight decay `c||θ||²`, as in AlphaZero loss).'
            ' Stabilises small-magnitude positional feature weights that drifted under plain Adam.',
            ('**v2 changes (AlphaZero-inspired):**\n'
             '1. `torch.optim.AdamW` — L2 weight decay `c||θ||²` stabilises small-weight features\n'
             '2. Step-decay LR schedule — LR drops 10× at configurable game milestones\n'
             '3. Temperature move selection — softmax over engine moves for first N full moves (training diversity)\n'
             '4. Resignation — engine resigns when value ≤ −threshold for K consecutive moves (training throughput)')
        )
        nb_sf['cells'][i]['source'] = [s]

    # ── 2. Game function: replace with temperature + resignation version ────────
    if 'def play_game_vs_sf_nn(sf, model, depth, engine_color):' in s:
        nb_sf['cells'][i]['source'] = [
            "def temperature_move(board, model, temperature=1.0):\n"
            "    \"\"\"1-ply softmax move selection for training diversity.\"\"\"\n"
            "    import numpy as np\n"
            "    moves = list(board.legal_moves)\n"
            "    if not moves:\n"
            "        return None\n"
            "    if len(moves) == 1:\n"
            "        return moves[0]\n"
            "    scores = []\n"
            "    for move in moves:\n"
            "        board.push(move)\n"
            "        feat = extract_features(board)\n"
            "        x = torch.tensor([feat[n] for n in FEATURE_NAMES], dtype=torch.float32)\n"
            "        with torch.no_grad():\n"
            "            v = -model(x).item()  # negate: value from current player's perspective\n"
            "        board.pop()\n"
            "        scores.append(v)\n"
            "    probs = torch.softmax(torch.tensor(scores, dtype=torch.float32) / temperature, dim=0).numpy()\n"
            "    return moves[np.random.choice(len(moves), p=probs)]\n"
            "\n"
            "\n"
            "def play_game_vs_sf_nn(\n"
            "    sf, model, depth, engine_color,\n"
            "    temperature=1.0, temp_moves=15,\n"
            "    resign_threshold=0.9, resign_consecutive=4,\n"
            "):\n"
            "    \"\"\"\n"
            "    Play one game: NN engine (engine_color) vs Stockfish.\n"
            "\n"
            "    temperature      : softmax temperature for engine's first temp_moves full moves.\n"
            "                       Set to 0 to disable (always greedy).\n"
            "    temp_moves       : number of full moves to use temperature; thereafter greedy.\n"
            "    resign_threshold : engine resigns when its value < -resign_threshold\n"
            "                       for resign_consecutive consecutive engine half-moves.\n"
            "\n"
            "    Returns: positions, outcome, terminated_by, pgn_str\n"
            "    positions : list of (feature_dict, color) 2-tuples for every half-move.\n"
            "    \"\"\"\n"
            "    board            = chess.Board()\n"
            "    game             = chess.pgn.Game()\n"
            "    node             = game\n"
            "    positions        = []\n"
            "    terminated_by    = \"natural\"\n"
            "    consecutive_bad  = 0\n"
            "\n"
            "    while True:\n"
            "        if board.is_game_over(claim_draw=True):\n"
            "            break\n"
            "\n"
            "        features = extract_features(board)\n"
            "        color    = board.turn\n"
            "        ply      = len(positions)\n"
            "\n"
            "        if board.turn == engine_color:\n"
            "            x = torch.tensor(\n"
            "                [features[n] for n in FEATURE_NAMES], dtype=torch.float32\n"
            "            )\n"
            "            with torch.no_grad():\n"
            "                engine_value = model(x).item()\n"
            "\n"
            "            # Resignation check (AlphaZero: resign at -0.9 for 4 consecutive moves)\n"
            "            if engine_value < -resign_threshold:\n"
            "                consecutive_bad += 1\n"
            "                if consecutive_bad >= resign_consecutive:\n"
            "                    terminated_by = \"resignation\"\n"
            "                    break\n"
            "            else:\n"
            "                consecutive_bad = 0\n"
            "\n"
            "            # Move selection: temperature for early game, greedy thereafter\n"
            "            if temperature > 0 and ply < temp_moves * 2:\n"
            "                move = temperature_move(board, model, temperature)\n"
            "            else:\n"
            "                move, _ = get_best_move(board, model, depth=depth, randomize=False)\n"
            "        else:\n"
            "            result = sf.play(board, chess.engine.Limit(time=0.5), ponder=False)\n"
            "            move   = result.move\n"
            "            consecutive_bad = 0  # reset: only track engine's consecutive bad moves\n"
            "\n"
            "        if move is None:\n"
            "            break\n"
            "\n"
            "        node = node.add_variation(move)\n"
            "        board.push(move)\n"
            "        positions.append((features, color))\n"
            "\n"
            "    if terminated_by == \"resignation\":\n"
            "        outcome = \"0-1\" if engine_color == chess.WHITE else \"1-0\"\n"
            "    else:\n"
            "        outcome = board.result(claim_draw=True)\n"
            "\n"
            "    game.headers[\"Result\"] = outcome\n"
            "    return positions, outcome, terminated_by, str(game)"
        ]

    # ── 3. Training function: add LR schedule + new params ────────────────────
    if 'def train_vs_sf_nn(' in s and 'AdamW' in s:
        s = s.replace(
            '    weight_decay     = 1e-4,\n    log_positions    = False,',
            (
                '    weight_decay       = 1e-4,\n'
                '    lr_milestones      = None,     # game numbers to drop LR; default [200, 500]\n'
                '    lr_gamma           = 0.1,      # LR multiplier at each milestone\n'
                '    temperature        = 1.0,      # softmax temp for engine early-game moves (0=off)\n'
                '    temp_moves         = 15,       # full moves to use temperature (~first 30 plies)\n'
                '    resign_threshold   = 0.9,\n'
                '    resign_consecutive = 4,\n'
                '    log_positions      = False,'
            )
        )
        s = s.replace(
            '    conn = setup_database(db_path)\n    os.makedirs(model_dir, exist_ok=True)',
            (
                '    if lr_milestones is None:\n'
                '        lr_milestones = [200, 500]\n'
                '    conn = setup_database(db_path)\n    os.makedirs(model_dir, exist_ok=True)'
            )
        )
        s = s.replace(
            '    model.eval()   # eval mode during play; Adam still updates parameters\n'
            '    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)',
            (
                '    model.eval()   # eval mode during play; Adam still updates parameters\n'
                '    optimizer = torch.optim.AdamW(\n'
                '        model.parameters(), lr=learning_rate, weight_decay=weight_decay\n'
                '    )\n'
                '    # Step-decay LR schedule (AlphaZero: drop 10x at fixed training milestones)\n'
                '    scheduler = torch.optim.lr_scheduler.MultiStepLR(\n'
                '        optimizer, milestones=lr_milestones, gamma=lr_gamma,\n'
                '        last_epoch=start_game - 1,  # fast-forwards when resuming\n'
                '    )'
            )
        )
        s = s.replace(
            '            positions, outcome, terminated_by, pgn = play_game_vs_sf_nn(\n'
            '                sf, model, depth, engine_color\n'
            '            )',
            (
                '            positions, outcome, terminated_by, pgn = play_game_vs_sf_nn(\n'
                '                sf, model, depth, engine_color,\n'
                '                temperature=temperature, temp_moves=temp_moves,\n'
                '                resign_threshold=resign_threshold,\n'
                '                resign_consecutive=resign_consecutive,\n'
                '            )'
            )
        )
        s = s.replace(
            '            outcomes[outcome] = outcomes.get(outcome, 0) + 1\n'
            '\n'
            '            if game_num % checkpoint_every == 0:',
            (
                '            outcomes[outcome] = outcomes.get(outcome, 0) + 1\n'
                '            scheduler.step()  # one LR step per game\n'
                '\n'
                '            if game_num % checkpoint_every == 0:'
            )
        )
        s = s.replace(
            '                    print(\n'
            '                        f"Game {game_num:5d} ({color_label}) | {outcome:7s} | "\n'
            '                        f"{len(positions):3d} half-moves | {elapsed:.1f}s | "\n'
            '                        f"td_err={td_error:.3f} | decisive={dec}/{total} | {rate:.0f} games/hr"\n'
            '                    )',
            (
                '                    lr_now = scheduler.get_last_lr()[0]\n'
                '                    print(\n'
                '                        f"Game {game_num:5d} ({color_label}) | {outcome:7s} | "\n'
                '                        f"{len(positions):3d} half-moves | {elapsed:.1f}s | "\n'
                '                        f"td_err={td_error:.3f} | lr={lr_now:.2e} | "\n'
                '                        f"decisive={dec}/{total} | {rate:.0f} games/hr"\n'
                '                    )'
            )
        )
        nb_sf['cells'][i]['source'] = [s]

    # ── 4. Config cell ─────────────────────────────────────────────────────────
    if 'RUN_ID           = 28' in s and 'WEIGHT_DECAY' in s:
        s = s.replace(
            'WEIGHT_DECAY       = 1e-4       # AdamW L2 regularisation (AlphaZero: c||theta||^2)\n',
            (
                'WEIGHT_DECAY       = 1e-4       # AdamW L2 regularisation (AlphaZero: c||theta||^2)\n'
                'LR_MILESTONES      = [200, 500] # games at which LR drops by LR_GAMMA\n'
                'LR_GAMMA           = 0.1        # 0.001 -> 0.0001 -> 0.00001\n'
                'TEMPERATURE        = 1.0        # softmax temp for engine early moves (0 = greedy always)\n'
                'TEMP_MOVES         = 15         # full moves (~30 plies) to use temperature\n'
                'RESIGN_THRESHOLD   = 0.9        # resign when engine value < -this\n'
                'RESIGN_CONSECUTIVE = 4          # consecutive half-moves below threshold to resign\n'
            )
        )
        s = s.replace(
            '    weight_decay       = WEIGHT_DECAY,\n',
            (
                '    weight_decay       = WEIGHT_DECAY,\n'
                '    lr_milestones      = LR_MILESTONES,\n'
                '    lr_gamma           = LR_GAMMA,\n'
                '    temperature        = TEMPERATURE,\n'
                '    temp_moves         = TEMP_MOVES,\n'
                '    resign_threshold   = RESIGN_THRESHOLD,\n'
                '    resign_consecutive = RESIGN_CONSECUTIVE,\n'
            )
        )
        nb_sf['cells'][i]['source'] = [s]

with open('selfplay_sf_nn_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_sf, f, indent=1, ensure_ascii=False)
print("selfplay_sf_nn_v2.ipynb: done")

# ══════════════════════════════════════════════════════════════════════════════
# selfplay_nn_v2.ipynb  (LR schedule + resignation; epsilon-greedy kept for diversity)
# ══════════════════════════════════════════════════════════════════════════════
with open('selfplay_nn_v2.ipynb', encoding='utf-8') as f:
    nb_nn = json.load(f)

for i, cell in enumerate(nb_nn['cells']):
    s = src(cell)

    # ── 1. play_game: add resignation ─────────────────────────────────────────
    if 'def play_game(' in s and 'epsilon' in s and 'resign' not in s:
        s = s.replace(
            'def play_game(model, depth, move_limit=200, epsilon=0.1, opponent_model=None):',
            (
                'def play_game(\n'
                '    model, depth, move_limit=200, epsilon=0.1, opponent_model=None,\n'
                '    resign_threshold=0.9, resign_consecutive=4,\n'
                '):'
            )
        )
        # Add consecutive_bad tracker after board/positions init
        s = s.replace(
            '    terminated_by = "natural"',
            '    terminated_by    = "natural"\n    consecutive_bad  = 0'
        )
        # Insert resignation check before epsilon-greedy block
        old_eps = '        if is_learner:\n            if random.random() < epsilon:'
        new_eps = (
            '        if is_learner:\n'
            '            # Resignation check\n'
            '            x_r = torch.tensor([features[n] for n in FEATURE_NAMES], dtype=torch.float32)\n'
            '            with torch.no_grad():\n'
            '                lv = model(x_r).item()\n'
            '            if lv < -resign_threshold:\n'
            '                consecutive_bad += 1\n'
            '                if consecutive_bad >= resign_consecutive:\n'
            '                    terminated_by = "resignation"\n'
            '                    break\n'
            '            else:\n'
            '                consecutive_bad = 0\n'
            '            if random.random() < epsilon:'
        )
        if old_eps in s:
            s = s.replace(old_eps, new_eps)
        # Add resignation outcome handling
        s = s.replace(
            '    if terminated_by == "move_limit":\n'
            '        outcome = "1/2-1/2"\n'
            '    else:\n'
            '        outcome = board.result(claim_draw=True)',
            (
                '    if terminated_by == "resignation":\n'
                '        learner_color = chess.WHITE\n'
                '        for _, c, is_l in positions:\n'
                '            if is_l:\n'
                '                learner_color = c\n'
                '                break\n'
                '        outcome = "0-1" if learner_color == chess.WHITE else "1-0"\n'
                '    elif terminated_by == "move_limit":\n'
                '        outcome = "1/2-1/2"\n'
                '    else:\n'
                '        outcome = board.result(claim_draw=True)'
            )
        )
        nb_nn['cells'][i]['source'] = [s]

    # ── 2. train(): add LR schedule + resign params ───────────────────────────
    if 'def train(' in s and 'AdamW' in s and 'resign' not in s:
        s = s.replace(
            '    weight_decay              = 1e-4,\n    hidden_size',
            (
                '    weight_decay              = 1e-4,\n'
                '    lr_milestones             = None,   # game numbers to drop LR; default [200, 500]\n'
                '    lr_gamma                  = 0.1,    # LR multiplier at each milestone\n'
                '    resign_threshold          = 0.9,\n'
                '    resign_consecutive        = 4,\n'
                '    hidden_size'
            )
        )
        s = s.replace(
            '    conn = setup_database(db_path)\n\n    if resume:',
            (
                '    if lr_milestones is None:\n'
                '        lr_milestones = [200, 500]\n'
                '    conn = setup_database(db_path)\n\n    if resume:'
            )
        )
        s = s.replace(
            '    model.train()\n    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)',
            (
                '    model.train()\n'
                '    optimizer = torch.optim.AdamW(\n'
                '        model.parameters(), lr=learning_rate, weight_decay=weight_decay\n'
                '    )\n'
                '    scheduler = torch.optim.lr_scheduler.MultiStepLR(\n'
                '        optimizer, milestones=lr_milestones, gamma=lr_gamma,\n'
                '        last_epoch=start_game - 1,\n'
                '    )'
            )
        )
        # Pass resign params to play_game
        s = s.replace(
            '            opponent_model = opponent_model,\n        )',
            (
                '            opponent_model     = opponent_model,\n'
                '            resign_threshold   = resign_threshold,\n'
                '            resign_consecutive = resign_consecutive,\n'
                '        )'
            )
        )
        # Add scheduler.step() per game
        s = s.replace(
            '        outcomes[outcome] += 1\n',
            '        outcomes[outcome] += 1\n        scheduler.step()\n'
        )
        # Add lr_now to verbose print
        s = s.replace(
            '            if verbose:\n                print(f"Game {game_num:4d}',
            (
                '            if verbose:\n'
                '                lr_now = scheduler.get_last_lr()[0]\n'
                '                print(f"Game {game_num:4d}'
            )
        )
        s = s.replace(
            'decisive={100*dec_rate:.1f}% | {rate:.0f} games/hr")',
            'decisive={100*dec_rate:.1f}% | lr={lr_now:.2e} | {rate:.0f} games/hr")'
        )
        nb_nn['cells'][i]['source'] = [s]

    # ── 3. Config cell ─────────────────────────────────────────────────────────
    if 'WEIGHT_DECAY' in s and 'LEARNING_RATE' in s and 'RUN_ID' in s and 'LR_MILESTONES' not in s:
        s = s.replace(
            'WEIGHT_DECAY     = 1e-4         # AdamW L2 regularisation\n',
            (
                'WEIGHT_DECAY          = 1e-4         # AdamW L2 regularisation\n'
                'LR_MILESTONES         = [200, 500]   # games at which LR drops\n'
                'LR_GAMMA              = 0.1           # LR multiplier (0.001->0.0001->0.00001)\n'
                'RESIGN_THRESHOLD      = 0.9\n'
                'RESIGN_CONSECUTIVE    = 4\n'
            )
        )
        s = s.replace(
            '    weight_decay              = WEIGHT_DECAY,\n',
            (
                '    weight_decay              = WEIGHT_DECAY,\n'
                '    lr_milestones             = LR_MILESTONES,\n'
                '    lr_gamma                  = LR_GAMMA,\n'
                '    resign_threshold          = RESIGN_THRESHOLD,\n'
                '    resign_consecutive        = RESIGN_CONSECUTIVE,\n'
            )
        )
        nb_nn['cells'][i]['source'] = [s]

with open('selfplay_nn_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb_nn, f, indent=1, ensure_ascii=False)
print("selfplay_nn_v2.ipynb: done")
