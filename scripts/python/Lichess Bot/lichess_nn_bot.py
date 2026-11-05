"""
lichess_nn_bot.py -- Lichess Bot API client for the neural net engine (engine_nn.py).

Plays Lichess games with the current network and learns from each finished game
with a TD(0) + Adam update (engine_nn.nn_td_update). Each game is logged to
data/chess_learning.db under a new run_id, and a checkpoint is saved after each game.

Requires LICHESS_BOT_TOKEN (token with bot:play scope). Run from a terminal;
Ctrl+C to stop. The Adam optimizer state is not persisted across restarts; to
resume, point SEED_CHECKPOINT / SEED_RUN_ID / SEED_GAME_NUMBER at the latest checkpoint.
"""

import json
import os
import random
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TD_DIR = PROJECT_ROOT / "scripts" / "python" / "Temporal Difference"
sys.path.insert(0, str(TD_DIR))

import engine_nn  # noqa: E402
import lichess_chat  # noqa: E402
from engine_learnable_material import FEATURE_NAMES, extract_features  # noqa: E402

API = "https://lichess.org"
TOKEN = os.environ.get("LICHESS_BOT_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

DB_PATH = PROJECT_ROOT / "data" / "chess_learning.db"
MODEL_DIR = PROJECT_ROOT / "models"
SEED_CHECKPOINT = MODEL_DIR / "nn_sf_run_27_game_390.pt"
SEED_RUN_ID = 27
SEED_GAME_NUMBER = 390

MAX_DEPTH = 3
LEARNING_RATE = 0.001
DRAW_LR_SCALE = 0.1
ALLOWED_SPEEDS = {"blitz", "rapid", "classical"}  # engine is too slow for bullet
ACCEPT_RATED = True
ACCEPT_BOT_CHALLENGES = True
ACCEPT_HUMAN_CHALLENGES = True
MAX_CONCURRENT_GAMES = 2
GAMES_RESERVED_FOR_HUMANS = 1  # bot games may only use MAX_CONCURRENT_GAMES minus this
BOT_DAILY_LIMIT = 100  # Lichess cap on games against other bots per day
MATCHMAKING_ENABLED = True
MATCHMAKING_IDLE_S = 300  # pause between matchmaking attempts while idle
STREAM_RETRY_MIN_S = 5  # reconnect delay after an ordinary stream error
RATE_LIMIT_BACKOFF_S = 60  # first reconnect delay after an HTTP 429
STREAM_RETRY_MAX_S = 600  # cap for the doubling 429 backoff
BOT_CHALLENGE_WAIT_S = 60
OPEN_CHALLENGE_WAIT_S = 900
CHALLENGE_RATED = True
RATING_WINDOW = 300
# (initial seconds, increment seconds, speed) chosen at random for outgoing challenges
TIME_CONTROLS = [(180, 2, "blitz"), (300, 3, "blitz"), (600, 0, "rapid")]
ENGINE_VERSION = "lichess_nn_v1"

_lock = threading.Lock()  # guards the shared state below
_active_games: set[str] = set()
_started_count = 0  # total gameStart events; detects when an outgoing challenge is accepted
_pending_accepts = 0  # challenges accepted but whose gameStart has not arrived yet
_closed_ids: set[str] = set()  # challenges declined or canceled
_skip_bots: set[str] = set()  # bots that declined a matchmaking challenge
_daily = {"date": None, "bot_games": 0}
BOT_ID = ""
BOT_USERNAME = ""
MY_PERFS: dict = {}

CHAT_ENABLED = True
CHAT_COOLDOWN_S = 15  # minimum gap between chat messages the bot sends
CHAT_QUEUE_MAX = 3  # drop chat jobs beyond this many waiting
_chat_executor = ThreadPoolExecutor(max_workers=1)  # one LLM call and post at a time
_chat_pending = 0
_last_chat_sent = 0.0
_last_learning: dict | None = None  # summary of the most recent logged game


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def choose_depth(remaining_ms: int) -> int:
    # Back off search depth when the clock is low
    if remaining_ms < 10_000:
        return 1
    if remaining_ms < 30_000:
        return 2
    return MAX_DEPTH


def ndjson(response):
    for line in response.iter_lines():
        if line:
            yield json.loads(line)


def start_board(initial_fen: str) -> chess.Board:
    return chess.Board() if initial_fen in (None, "", "startpos") else chess.Board(initial_fen)


def board_from_moves(initial_fen: str, moves: str) -> chess.Board:
    board = start_board(initial_fen)
    for uci in moves.split():
        board.push_uci(uci)
    return board


def build_pgn(game_id, initial_fen, moves, white, black, outcome, status, opponent_is_bot) -> str:
    game = chess.pgn.Game.from_board(start_board(initial_fen))
    game.headers["Event"] = "Lichess bot game"
    game.headers["Site"] = f"{API}/{game_id}"
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = outcome
    game.headers["Termination"] = status
    game.headers["Opponent"] = "bot" if opponent_is_bot else "human"
    node = game
    for uci in moves:
        node = node.add_variation(chess.Move.from_uci(uci))
    return str(game)


class Learner:
    """Network, optimizer, and DB connection shared by all game threads."""

    def __init__(self, checkpoint: Path, seed_run_id: int, seed_game_number: int):
        self.model = engine_nn.load_model(str(checkpoint))
        self.optimizer = engine_nn.make_optimizer(self.model, lr=LEARNING_RATE)
        self.model_lock = threading.Lock()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.seed_run_id = seed_run_id
        self.seed_game_number = seed_game_number
        # Checkpoint the next game will be played against
        self.current_ckpt = (seed_run_id, seed_game_number)
        self.run_id = self._create_run()
        self.game_number = 0

    def _create_run(self) -> int:
        run_id = self.conn.execute("SELECT COALESCE(MAX(run_id), 0) + 1 FROM runs").fetchone()[0]
        now = utc_now()
        notes = (f"Lichess bot games; seeded from run_id={self.seed_run_id} "
                 f"game={self.seed_game_number}; TD(0)+Adam online learning; "
                 f"draw_lr_scale={DRAW_LR_SCALE}")
        self.conn.execute(
            "INSERT INTO runs (run_id, started_at, depth, learning_rate, move_limit, "
            "notes, last_updated, engine_version) VALUES (?,?,?,?,?,?,?,?)",
            (run_id, now, MAX_DEPTH, LEARNING_RATE, None, notes, now, ENGINE_VERSION),
        )
        self.conn.commit()
        return run_id

    def choose_move(self, board: chess.Board, depth: int):
        with self.model_lock:
            return engine_nn.get_best_move(board, self.model, depth=depth)

    def record_game(self, game_id, bot_color, initial_fen, moves, status, winner,
                    white_name, black_name, duration_s, opponent_is_bot) -> None:
        if status in ("aborted", "noStart") or not moves:
            print(f"[game {game_id}] not logged (status={status}, moves={len(moves)})")
            return

        outcome = {"white": "1-0", "black": "0-1"}.get(winner, "1/2-1/2")

        # Replay the game to collect features before each half-move
        board = start_board(initial_fen)
        feat_dicts, colors = [], []
        for uci in moves:
            feat_dicts.append(extract_features(board))
            colors.append(board.turn)
            board.push_uci(uci)

        pgn = build_pgn(game_id, initial_fen, moves, white_name, black_name, outcome, status,
                        opponent_is_bot)

        with self.model_lock:
            result = engine_nn.nn_td_update(
                self.model, self.optimizer, feat_dicts, colors, outcome, DRAW_LR_SCALE
            )
            self.game_number += 1
            gn = self.game_number
            played_ckpt = self.current_ckpt
            opp_color = "black" if bot_color == chess.WHITE else "white"

            cur = self.conn.cursor()
            cur.execute(
                "INSERT INTO games (run_id, game_number, outcome, n_halfmoves, duration_s, "
                "terminated_by, human_color, seed_run_id, seed_game_number) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (self.run_id, gn, outcome, len(moves), round(duration_s, 2), status,
                 opp_color, played_ckpt[0], played_ckpt[1]),
            )
            cur.execute(
                "INSERT INTO game_records (run_id, game_number, pgn, td_error) VALUES (?,?,?,?)",
                (self.run_id, gn, pgn, float(result["td_error"])),
            )
            for name in FEATURE_NAMES:
                cur.execute(
                    "INSERT INTO weight_deltas (run_id, game_number, feature_name, delta) "
                    "VALUES (?,?,?,?)",
                    (self.run_id, gn, name, float(result["deltas"][name])),
                )
                cur.execute(
                    "INSERT INTO weights (run_id, game_number, feature_name, weight_value) "
                    "VALUES (?,?,?,?)",
                    (self.run_id, gn, name, float(result["attr_weights"][name])),
                )
            cur.execute("UPDATE runs SET last_updated = ? WHERE run_id = ?",
                        (utc_now(), self.run_id))
            self.conn.commit()

            ckpt = MODEL_DIR / f"lichess_nn_run_{self.run_id}_game_{gn}.pt"
            engine_nn.save_model(self.model, str(ckpt))
            self.current_ckpt = (self.run_id, gn)

        print(f"[game {game_id}] logged as run {self.run_id} game {gn}: {outcome} "
              f"({status}, {len(moves)} half-moves), td_error={result['td_error']:.4f}, "
              f"checkpoint {ckpt.name}")

        top = sorted(result["deltas"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
        return {"outcome": outcome, "td_error": round(float(result["td_error"]), 4),
                "largest_weight_changes": {k: round(float(v), 5) for k, v in top}}


def _slot_available(is_bot: bool) -> bool:
    # Bot games leave GAMES_RESERVED_FOR_HUMANS slots open for human opponents
    limit = MAX_CONCURRENT_GAMES - GAMES_RESERVED_FOR_HUMANS if is_bot else MAX_CONCURRENT_GAMES
    with _lock:
        return len(_active_games) + _pending_accepts < limit


def _try_consume_bot_game() -> bool:
    today = datetime.now(timezone.utc).date()
    with _lock:
        if _daily["date"] != today:
            _daily.update(date=today, bot_games=0)
        if _daily["bot_games"] >= BOT_DAILY_LIMIT:
            return False
        _daily["bot_games"] += 1
        return True


def _is_bot(user: dict) -> bool:
    return user.get("title") == "BOT"


def handle_challenge(challenge: dict) -> None:
    global _pending_accepts
    cid = challenge["id"]
    challenger = challenge.get("challenger") or {}
    if challenger.get("id") == BOT_ID:
        return  # our own outgoing challenge; nothing to answer
    is_bot = _is_bot(challenger)

    if challenge.get("variant", {}).get("key") != "standard":
        reason = "standard"
    elif challenge.get("speed") not in ALLOWED_SPEEDS:
        reason = "timeControl"
    elif challenge.get("rated") and not ACCEPT_RATED:
        reason = "casual"
    elif is_bot and not ACCEPT_BOT_CHALLENGES:
        reason = "noBot"
    elif not is_bot and not ACCEPT_HUMAN_CHALLENGES:
        reason = "onlyBot"
    elif not _slot_available(is_bot):
        reason = "later"
    elif is_bot and not _try_consume_bot_game():
        reason = "later"
    else:
        reason = None

    kind = "bot" if is_bot else "human"
    if reason is None:
        with _lock:
            _pending_accepts += 1
        resp = requests.post(f"{API}/api/challenge/{cid}/accept", headers=HEADERS, timeout=10)
        if not resp.ok:
            with _lock:
                _pending_accepts -= 1
        print(f"[challenge {cid}] accepted from {kind} (HTTP {resp.status_code})")
    else:
        requests.post(f"{API}/api/challenge/{cid}/decline", headers=HEADERS,
                      data={"reason": reason}, timeout=10)
        print(f"[challenge {cid}] declined {kind} ({reason})")


def _is_rate_limited(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    return response is not None and response.status_code == 429


def _idle() -> bool:
    with _lock:
        return not _active_games and _pending_accepts == 0


def _wait_for_outcome(cid: str, baseline: int, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with _lock:
            if _started_count > baseline:
                return "started"
            if cid in _closed_ids:
                return "closed"
        time.sleep(1)
    return "timeout"


def _wait_until_idle() -> None:
    while not _idle():
        time.sleep(10)


def _pick_bot_opponent(speed: str) -> str | None:
    my_rating = MY_PERFS.get(speed, {}).get("rating")
    with requests.get(f"{API}/api/bot/online", headers=HEADERS, params={"nb": 50},
                      stream=True, timeout=(10, 30)) as r:
        r.raise_for_status()
        bots = list(ndjson(r))
    random.shuffle(bots)
    checked = 0
    for bot in bots:
        name = bot.get("username") or bot.get("id")
        if bot.get("id") == BOT_ID or name in _skip_bots:
            continue
        if checked >= 15:
            break
        checked += 1
        resp = requests.get(f"{API}/api/user/{name}", headers=HEADERS, timeout=10)
        resp.raise_for_status()  # a 429 here should stop the search, not continue it
        other = resp.json().get("perfs", {}).get(speed, {}).get("rating")
        time.sleep(1)  # stay well within Lichess rate limits
        if my_rating and other and abs(other - my_rating) <= RATING_WINDOW:
            return name
    return None


def _challenge_bot(username: str, limit: int, inc: int) -> str | None:
    data = {"rated": str(CHALLENGE_RATED).lower(), "clock.limit": limit,
            "clock.increment": inc, "variant": "standard", "color": "random"}
    r = requests.post(f"{API}/api/challenge/{username}", headers=HEADERS, data=data, timeout=10)
    if not r.ok:
        print(f"[matchmaking] challenge to {username} failed: HTTP {r.status_code} {r.text[:200]}")
        return None
    body = r.json()
    return body.get("id") or body.get("challenge", {}).get("id")


def _create_open_challenge(limit: int, inc: int) -> str | None:
    data = {"rated": str(CHALLENGE_RATED).lower(), "clock.limit": limit,
            "clock.increment": inc, "variant": "standard", "color": "random"}
    r = requests.post(f"{API}/api/challenge/open", headers=HEADERS, data=data, timeout=10)
    if not r.ok:
        print(f"[matchmaking] open challenge failed: HTTP {r.status_code} {r.text[:200]}")
        return None
    body = r.json()
    ch = body.get("challenge", body)
    print(f"[matchmaking] open challenge for humans: {ch.get('url', '(no url returned)')}")
    return ch.get("id")


def _cancel_challenge(cid: str) -> None:
    requests.post(f"{API}/api/challenge/{cid}/cancel", headers=HEADERS, timeout=10)


def matchmaker_loop() -> None:
    bot_turn = True
    while True:
        time.sleep(MATCHMAKING_IDLE_S)
        if not _idle():
            continue
        kind, bot_turn = ("bot", False) if bot_turn else ("human", True)
        limit, inc, speed = random.choice(TIME_CONTROLS)
        try:
            with _lock:
                baseline = _started_count
            if kind == "bot":
                if not _try_consume_bot_game():
                    print("[matchmaking] bot daily limit reached; skipping")
                    continue
                opponent = _pick_bot_opponent(speed)
                if opponent is None:
                    print("[matchmaking] no suitable bot found")
                    continue
                cid = _challenge_bot(opponent, limit, inc)
                if cid is None:
                    continue
                outcome = _wait_for_outcome(cid, baseline, BOT_CHALLENGE_WAIT_S)
                if outcome == "closed":
                    _skip_bots.add(opponent)
                elif outcome == "timeout":
                    _cancel_challenge(cid)
                print(f"[matchmaking] bot {opponent} ({speed} {limit}+{inc}): {outcome}")
            else:
                cid = _create_open_challenge(limit, inc)
                if cid is None:
                    continue
                outcome = _wait_for_outcome(cid, baseline, OPEN_CHALLENGE_WAIT_S)
                if outcome == "timeout":
                    _cancel_challenge(cid)
                print(f"[matchmaking] human ({speed} {limit}+{inc}): {outcome}")
            if outcome == "started":
                _wait_until_idle()
        except (requests.RequestException, ValueError, KeyError) as exc:
            print(f"[matchmaking] error: {exc!r}")


def post_chat(game_id: str, text: str, room: str = "player") -> None:
    resp = requests.post(f"{API}/api/bot/game/{game_id}/chat", headers=HEADERS,
                         data={"room": room, "text": text}, timeout=10)
    if resp.ok:
        print(f"[chat {game_id}] {text}")
    else:
        print(f"[chat {game_id}] post failed: HTTP {resp.status_code} {resp.text[:120]}")


def _chat_job(game_id: str, make_text, room: str) -> None:
    global _chat_pending, _last_chat_sent
    try:
        wait = CHAT_COOLDOWN_S - (time.time() - _last_chat_sent)
        if wait > 0:
            time.sleep(wait)
        text = make_text()
        if text:
            post_chat(game_id, text, room)
            _last_chat_sent = time.time()
    except (requests.RequestException, ValueError) as exc:
        print(f"[chat {game_id}] error: {exc!r}")
    finally:
        with _lock:
            _chat_pending -= 1


def submit_chat(game_id: str, make_text, room: str = "player") -> None:
    # make_text is called on the chat worker thread so the LLM never blocks move play
    global _chat_pending
    if not CHAT_ENABLED:
        return
    with _lock:
        if _chat_pending >= CHAT_QUEUE_MAX:
            print(f"[chat {game_id}] queue full; message dropped")
            return
        _chat_pending += 1
    _chat_executor.submit(_chat_job, game_id, make_text, room)


def _on_chat_line(game_id: str, event: dict, board, bot_color, engine_value) -> None:
    text = event.get("text", "")
    sender = event.get("username", "")
    if event.get("room") != "player" or not BOT_USERNAME:
        return  # only answer in the player room
    if sender.lower() == BOT_USERNAME.lower() or BOT_USERNAME.lower() not in text.lower():
        return  # ignore our own messages and anything not addressed to the bot
    if board is None or bot_color is None:
        return
    facts = lichess_chat.position_facts(board, bot_color, None, engine_value, _last_learning)
    submit_chat(game_id, lambda f=facts, s=sender, t=text: lichess_chat.respond(f, s, t))


def play_game(game_id: str, bot_id: str, learner: Learner) -> None:
    global _last_learning
    print(f"[game {game_id}] started")
    t_start = time.time()
    my_color = None
    opponent_is_bot = False
    initial_fen = "startpos"
    names = {"white": "?", "black": "?"}
    last_ply = None
    final = None
    latest_board = None
    last_score = None
    opened = False
    last_narrated = 0
    url = f"{API}/api/bot/game/stream/{game_id}"

    with requests.get(url, headers=HEADERS, stream=True, timeout=(10, 60)) as r:
        r.raise_for_status()
        for event in ndjson(r):
            kind = event.get("type")
            if kind == "gameFull":
                white, black = event["white"], event["black"]
                my_color = (chess.WHITE if white.get("id", "").lower() == bot_id.lower()
                            else chess.BLACK)
                names = {"white": white.get("name", "Anonymous"),
                         "black": black.get("name", "Anonymous")}
                opponent_is_bot = _is_bot(black if my_color == chess.WHITE else white)
                initial_fen = event.get("initialFen", "startpos")
                state = event["state"]
            elif kind == "gameState":
                state = event
            elif kind == "chatLine":
                _on_chat_line(game_id, event, latest_board, my_color, last_score)
                continue
            else:
                continue

            if state.get("status") != "started":
                final = state
                print(f"[game {game_id}] ended: {state.get('status')}")
                break

            latest_board = board_from_moves(initial_fen, state["moves"])
            board = latest_board
            if board.turn != my_color or board.is_game_over():
                continue
            # Skip duplicate state events for a position we already answered
            if board.ply() == last_ply:
                continue

            remaining = state["wtime"] if board.turn == chess.WHITE else state["btime"]
            depth = choose_depth(remaining)
            t0 = time.time()
            move, score = learner.choose_move(board, depth)
            if move is None:
                continue
            last_score = score
            san = board.san(move)
            board_after = board.copy()
            board_after.push(move)
            resp = requests.post(f"{API}/api/bot/game/{game_id}/move/{move.uci()}",
                                 headers=HEADERS, timeout=10)
            last_ply = board.ply() + 1
            print(f"[game {game_id}] ply {board.ply() + 1}: {move.uci()} "
                  f"score={score:+.3f} depth={depth} {time.time() - t0:.1f}s "
                  f"(HTTP {resp.status_code})")

            if CHAT_ENABLED:
                facts = lichess_chat.position_facts(board_after, my_color, san, score,
                                                    _last_learning)
                if not opened:
                    opened = True
                    submit_chat(game_id, lambda f=facts: lichess_chat.opening(f))
                elif (board_after.fullmove_number % 10 == 0
                      and board_after.fullmove_number != last_narrated):
                    last_narrated = board_after.fullmove_number
                    submit_chat(game_id, lambda f=facts: lichess_chat.narrate(f, "midgame"))

    if final is None or my_color is None:
        print(f"[game {game_id}] stream ended before a final state; not logged")
        return

    summary = learner.record_game(
        game_id, my_color, initial_fen, final["moves"].split(), final["status"],
        final.get("winner"), names["white"], names["black"], time.time() - t_start,
        opponent_is_bot,
    )
    if summary is None:
        return
    _last_learning = summary
    if CHAT_ENABLED:
        end_board = board_from_moves(initial_fen, final["moves"])
        end_facts = lichess_chat.position_facts(end_board, my_color, None, last_score, summary)
        submit_chat(game_id, lambda f=end_facts: lichess_chat.narrate(f, "end"))


def game_worker(game_id: str, bot_id: str, learner: Learner) -> None:
    try:
        play_game(game_id, bot_id, learner)
    except Exception as exc:  # keep the event loop alive
        print(f"[game {game_id}] error: {exc!r}")
    finally:
        with _lock:
            _active_games.discard(game_id)


def run() -> None:
    global BOT_ID, BOT_USERNAME, MY_PERFS, _started_count, _pending_accepts
    if not TOKEN:
        sys.exit("LICHESS_BOT_TOKEN is not set")

    learner = Learner(SEED_CHECKPOINT, SEED_RUN_ID, SEED_GAME_NUMBER)

    me = requests.get(f"{API}/api/account", headers=HEADERS, timeout=10).json()
    BOT_ID = me["id"]
    BOT_USERNAME = me["username"]
    MY_PERFS = me.get("perfs", {})
    print(f"Logged in as {me['username']}; logging to run_id={learner.run_id}; "
          f"seed {SEED_CHECKPOINT.name}")

    if MATCHMAKING_ENABLED:
        threading.Thread(target=matchmaker_loop, daemon=True).start()
        print("Matchmaking enabled (bot and open human challenges)")

    retry_delay = STREAM_RETRY_MIN_S
    while True:
        try:
            with requests.get(f"{API}/api/stream/event", headers=HEADERS,
                              stream=True, timeout=(10, 60)) as r:
                r.raise_for_status()
                retry_delay = STREAM_RETRY_MIN_S
                for event in ndjson(r):
                    kind = event.get("type")
                    if kind == "challenge":
                        handle_challenge(event["challenge"])
                    elif kind in ("challengeDeclined", "challengeCanceled"):
                        with _lock:
                            _closed_ids.add(event["challenge"]["id"])
                    elif kind == "gameStart":
                        gid = event["game"]["id"]
                        with _lock:
                            _active_games.add(gid)
                            _started_count += 1
                            if _pending_accepts > 0:
                                _pending_accepts -= 1
                        threading.Thread(target=game_worker,
                                         args=(gid, BOT_ID, learner), daemon=True).start()
        except (requests.RequestException, json.JSONDecodeError) as exc:
            if _is_rate_limited(exc):
                retry_delay = min(max(retry_delay * 2, RATE_LIMIT_BACKOFF_S), STREAM_RETRY_MAX_S)
            else:
                retry_delay = STREAM_RETRY_MIN_S
            print(f"Event stream error: {exc!r}; reconnecting in {retry_delay}s")
            time.sleep(retry_delay)


if __name__ == "__main__":
    run()
