# Stockfish chess engine

This folder now contains only the chess engine adapter and its setup instructions.
`stockfish_engine.py` uses the existing `python-chess` dependency to speak UCI to
the official Stockfish executable. No extra Python wrapper package is needed.

## Run

Stockfish is installed locally on this machine under `bin/stockfish`. To set up
a fresh Windows checkout, run from the repository root:

```powershell
powershell -File .\engine_stuff\install_stockfish.ps1
.\so101_simulation\play_chess.cmd
```

The installer downloads the official **Stockfish 19** universal Windows build,
checks GitHub's SHA256 digest and extracts its license/source files alongside it.
`bin/` is gitignored; it is not copied into Git. Official release:
https://github.com/official-stockfish/Stockfish/releases/tag/sf_19

You can also use an existing executable:

```powershell
.\so101_simulation\play_chess.cmd --engine "C:\path\to\stockfish.exe"
# Alternatively:
$env:STOCKFISH_PATH = 'C:\path\to\stockfish.exe'
.\so101_simulation\play_chess.cmd
```

The adapter searches an explicit `--engine`, then `STOCKFISH_PATH`, then PATH,
then the local download (and `/usr/games/stockfish` on Linux). On Linux or WSL,
use `sudo apt install stockfish`, or `bash engine_stuff/install_stockfish.sh` to
download the same official build into this folder without `sudo`. The Windows
installer is for the native Windows simulation. Play is offline once installed. If an
engine is missing or fails, the program reports the error; it never silently
switches back to the old placeholder.

```powershell
# Lower skill for a less difficult opponent; 0..20, default 10.
.\so101_simulation\play_chess.cmd --skill 3 --think-time 0.3
# Optional depth cap as well as the thinking time limit.
.\so101_simulation\play_chess.cmd --depth 8 --think-time 0.5
# Quick camera/engine round trip without arm animation or windows.
.\so101_simulation\play_chess.cmd --headless --no-animation --moves e2e4
```

Skill is Stockfish's strength setting, not an asserted Elo rating. Default thinking
time is 0.5 seconds per move; one thread and 64 MB hash leave resources for MuJoCo.
The startup banner prints the actual engine name. This is a competitive engine
integration, not a claim that the robot's physical play has been validated.

## How the connection works

1. Perception observes the physical human move; existing rules/coordinator checks
   identify and verify it before updating the confirmed `chess.Board`.
2. `play_chess.py` passes a copy of that confirmed board and history to Stockfish.
3. Stockfish returns a legal `chess.Move`, including promotion suffix if needed.
   Search does not mutate the official board or the simulator.
4. Existing motion code executes that move. The overhead RGB camera verifies
   board, promotion rack and capture tray before the coordinator commits it.

Castling, captures, en passant, promotions, checks and draws continue through the
existing rules and physical-move pipeline. Choosing a move does not bypass
perception or turn validation. The engine is started once per run and closed on
exit. Explicit reload sends a new-game token to clear previous search state.
This change adds no ROS movement or MoveIt integration.

## Verify the integration

From the repository root, with Stockfish installed:

```powershell
.\.venv\Scripts\python.exe -m pytest engine_stuff/test_stockfish_engine.py so101_simulation/test_perception.py -q
.\.venv\Scripts\python.exe so101_simulation/play_chess.py --headless --moves e2e4 --think-time 0.1
```

The first command covers engine replies and the existing perception contracts,
including the rendered game round trip. The second enables the actual simulated
arm transfer; omit `--headless` and use `--demo --exit-after-demo` to watch it.
Engine tests include legal replies with castling/en-passant/promotion available,
escaping check, mate, terminal positions, repetition history and missing-engine
errors. No particular opening reply is asserted because Stockfish skill/time
settings can change its choice.

## Cleanup audit

The supplied `sim_game.py` was an unfinished chess skeleton: it referenced missing
`stockfish_engine.py` and `scene.xml`, with unimplemented square mapping and arm
movement. It is superseded by the working `so101_simulation/play_chess.py`.

The other 22 files implemented block stacking, cube generation, block perception
and diagnostic simulations. None was imported by the chess or Connect 4 runners.
They were removed from this folder as requested:

```text
block_dims.json  check_pads.py  check_scene.py  check_setup.py
engine.py  env.py  gen_blocks.py  grasp_test.py  inspect_model.py
look.py  make_cube_stl.py  make_scene.py  perception.py  reach_check.py
replay.py  run_stack.py  stack.py  stack_engine.py  test_vision.py
vision.py  watch.py  workspace.py
```

All 23 original files are preserved in the gitignored local backup
`.local_backups/engine_stuff_before_stockfish_20261027.zip`. Other simulation
folders and their assets were not deleted. Stockfish retains its upstream GPL
license; if distributing binaries, follow the included upstream license and
source instructions.
