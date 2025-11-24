import argparse
import csv
import shlex
import subprocess
import re
from pathlib import Path

import numpy as np

from opening_moves import possible_moves

WINRATE_PATTERNS = [
    re.compile(r'win percentage:?\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?', re.I),
]


def parse_winrate(text: str, player: int = 1):
    """
    Parse winrate for the specified player from simulator stdout.
    Returns a float in [0,1] or None if not found.
    """
    if not text:
        return None

    text_norm = text.replace(',', '.')
    # prefer lines that mention the player explicitly
    line_player_re = re.compile(r'Player\s*{}\b'.format(player), re.I)
    win_re = re.compile(r'win percentage\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?', re.I)

    for line in text_norm.splitlines():
        if line_player_re.search(line):
            m = win_re.search(line)
            if m:
                val = float(m.group(1))
                return val / 100.0 if val > 1.0 else val

    # fallback: search for "Player <n> ... win percentage ..." across whole text
    combined = re.search(r'Player\s*{}\b.*?win percentage\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%?'.format(player), text_norm, re.I | re.S)
    if combined:
        val = float(combined.group(1))
        return val / 100.0 if val > 1.0 else val

    # last-resort: any win percentage anywhere
    m = WINRATE_PATTERNS[0].search(text_norm)
    if m:
        val = float(m.group(1))
        return val / 100.0 if val > 1.0 else val

    return None


def run_simulator(sim_cmd_template: str, input_path: Path, timeout: int = 60):
    cmd = shlex.split(sim_cmd_template.format(input=str(input_path)))
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
        out = proc.stdout.decode(errors='ignore')
    except subprocess.TimeoutExpired:
        out = "SIMULATOR_TIMEOUT"
    except Exception as e:
        out = f"SIMULATOR_ERROR: {e}"
    return out


def main(args):
    input_folder = Path(args.input_dir)
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    results_path = output_folder / (args.results_file or "move_evals.csv")

    csv_files = sorted(input_folder.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in", input_folder)
        return

    # open results CSV
    with results_path.open("w", newline="") as rf:
        writer = csv.writer(rf)
        writer.writerow(["source_csv", "move_idx", "dest_row", "dest_col", "winrate", "sim_stdout_snippet", "modified_csv_path"])

        for src_csv in csv_files:
            board = np.loadtxt(src_csv, delimiter=",")
            for mi, pm in enumerate(possible_moves):
                pm_arr = np.array(pm)
                if pm_arr.shape != board.shape:
                    print(f"skipping {src_csv.name} move {mi}: shape mismatch {pm_arr.shape} vs {board.shape}")
                    continue

                # Skip moves that attempt to modify an obstacle (cell value 3) on the base board
                nz = np.argwhere(pm_arr != 0)
                if nz.size:
                    rows, cols = nz[:, 0], nz[:, 1]
                    if np.any(board[rows, cols] == 3):
                        print(f"skipping {src_csv.name} move {mi}: attempts to modify obstacle cell(s)")
                        continue

                # overlay: non-zero entries in pm replace board entries
                modified = np.where(pm_arr != 0, pm_arr, board)

                # create filename
                out_name = f"{src_csv.stem}_move{mi}.csv"
                out_path = output_folder / out_name

                # Reuse existing modified csv if present, otherwise save
                if out_path.exists():
                    print(f"reusing existing modified csv: {out_path.name}")
                else:
                    np.savetxt(out_path, modified, delimiter=",", fmt="%d")

                # run simulator (simulator command must include '{input}' placeholder)
                sim_out = run_simulator(args.sim_cmd, out_path, timeout=args.timeout)

                # try to find winrate for player 1
                win = parse_winrate(sim_out, player=1)
                snippet = (sim_out[:400].replace("\n", " ") + ("..." if len(sim_out) > 400 else ""))

                # try to infer destination location from pm (first non-zero)
                nz = np.argwhere(pm_arr != 0)
                if nz.size:
                    dest_row, dest_col = int(nz[0][0]), int(nz[0][1])
                else:
                    dest_row, dest_col = -1, -1

                # write result (include stdout snippet)
                writer.writerow([src_csv.name, mi, dest_row, dest_col, win if win is not None else "", "", str(out_path)])

                # optional small delay between runs
                if args.delay:
                    import time
                    time.sleep(args.delay)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Overlay opening moves onto CSV boards, run simulator and record winrates.")
    p.add_argument("--input-dir", required=True, help="Folder containing source CSV boards")
    p.add_argument("--output-dir", required=True, help="Folder to write modified CSVs and results")
    p.add_argument("--sim-cmd", required=True, help="Simulator command template. Use '{input}' where modified csv path should be inserted. Example: \"python simulate.py --board {input}\"")
    p.add_argument("--results-file", default="move_evals.csv", help="CSV filename for results (written into output-dir)")
    p.add_argument("--timeout", type=int, default=60, help="Per-simulation timeout in seconds")
    p.add_argument("--delay", type=float, default=0.0, help="Optional delay between simulator runs")
    args = p.parse_args()
    main(args)