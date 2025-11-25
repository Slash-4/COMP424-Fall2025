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

# python evaluate_moves.py --output-dir ./results --sim-cmd "python simulator.py --player_1 student_agent --player_2 random_agent --autoplay"

moves = [(1,0),
        (0,1),
        (1,1),
        (2,0),
        (0,2),
        (2,1),
        (1,2),
        (2,2),
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


def run_simulator(sim_cmd_template: str, move_idx: int, timeout: int = 60):
    """
    Run simulator with move index written to a text file.
    Returns stdout.
    """
    cmd = shlex.split(sim_cmd_template.format(move=move_idx))

    
    with open("agents/move.txt", "w") as mo:
        mo.write(str(move_idx))
    mo.close()
    


    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        out = proc.stdout.decode(errors='ignore')
    except subprocess.TimeoutExpired:
        out = "SIMULATOR_TIMEOUT"
    except Exception as e:
        out = f"SIMULATOR_ERROR: {e}"
    return out


def main(args):
    output_folder = Path(args.output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    results_path = output_folder / (args.results_file or "move_evals.csv")

    # Check if results file exists to determine if we should write headers
    file_exists = results_path.exists()

    win_rates = []

    # open results CSV in append mode
    with results_path.open("a", newline="") as rf:
        writer = csv.writer(rf)
        
        # Write header only if file is new
        if not file_exists:
            # writer.writerow(["move_idx", "dest_row", "dest_col", "winrate"])
            writer.writerow(moves)

        for mi, pm in enumerate(possible_moves):
            pm_arr = np.array(pm)

            # run simulator with move index
            sim_out = run_simulator(args.sim_cmd, mi, timeout=args.timeout)

            # try to find winrate for player 1
            win = parse_winrate(sim_out, player=1)

            win_rates.append(win)

            
            dest_row, dest_col = moves[mi]
            print(f"move {mi}: dest ({dest_row},{dest_col}) winrate={win}")

 
            # write result (append)
        # writer.writerow([mi, dest_row, dest_col, win if win is not None else ""])

        writer.writerow(win_rates)


        # optional small delay between runs
        if args.delay:
            import time
            time.sleep(args.delay)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate opening moves by writing move index and running simulator.")
    p.add_argument("--output-dir", required=True, help="Folder to write results CSV")
    p.add_argument("--sim-cmd", required=True, help="Simulator command template. Use '{move}' where move index should be inserted. Example: \"python simulate.py --move {move}\"")
    p.add_argument("--results-file", default="move_evals.csv", help="CSV filename for results (written into output-dir)")
    p.add_argument("--timeout", type=int, default=60, help="Per-simulation timeout in seconds")
    p.add_argument("--delay", type=float, default=0.0, help="Optional delay between simulator runs")
    args = p.parse_args()
    main(args)