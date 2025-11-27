# Students: Antoine Parise, Jiaqi Yang, and Thomas Nguyen
from numpy.typing import NDArray
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np

from helpers import execute_move, check_endgame, MoveCoordinates


import signal

class Timeout(Exception):
    pass

def handler(signum, frame):
    raise Timeout()

signal.signal(signal.SIGALRM, handler)

def run_with_timeout(seconds, func, *args, **kwargs):
    signal.alarm(seconds)
    try:
        return func(*args, **kwargs)
    finally:
        signal.alarm(0) 

#------------- Debug tools ---------------- #

# Lightweight timing profiler for method-level benchmarking
import time
import functools

class SimpleProfiler:
  def __init__(self):
    self.data = {}  # label -> {'time': float, 'count': int}

  def _record(self, label, elapsed):
    d = self.data.setdefault(label, {'time': 0.0, 'count': 0})
    d['time'] += elapsed
    d['count'] += 1

  def profile(self, label):
    def decorator(fn):
      @functools.wraps(fn)
      def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        try:
          return fn(*args, **kwargs)
        finally:
          t1 = time.perf_counter()
          self._record(label, t1 - t0)
      return wrapper
    return decorator

  def report(self, top=10):
    items = sorted(self.data.items(), key=lambda kv: kv[1]['time'], reverse=True)[:top]
    out = ["Profiler report (label, total_time_s, calls, avg_s):"]
    for label, v in items:
      avg = v['time'] / v['count'] if v['count'] else 0.0
      out.append(f"{label:30} {v['time']:.6f}s  {v['count']:6d}  avg={avg:.6f}s")
    return "\n".join(out)


PROFILER = SimpleProfiler()

#bitmask generator
one_tile_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), 
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]
two_tile_offsets = [(-2, 0), (2, 0), (0, -2), (0, 2),
                    (-2, 1), (2, 1), (1, -2), (1, 2),
                    (-2, -1), (2, -1), (-1, -2), (-1, 2),
                    (-2, -2), (-2, 2), (2, -2), (2, 2)]

NEIGHBORS_1TILE = [[] for _ in range(49)]
NEIGHBORS_2TILE = [[] for _ in range(49)]

for pos in range(49):
    r, c = pos // 7, pos % 7
    for dr, dc in one_tile_offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 7 and 0 <= nc < 7:
            NEIGHBORS_1TILE[pos].append((nr * 7 + nc, nr, nc))
    for dr, dc in two_tile_offsets:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 7 and 0 <= nc < 7:
            NEIGHBORS_2TILE[pos].append((nr * 7 + nc, nr, nc))

def board_to_bitmasks(chess_board, player: int) -> tuple[int, int]:
    """Convert board to player and obstacle bitmasks (bit i = board[i//7, i%7])"""
    player_mask = 0
    obstacle_mask = 0
    for r in range(7):
        for c in range(7):
            idx = r * 7 + c
            if chess_board[r, c] == player:
                player_mask |= (1 << idx)
            elif chess_board[r, c] == 3 or chess_board[r, c] == 3- player:  # obstacle
                obstacle_mask |= (1 << idx)
    return player_mask, obstacle_mask

@PROFILER.profile("MCTS.super_fast_moves")
def super_fast_moves(chess_board, player: int) -> list[MoveCoordinates]:
    
    player_mask, obstacle_mask = board_to_bitmasks(chess_board, player)
    occupied_mask = player_mask | obstacle_mask
    
    moves = []
    destinations_one_tile = set()  # Track one-tile destinations to avoid dupes
    
    # Iterate source positions
    for src_bit in range(49):
        if not (player_mask & (1 << src_bit)):
            continue
        
        src_r, src_c = src_bit // 7, src_bit % 7
        
        # One-tile moves (deduplicate by destination)
        for dst_bit, dst_r, dst_c in NEIGHBORS_1TILE[src_bit]:
            if not (occupied_mask & (1 << dst_bit)):  # empty
                dest_key = (dst_r, dst_c)
                if dest_key not in destinations_one_tile:
                    destinations_one_tile.add(dest_key)
                    moves.append(MoveCoordinates((src_r, src_c), (dst_r, dst_c)))
        
        # Two-tile moves (keep all)
        for dst_bit, dst_r, dst_c in NEIGHBORS_2TILE[src_bit]:
            if not (occupied_mask & (1 << dst_bit)):  # empty
                moves.append(MoveCoordinates((src_r, src_c), (dst_r, dst_c)))
    
    return moves
  

opening_moves: dict[NDArray[np.int32], MoveCoordinates] = {}


class MinimaxNode:
  def __init__(self, chess_board, max_player: int, min_player: int, is_max: bool):
    self.board = chess_board
    self.is_max = is_max
    self.player = max_player if is_max else min_player
    self.opponent = min_player if is_max else max_player
    self.max_player = max_player
    self.min_player = min_player

    
  def is_max_node(self):
    return self.is_max

  def is_terminal(self):
    is_endgame, _, _ = check_endgame(self.board)
    return is_endgame

  def get_successors(self, valid_moves:list[MoveCoordinates]) -> list["MinimaxNode"]:
    """
    Get all children for the current state
    """
    succ = []

    for move in valid_moves:
      board_ = np.copy(self.board)
      execute_move(board_, move, self.player)
      succ.append(MinimaxNode(board_, self.max_player, self.min_player, not self.is_max))

    return succ

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """
  def __init__(self):
    super(StudentAgent, self).__init__()
    self.start_time = 0
    self.name = "StudentAgent"
    self.max_depth = 5
    self.start_depth = 2
    self.n_moves = 0  # to keep track of total nb of moves
    self.N_OPENING = 0  # placeholder value
    # self.best_move = None  # store best max-player move so far for current turn

    self.verbose = 1

    # masks for heuristic calculations
    mask1 = np.ones((7, 7), dtype=bool)
    mask1[0, :] = False
    mask1[-1, :] = False
    mask1[:, 0] = False
    mask1[:, -1] = False
    self.mask1 = mask1  # non-edges
    mask2 = np.zeros((7, 7), dtype=bool)
    mask2[0, :] = True
    mask2[-1, :] = True
    mask2[:, 0] = True
    mask2[:, -1] = True
    self.mask2 = mask2  # edges
    mask3 = np.zeros((7, 7), dtype=bool)
    mask3[0][0] = True
    mask3[0][-1] = True
    mask3[-1][0] = True
    mask3[-1][-1] = True
    self.mask3 = mask3  # corners



  def utility(self, state: MinimaxNode) -> float:
    # # f1 to f3: nb of max player discs in mask
    # f1 = np.sum(state.board[self.mask1] == state.max_player)  # non-edges
    # f2 = np.sum(state.board[self.mask2] == state.max_player)  # edges
    # f3 = np.sum(state.board[self.mask3] == state.max_player)  # corners
    # #
    # # # f4 to f6: nb of min player discs in mask
    # f4 = np.sum(state.board[self.mask1] == state.min_player)  # non-edges
    # f5 = np.sum(state.board[self.mask2] == state.min_player)  # edges
    # f6 = np.sum(state.board[self.mask3] == state.min_player)  # corners
    # # return f1 + f2 + f3 - (f4 + f5 + f6)  # better W rate against below (0.53)
    # return f1 + f2  # 40% faster
    return np.sum(state.board == state.max_player)  # all, faster still


  def start_heuristic(self, state: MinimaxNode) -> float:
    return np.sum(state.board == state.max_player)  # all


  def cutoff(self, s: MinimaxNode, depth: int):
    pass


  def _ab_pruning(self, s: MinimaxNode, alpha: float, beta: float, depth: int) -> float:
    """
    Recursive alpha-beta pruning call
    """
    if s.is_terminal() or depth >= self.max_depth or time.time() - self.start_time > 1.99:
      return self.utility(s)

    # valid_moves = newer_get_valid_moves(s.board, s.player)
    valid_moves = super_fast_moves(s.board, s.player)
    # valid_moves = super_fast_moves(s.friendly_mask, s.obstacle_mask)


    if len(valid_moves) == 0:
      return self.utility(s)

    succ = s.get_successors(valid_moves)

    # if depth <= 2:
    #   sorted_moves = sorted(zip(succ, valid_moves),
    #                         key = lambda t: np.sum(t[0].board == t[0].max_player),
    #                         reverse=True)
    #   succ = [t[0] for t in sorted_moves]

    if s.is_max_node(): #max player case
      for s_ in succ:
        alpha = max(alpha, self._ab_pruning(s_, alpha, beta, depth + 1))
        if alpha >= beta: return beta
      return alpha
    else: #min player case
      for s_ in succ:
        beta = min(beta, self._ab_pruning(s_, alpha, beta, depth + 1))
        if alpha >= beta: return alpha
      return beta


  @PROFILER.profile("StudentAgent.run_ab_pruning")
  def run_ab_pruning(self, chess_board, player, opponent) -> MoveCoordinates|None:
    """
    Start alpha-beta pruning
    """
    valid_moves = super_fast_moves(chess_board, player)

    n = len(valid_moves)
    # print("==========================================")
    # print(f"# of valid moves: {n}")
    if n == 0:
      return None
    elif n == 1:
      return valid_moves[0]
    # if len(valid_moves) == 0:
    #   return None

    node = MinimaxNode(chess_board, player, opponent, True)
    succ = node.get_successors(valid_moves)

    best_move = None

    alpha = -sys.maxsize
    beta = sys.maxsize

    child_move_pairs = list(zip(succ, valid_moves))
    child_move_pairs.sort(key = lambda t: np.sum(t[0].board == t[0].max_player), reverse=True)

    # compute alpha and get best move for the turn, with iterative deepening
    try:
       signal.setitimer(signal.ITIMER_REAL, 1.99)

       for child, move in child_move_pairs:
        alpha_ = self._ab_pruning(child, alpha, beta, self.start_depth)

        if alpha < alpha_:
          alpha = alpha_
          best_move = move
        

    except Timeout:
      pass

    finally:
       signal.setitimer(signal.ITIMER_REAL, 0)


    # self.max_depth = 4
    return best_move


  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    # Some simple code to help you with timing. Consider checking

    self.start_time = time.time()
    
    if self.n_moves < self.N_OPENING:
      next_move = opening_moves[chess_board]
    else:
      next_move = self.run_ab_pruning(chess_board, player, opponent)
    self.n_moves += 1

    time_taken = time.time() - self.start_time

    print("Student agent's turn took ", time_taken, "seconds.")

    # Print profiler
    if self.verbose:
      print(PROFILER.report(top=10))

    return next_move
