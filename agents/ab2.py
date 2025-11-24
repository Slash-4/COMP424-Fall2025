# Student agent: Add your own agent here
import copy
from random import betavariate
from numpy.typing import NDArray

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates, get_directions, \
  get_two_tile_directions, count_disc_count_change
import time

# Lightweight timing profiler for method-level benchmarking
from agents.simple_profiler import SimpleProfiler

PROFILER = SimpleProfiler()


def _get_valid_moves(chess_board, player: int) -> list[MoveCoordinates]:
  """
  Vectorized get_valid_moves using numpy broadcasting.
  Returns a list[MoveCoordinates].
  """
  board_h, board_w = chess_board.shape
  # locate source pieces
  src_rows, src_cols = np.nonzero(chess_board == player)

  if src_rows.size == 0:
    return []

  # all offsets to consider
  offsets = np.array(get_directions() + get_two_tile_directions(), dtype=int)  # (M,2)
  M = offsets.shape[0]

  # create (N,1,2) src array and broadcast with (1,M,2) offsets -> (N,M,2) dests
  src = np.stack((src_rows, src_cols), axis=1)[:, None, :]  # (N,1,2)
  dests = src + offsets[None, :, :]  # (N,M,2)

  # flatten dest coordinates and corresponding src repeats
  dest_rows = dests[..., 0].ravel()
  dest_cols = dests[..., 1].ravel()
  src_rows_rep = np.repeat(src_rows, M)
  src_cols_rep = np.repeat(src_cols, M)

  # mask: dests inside board
  in_bounds = (dest_rows >= 0) & (dest_rows < board_h) & (dest_cols >= 0) & (dest_cols < board_w)
  if not np.any(in_bounds):
    return []

  dest_rows_ib = dest_rows[in_bounds].astype(int)
  dest_cols_ib = dest_cols[in_bounds].astype(int)
  src_rows_ib = src_rows_rep[in_bounds].astype(int)
  src_cols_ib = src_cols_rep[in_bounds].astype(int)

  # mask: dest empty
  empty_mask = (chess_board[dest_rows_ib, dest_cols_ib] == 0)
  if not np.any(empty_mask):
    return []

  final_src_rows = src_rows_ib[empty_mask]
  final_src_cols = src_cols_ib[empty_mask]
  final_dest_rows = dest_rows_ib[empty_mask]
  final_dest_cols = dest_cols_ib[empty_mask]

  # build MoveCoordinates list
  moves = [MoveCoordinates((int(sr), int(sc)), (int(dr), int(dc)))
           for sr, sc, dr, dc in zip(final_src_rows, final_src_cols, final_dest_rows, final_dest_cols)]

  return moves


def print_tree(node, prefix: str = "", is_tail: bool = True):
  """Print tree sideways with branches going upward."""
  if node.parent:
    UCT = node.parent.UCB1(node, 1.4)
  else:
    UCT = 0.0

  if node.children:
    for i, child in enumerate(node.children[:-1]):
      print_tree(child, prefix + ("│   " if not is_tail else "    "), False)
    print_tree(node.children[-1], prefix + ("│   " if not is_tail else "    "), True)
  # print(prefix + ("└── " if is_tail else "┌── ") + f"[{node.minmax}] {node.wins}/{node.visits} UCT:{UCT: .3} Rat:{node.ratio: .3} ")
  print(prefix + ("└── " if is_tail else "┌── ") + f"[{node.minmax}] {node.wins}/{node.visits} UCT:{UCT: .3} ")


opening_moves: dict[NDArray[np.int32], MoveCoordinates] = {}


class MinimaxNode:
  def __init__(self, chess_board, max_player: int, min_player: int, is_max):
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

  def get_successors(self, valid_moves: list[MoveCoordinates] = None) -> list["MinimaxNode"]:
    """
    Get all children for the current state
    """
    if valid_moves is None:
      valid_moves = _get_valid_moves(self.board, self.player)
    succ = []

    for move in valid_moves:
      board_ = np.copy(self.board)
      execute_move(board_, move, self.player)
      succ.append(MinimaxNode(board_, self.max_player, self.min_player, not self.is_max))

    return succ


@register_agent("ab2_agent")
class StudentAgentAB(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgentAB, self).__init__()
    self.start_time = 0
    self.name = "AB2Agent"
    self.max_depth = 4
    self.start_depth = 2
    self.n_moves = 0  # to keep track of total nb of moves
    self.N_OPENING = 0  # placeholder value
    self.best_move = None  # store best max-player move so far for current turn

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

    self.random_pool = np.random.randint(0, 48, size=10_000)

  def utility(self, state: MinimaxNode) -> float:
    # f1 to f3: nb of max player discs in mask
    f1 = np.sum(state.board[self.mask1] == state.max_player)  # non-edges
    f2 = np.sum(state.board[self.mask2] == state.max_player)  # edges
    f3 = np.sum(state.board[self.mask3] == state.max_player)  # corners

    # f4 to f6: nb of min player discs in mask
    f4 = np.sum(state.board[self.mask1] == state.min_player)  # non-edges
    f5 = np.sum(state.board[self.mask2] == state.min_player)  # edges
    f6 = np.sum(state.board[self.mask3] == state.min_player)  # corners
    return f1 + f2 + f3 - (f4 + f5 + f6)

  def _ab_pruning(self, node: MinimaxNode, depth: int, alpha: int, beta: int, isMaxPlayer: bool) -> float:
    """
    Recursive alpha-beta pruning call
    """
    # if time.time() - self.start_time > 1.99:
    #   return -sys.maxsize
    #
    # valid_moves = _get_valid_moves(node.board, node.player)
    #
    # if depth >= self.max_depth or len(valid_moves) == 0 or node.is_terminal():
    #   return self.utility(node)
    if node.is_terminal() or depth >= self.max_depth or time.time() - self.start_time > 1.99:
      return self.utility(node)

    valid_moves = _get_valid_moves(node.board, node.player)

    if len(valid_moves) == 0:
      return self.utility(node)

    succ = node.get_successors(valid_moves)

    if isMaxPlayer:  # max player case
      maxUtilityScore = -sys.maxsize
      for move in succ:
        utility = self._ab_pruning(move, depth + 1, alpha, beta, False)
        maxUtilityScore = max(maxUtilityScore, utility)
        alpha = max(alpha, utility)

        if beta <= alpha:
          break

      return maxUtilityScore
    else:  # min player case
      minUtilityScore = sys.maxsize
      for move in succ:
        utility = self._ab_pruning(move, depth + 1, alpha, beta, True)
        minUtilityScore = min(minUtilityScore, utility)
        beta = min(beta, utility)
        if beta <= alpha:
          break

      return minUtilityScore

  @PROFILER.profile("StudentAgent.ab_pruning")
  def run_ab_pruning(self, chess_board, player, opponent) -> MoveCoordinates | None:
    """
    Start alpha-beta pruning
    """
    valid_moves = _get_valid_moves(chess_board, player)

    n = len(valid_moves)
    # print("==========================================")
    # print(f"# of valid moves: {n}")
    if n == 0:
      return None
    elif n == 1:
      return valid_moves[0]

    node = MinimaxNode(chess_board, player, opponent, True)
    succ = node.get_successors(valid_moves)

    best_value = -sys.maxsize

    alpha = -sys.maxsize
    beta = sys.maxsize

    # sorted_moves = sorted(zip(succ, valid_moves),
    #                       key = lambda t: np.sum(t[0].board == t[0].max_player),
    #                       reverse=True)

    # compute alpha beta
    # for child, move in sorted_moves:
    for child, move in zip(succ, valid_moves):
      value = self._ab_pruning(child, self.start_depth,
                               alpha, beta, False)

      if value > best_value:
        best_value = value
        self.best_move = move
      if time.time() - self.start_time > 1.99:
        break
      alpha = max(alpha, best_value)

    return self.best_move

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
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    self.start_time = time.time()

    # next_move = get_valid_moves(chess_board, player)[0] #this litterally wins against a random agent 82% of the time
    if self.n_moves < self.N_OPENING:
      next_move = opening_moves[chess_board]
    else:
      next_move = self.run_ab_pruning(chess_board, player, opponent)
    self.n_moves += 1

    time_taken = time.time() - self.start_time

    print("Alpha-beta (test version) agent's turn took ", time_taken, "seconds.")

    # Print profiler summary for this step
    print(PROFILER.report(top=10))

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)
    return next_move