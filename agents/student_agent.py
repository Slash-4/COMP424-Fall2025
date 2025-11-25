# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates, get_directions, get_two_tile_directions, count_disc_count_change

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

def _get_valid_moves(chess_board,player:int) -> list[MoveCoordinates]:
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
    dests = src + offsets[None, :, :]                         # (N,M,2)

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
  


@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "student_agent"
    self.move_count = 0 
    self.random_pool = np.random.randint(0, 48, size=10_000)
    
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
    start_time = time.time()

    # next_move = get_valid_moves(chess_board, player)[0] #this litterally wins against a random agent 82% of the time 

    if self.move_count == 0:
        
        file = open('/home/slash4/Desktop/Fall_Semester_2025/ai/final_proj/COMP424-Fall2025/agents/move.txt', 'r')
        move_idx = int(file.read())


        moves = get_valid_moves(chess_board, player)
        next_move = moves[move_idx]
        print(next_move.get_src(), next_move.get_dest())
    
    else:
        next_move = random_move(chess_board, player)
    
    self.move_count += 1


    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Print profiler summary for this step

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    # return random_move(chess_board,player)
    return next_move

class MCTSNode:
  def __init__(self, state, player:int, agent_player_id: int | None = None, parent=None, action=None, rng=np.random.default_rng()):
    """
    state         : board state for this node
    player        : player to move at this node (1 or 2)
    agent_player_id: which player id is "our" agent (used to evaluate rollouts)
    """
    self.state = deepcopy(state)

    # agent player id (who we consider the "maximizing" agent)
    self.agent_player_id = player if agent_player_id is None else agent_player_id

    self.parent = parent
    self.action = action
    self.children = []

    # player to move at this node
    self.player = player

    # minmax flag: 0 means node is for agent (max), 1 means opponent (min)
    self.minmax = 0 if self.player == self.agent_player_id else 1

    self.visits = 0.0
    self.wins  = 0.0

    self.rng = rng

    self.ratio = self.board_ratio()
    self.coin_delta = self.disk_delta()

    # Initialize list of legal moves for this node's player
    self.untried_action = _get_valid_moves(self.state, self.player)

  @PROFILER.profile("MCTSNode.is_terminal")
  def is_terminal(self) -> bool:
    if np.sum(self.state == 0) == 0:
        return True
    if len(_get_valid_moves(self.state, self.player)) == 0:
      return True
    return False
  
  @PROFILER.profile("MCTSNode.is_fully_expanded")
  def is_fully_expanded(self):
    return len(self.untried_action) == 0  

  @PROFILER.profile("MCTSNode.expand")
  def expand(self):
    """
    Expand new child node
    """
    #consider switching to the queue library to make this efficient
    action = self.untried_action.pop()
    new_state = deepcopy(self.state)

    execute_move(new_state, action, self.player)
    #flip player
    new_player = 3 - self.player 
 

    child = MCTSNode(new_state, new_player, self.agent_player_id, parent=self, action=action, rng=self.rng) 
    self.children.append(child)
    return child

  @PROFILER.profile("MCTSNode.best_child")
  def best_child(self, c=1.4):
    """
    Return best child using upper confidence tree comparison.  
    """

    if self.minmax == 0: #max player
      return max(self.children, key=lambda child: self.UCB1(child, c))
    else:
      return max(self.children, key=lambda child: self.UCB1(child, -c))


  @PROFILER.profile("MCTSNode.UCB1")
  def UCB1(self, child, c) -> float:
    # handle unvisited children to avoid division by zero
    #parameters
    b = 0.001 #balance
    g = 1 #greed
    c = 0.4 #exploration

    if child.visits == 0:
      return float("inf")
    return child.ratio*b  + np.sqrt(float(child.coin_delta))*g + (float(child.wins) / float(child.visits)) +\
        c * np.sqrt( np.log(max(1.0, float(self.visits))) / float(child.visits))
  
  @PROFILER.profile("MCTSNode.ratio")
  def board_ratio(self) -> float:

    friendly_col, _ = np.nonzero(self.state == self.agent_player_id)
    enemy_col, _ = np.nonzero(self.state == (3 - self.agent_player_id))

    friendly_tiles = friendly_col.size
    enemy_tiles = enemy_col.size
    
    if enemy_tiles == 0:
      return float("inf")

    return friendly_tiles/enemy_tiles
  
  @PROFILER.profile("MCTSNode.delta")
  def disk_delta(self) -> int:
    """
    Calculate the change in disks after the move
    """

    if self.parent:
      parent_col, _  = np.nonzero(self.parent.state == self.agent_player_id)
      current_col, _ = np.nonzero(self.state == self.agent_player_id)
    
      parent_tiles = parent_col.size
      current_tiles = current_col.size

      return current_tiles - parent_tiles
    else:
      return 0

    

  
  @PROFILER.profile("MCTSNode.rollout")
  def rollout(self) -> float:
    """
    Simulate game until completion 
    """
    curr_state = deepcopy(self.state)
    curr_player = self.player

    stuck_flag = False

    while True:
      allowed_moves = _get_valid_moves(curr_state, curr_player)
      number_allowed_moves = len(allowed_moves)

      if number_allowed_moves == 0:
        if stuck_flag:
          return 0.5
        
        curr_player = 3 - curr_player
        stuck_flag =True  
        continue

      move = allowed_moves[self.rng.integers(0, number_allowed_moves)]

      execute_move(curr_state, move, curr_player)
      is_endgame, p1, p2 = check_endgame(curr_state)
      if is_endgame:
        if p1 > p2:
          winner = 1
        elif p2 > p1:
          winner = 2
        else:
          return 0.5
        # return +1 for agent win, -1 for agent loss
        return 1.0 if winner == self.agent_player_id else 0

      curr_player = 3 - curr_player
      stuck_flag = False

  @PROFILER.profile("MCTSNode.lim_rollout")
  def limited_rollout(self) -> float:
    depth_limit = 200

    curr_state = deepcopy(self.state)
    curr_player = self.player

    stuck_flag = False
    n = 0

    while n < depth_limit:
      allowed_moves = _get_valid_moves(curr_state, curr_player)
      number_allowed_moves = len(allowed_moves)

      if number_allowed_moves == 0:
        if stuck_flag:
          break
        
        curr_player = 3 - curr_player
        stuck_flag = True  
        continue

      move = allowed_moves[self.rng.integers(0, number_allowed_moves)]
      execute_move(curr_state, move, curr_player)

      curr_player = 3 - curr_player
      stuck_flag = False
      n += 1

    our_score = np.count_nonzero(curr_state == self.agent_player_id)
    opponent_score = np.count_nonzero(curr_state == (3-self.agent_player_id))


    if our_score > opponent_score:
      return 1.0
    elif our_score < opponent_score:
      return 0.0
    else:
      return 0.5




  @PROFILER.profile("MCTSNode.backpropagate")
  def backpropagate(self, result) -> None:
    """
    Update tree
    """
    self.visits += 1
    self.wins += result

    if self.parent:
      self.parent.backpropagate(result)
























