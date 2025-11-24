from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates, get_directions, get_two_tile_directions, count_disc_count_change

# Lightweight timing profiler for method-level benchmarking
import time
import functools


@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"
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
    if self.move_count != 0:
        pass
    
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