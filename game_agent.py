"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #test value is 5.0
    value = 5.0
    
    return value

def backup_expand_tree(game,depth,player,alpha,beta):
    original_depth = depth
    depth = depth - 1
    if (original_depth - depth) % 2 == 1:
        score = float("-inf")
    if (original_depth - depth) % 2 == 0:
        score = float("inf")      
        
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if (depth > 0):
            next_score, _ = backup_expand_tree(new_game,depth,player,alpha,beta)
            if next_score > score and (original_depth - depth) % 2 == 1:
                score = next_score
                move = i
            if next_score < score and (original_depth - depth) % 2 == 0:
                score = next_score
                move = i
            if score < alpha and (original_depth - depth) % 2 == 0:
                return score, move
            if score < beta and (original_depth - depth) % 2 == 0:
                beta = score
            if score > alpha and (original_depth - depth) % 2 == 0:
                alpha = score
        else:
            new_score = player.score(new_game,player)   
            if new_score > score and original_depth % 2 == 1:
                score = new_score
                move = i
            if alpha > score and original_depth % 2 == 1:
                alpha = score
            if new_score < score and original_depth % 2 == 0:
                score = new_score
                move = i
            if score > beta and original_depth % 2 == 1:
                return score, move
    return score, move

def expand_tree(game,depth,player):
    score = -1
    original_depth = depth
    depth = depth - 1
        
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if depth > 0:
            next_score, _ = expand_tree(new_game,depth,player)
            if next_score > score and (original_depth - depth) % 2 == 1:
                score = next_score
                move = i
            if next_score < score and (original_depth - depth) % 2 == 0:
                score = next_score
                move = i
        else:
            new_score = player.score(new_game,player)    
            if new_score > score and original_depth % 2 == 1:
                score = new_score
                move = i
            if new_score < score and original_depth % 2 == 0:
                score = new_score
                move = i
    return score, move

def max_value(game,depth,player,score,move):
    if player.time_left() < player.TIMER_THRESHOLD:
        raise Timeout() 
    if len(game.get_legal_moves()) == 0:
        return score, None
    depth -= 1
    
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if depth > 0:
            new_score, _ = min_value(new_game,depth,player,float("inf"),move)
        else:
            new_score = player.score(new_game,player)
        if new_score > score:
            score = new_score
            move = i
                
    return score, move

def min_value(game,depth,player,score,move): 
    if player.time_left() < player.TIMER_THRESHOLD:
        raise Timeout() 
    if len(game.get_legal_moves()) == 0:
        return score, None
    depth -= 1   
    
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if depth > 0:
            new_score, _ = max_value(new_game,depth,player,float("-inf"),move)
        else:
            new_score = player.score(new_game,player)
        if new_score < score:
            score = new_score
            move = i
    
    return score, move

def max_value_ab(game,depth,player,alpha,beta,score,move):
    if player.time_left() < player.TIMER_THRESHOLD:
        raise Timeout() 
    if len(game.get_legal_moves()) == 0:
        return score, None
    depth -= 1  
    
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if depth > 0:
            new_score, _ = min_value_ab(new_game,depth,player,alpha,beta,float("inf"),move)
        else:
            new_score = player.score(new_game,player)
        if new_score > score:
            score = new_score
            move = i
        if score >= beta:
            return score, move
        alpha = max(alpha, score)
        
    return score, move

def min_value_ab(game,depth,player,alpha,beta,score,move):
    if player.time_left() < player.TIMER_THRESHOLD:
        raise Timeout() 
    if len(game.get_legal_moves()) == 0:
        return score, None
    depth -= 1
    
    for i in game.get_legal_moves():
        new_game = game.forecast_move(i)
        if depth > 0:
            new_score, _ = max_value_ab(new_game,depth,player,alpha,beta,float("-inf"),move)
        else:
            new_score = player.score(new_game,player)
        if new_score < score:
            score = new_score
            move = i
        if score <= alpha:
            return score, move
        beta = min(beta, score)
       
    return score, move

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.legal_moves = (0,0)

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        
        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        if (len(self.legal_moves) == 0):
            return (-1,-1)
        
        if self.iterative:
            for i in range(100):
                try:
                    if self.method == 'minimax':
                        score, move = self.minimax(game, i + 1)
                    else:
                        score, move = self.alphabeta(game, i + 1)
                except Timeout:
                    pass
        else:
            if self.method == 'minimax': 
                score, move = self.minimax(game, self.search_depth)
            else:
                score, move = self.alphabeta(game, self.search_depth)
        # Return the best move from the last completed search iteration
            
        return move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()             

        score, move = max_value(game,depth,self,float("-inf"),(-1,-1))
        
        return score, move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout() 
     
        score, move = max_value_ab(game,depth,self,alpha,beta,float("-inf"),(-1,-1))

        return score, move        
