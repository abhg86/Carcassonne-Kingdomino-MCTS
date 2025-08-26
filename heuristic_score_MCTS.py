import logging
import math
import random

import numpy as np
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from wingedsheep.carcassonne.objects.game_phase import GamePhase
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.objects.side import Side
from wingedsheep.carcassonne.objects.tile import Tile
from wingedsheep.carcassonne.utils.action_util import ActionUtil
from wingedsheep.carcassonne.utils.state_updater import StateUpdater
from MCTS_util import *

EPS = 1e-8
BOARD_SIZE = 35

log = logging.getLogger(__name__)


class Sc_MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, numMCTSSims, cpuct, board_size=BOARD_SIZE, numplays=10):
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.numplays = numplays          # number of plays for heuristic playout
        self.Wsa = {}  # stores number of wins for s,a
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited

        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.ActionSize = board_size*board_size*9 +1
        
    def refresh_tree(self):
        """
        This function clears the MCTS tree.
        """
        self.Wsa.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Vs.clear()

    def heuristic_playout(self, state:CarcassonneGameState):
        """
        This function performs a playout for num_plays iterations.
        It returns the win score of the player in the state it ends up in.
        """
        state = state.copy()
        for _ in range(self.numplays):
            if state.is_terminated():
                break
            valid_actions: [Action] = ActionUtil.get_possible_actions(state,nb_max=1,rdm=True)  
            action = valid_actions[0]       # already randomly chosen
            if action is not None:  
                state = StateUpdater.apply_action(game_state=state, action=action)
        wins = score_to_win(state.scores, True)           #-1 and 1 here
        res = np.zeros(len(wins))                   # 0 and 1 needed for regular MCTS
        if np.max(wins) == 0.1:             # 0.1 is the score for a draw
            res = np.ones(len(wins)) / len(wins)
        else:
            res[np.argmax(wins)] = 1
        return res
    
    def getActionProb(self, state, temp=1, num_plays:int=10):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        self.refresh_tree()
        for i in range(self.numMCTSSims):
            state = state.copy()
            self.search(state, num_plays=num_plays)

        s = to_hash(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.ActionSize)]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, state, num_plays:int=10):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.


        Returns:
            v: the value of the current state
        """

        s = to_hash(state)

        if state.is_terminated():
            # terminal node
            return score_to_win(state.scores, True)

        if s not in self.Ns:
            # leaf node
            self.Ns[s] = 0
            self.Vs[s] = ActionUtil.getValidMovesMask(state,self.ActionSize,BOARD_SIZE)  # get the valid moves
            return self.heuristic_playout(state)

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.ActionSize):
            if valids[a]:
                if (s, a) in self.Nsa:
                    Q = self.Wsa[(s, a)] / self.Nsa[(s, a)]
                    
                    u = Q + 0.4  * math.sqrt(math.log(self.Ns[s]) / (self.Nsa[(s, a)]))
                else:
                    u = float('inf')

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        action = number_to_action(a,state.phase, state.next_tile)
        next_s = StateUpdater.apply_action(game_state=state, action=action)     # deep copy already made within function

        next_s.deck.append(next_s.next_tile)                                    # acquiring random new next_tile 
        next_s.next_tile = random.choice(next_s.deck)
        next_s.deck.remove(next_s.next_tile)

        v = self.search(next_s, num_plays=num_plays)

        self.Ns[s] += 1
        if (s, a) not in self.Nsa:
            self.Nsa[(s, a)] = 1
            self.Wsa[(s, a)] = v[state.current_player]
        else:
            self.Nsa[(s, a)] += 1
            self.Wsa[(s, a)] += v[state.current_player]  # the player who played the action
        return v