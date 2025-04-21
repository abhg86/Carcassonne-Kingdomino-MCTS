import logging
import math

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


class A0_MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet,player, numMCTSSims, cpuct, board_size=BOARD_SIZE):
        self.game = game
        self.nnet = nnet
        self.numMCTSSims = numMCTSSims
        self.cpuct = cpuct
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores who won (if any) for board s
        self.Vs = {}  # stores game.getValidMoves for board s
        
        self.ActionSize = board_size*board_size*5 +1
        self.player = player



    def getActionProb(self, state, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        state.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        for i in range(self.numMCTSSims):
            self.search(state)

        s = to_hash(state)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.ActionSize())]

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

    def search(self, state):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the list of value for each player, to handle any number of players

        Returns:
            v: the value of the current state
        """

        s = to_hash(state)

        if s not in self.Es:
            self.Es[s] = A0_MCTS.score_to_win(state.scores, state.is_terminated())

        if self.Es[s] != [0]*self.state.players:
            # terminal node
            return self.Es[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(state)
            valids = ActionUtil.getValidMovesMask(state,self.ActionSize,BOARD_SIZE)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            values = [-v]*state.players
            values[state.current_player]=v
            return values

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.ActionSize):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        action = number_to_action(a,state.phase, state.next_tile)
        next_s = StateUpdater.apply_action(game_state=state, action=action)     # deep copy already made within function

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v[state.current_player]) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v[state.current_player]
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v