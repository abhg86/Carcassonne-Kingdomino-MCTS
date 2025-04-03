from math import log, sqrt
import random  
import numpy as np
from typing import Optional  
import time
  
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState  
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.utils.state_updater import StateUpdater
from wingedsheep.carcassonne.utils.action_util import ActionUtil


class TranspositionTable():
    def __init__(self, MaxLegalMoves=35*35*5 +1) -> None:
        self.Table = {}
        self.MaxLegalMoves = MaxLegalMoves

    def add(self,board,placed_meeples, next_tile):
        nplayouts = [0.0 for x in range (self.MaxLegalMoves)]
        nwins = [0.0 for x in range (self.MaxLegalMoves)]
        self.Table[(str(board),str(placed_meeples), hash(next_tile))] = [0, nplayouts, nwins]

    def look (self,board, placed_meeples, next_tile):
        return self.Table.get((str(board),str(placed_meeples),hash(next_tile)), None)


class UCT():
    def __init__(self, player):
        self.table = TranspositionTable()
        self.player = player

    def playout(self,state:CarcassonneGameState):
        state = state.copy()
        while not state.is_terminated():  
            valid_actions: [Action] = ActionUtil.get_possible_actions(state,nb_max=1,rdm=True)  
            # action: Optional[Action] = random.choice(valid_actions) 
            action = valid_actions[0] 
            if action is not None:  
                state = StateUpdater.apply_action(game_state=state, action=action)
        if np.argmax(state.scores) == self.player:
            return 1.0
        return 0.0

    def expand(self,state:CarcassonneGameState):
        if state.is_terminated():
            if np.argmax(state.scores) == self.player:
                return 1.0
            return 0.0
        t = self.table.look(state.board, state.placed_meeples, state.next_tile)
        if t != None:
            bestValue = 0
            best = 0
            moves = ActionUtil.get_possible_actions(state)
            for i in range (0, len (moves)):
                val = 1000000.0
                n = t[0]
                ni = t[1][i]
                wi = t[2][i]
                if ni > 0:
                    Q = wi / ni
                    if state.current_player != self.player:
                        Q = 1 - Q
                    val = Q + 0.4 * sqrt(log(n) / ni)
                if val > bestValue:
                    bestValue = val
                    best = i
            state = StateUpdater.apply_action(state, moves[best])
            res = self.expand(state)
            t[0] += 1
            t[1][best] += 1
            t[2][best] += res
            return res
        else:
            self.table.add(state.board, state.placed_meeples, state.next_tile)
            return self.playout(state)
    
    def BestMoveUCT (self,state, n):
        for i in range (n):
            s1 = state.copy()
            res = self.expand(s1)
        t = self.table.look(state.board, state.placed_meeples, state.next_tile)
        moves = ActionUtil.get_possible_actions(state)
        best = moves[0]
        bestValue = t[1][0]
        for i in range (1, len(moves)):
            if (t[1][i] > bestValue):
                bestValue = t[1][i]
                best = moves[i]
        return best