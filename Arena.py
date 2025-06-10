import logging

from tqdm import tqdm

from MCTS_util import score_to_win, action_to_number, number_to_action
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet
from wingedsheep.carcassonne.utils.action_util import ActionUtil

log = logging.getLogger(__name__)

BOARD_SIZE = 35


class Arena():
    """
    An Arena class where any 2 agents can be pit against each other.
    """

    def __init__(self, player1, player2, display=None, board_size=BOARD_SIZE):
        """
        Input:
            player 1,2: two functions that takes board as input, return action
            display: a function that takes board as input and prints it. Is necessary for verbose
                     mode.

        """
        self.player1 = player1
        self.player2 = player2
        self.display = display
        self.board_size = board_size
        self.ActionSize = board_size * board_size * 9 + 1

    def playGame(self, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        players = [self.player1, self.player2]
        curPlayer = 0
        game = CarcassonneGame(  
            players=2,  
            tile_sets=[TileSet.BASE],  
            supplementary_rules=[]  
        )  
        state =game.state
        it = 0

        for player in players:
            if hasattr(player, "startGame"):
                player.startGame()

        while not state.is_terminated():
            it += 1
            if verbose:
                print("Turn ", str(it), "Player ", str(curPlayer + 1))
                game.render()
            action = players[curPlayer](state)

            valids = ActionUtil.getValidMovesMask(state,self.ActionSize,BOARD_SIZE)

            if valids[action_to_number(action)] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action_to_number(action)] > 0

            # Notifying the opponent for the move
            # opponent = players[-curPlayer + 1]
            # if hasattr(opponent, "notify"):
            #     opponent.notify(board, action)

            game.step(curPlayer, action)
            state = game.state
            curPlayer = game.get_current_player()

        for player in players:
            if hasattr(player, "endGame"):
                player.endGame()

        if verbose:
            print("Game over: Turn ", str(it), "Result ", str(state.is_terminated()))
            game.render()
        return score_to_win(state.scores, state.is_terminated())

    def playGames(self, num, verbose=False):
        """
        Plays num games in which player1 starts num/2 games and player2 starts
        num/2 games.

        Returns:
            oneWon: games won by player1
            twoWon: games won by player2
            draws:  games won by nobody
        """

        num = int(num / 2)
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in tqdm(range(num), desc="Arena.playGames (1)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult[0] == 1:
                oneWon += 1
            elif gameResult[0] == -1:
                twoWon += 1
            else:
                draws += 1

        self.player1, self.player2 = self.player2, self.player1

        for _ in tqdm(range(num), desc="Arena.playGames (2)"):
            gameResult = self.playGame(verbose=verbose)
            if gameResult[0] == -1:
                oneWon += 1
            elif gameResult[0] == 1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws