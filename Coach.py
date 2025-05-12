import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm import tqdm

from Arena import Arena
from A0_MCTS import A0_MCTS
from MCTS_util import *
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet

log = logging.getLogger(__name__)

BOARD_SIZE = 35

class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, nnet, args, board_size=BOARD_SIZE):
        self.nnet = nnet
        self.pnet = self.nnet.__class__(nnet.args)  # the competitor network
        self.args = args
        self.board_size = board_size
        self.mcts = A0_MCTS(self.nnet, self.args.numMCTSSims, self.args.cpuct,board_size=board_size)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()

    def executeEpisode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (state, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        game = CarcassonneGame(  
            players=2,  
            tile_sets=[TileSet.BASE],  
            supplementary_rules=[]  
        )  
        state = game.state
        self.curPlayer = 0
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            pi = self.mcts.getActionProb(state, temp=temp)
            # TODO : symetries
            # sym = self.game.getSymmetries(canonicalBoard, pi)
            # for b, p in sym:
            #     trainExamples.append([b, self.curPlayer, p, None])
            trainExamples.append((to_numpy(state), self.curPlayer, pi, None))

            action = np.random.choice(len(pi), p=pi)
            game.step(self.curPlayer, number_to_action(action, state.phase, state.next_tile, board_size=self.board_size))
            state = game.state
            self.curPlayer = game.get_current_player()

            r = state.is_terminated()

            if r:
                score = score_to_win(state.scores, r)
                return [(x[0], x[2], score[x[1]]) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    self.mcts = A0_MCTS(self.nnet, self.args.numMCTSSims, self.args.cpuct, self.board_size)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history 
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            # self.saveTrainExamples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.trainExamplesHistory:
                trainExamples.extend(e)
            shuffle(trainExamples)

            ## FOR ARENA ##
            # # training new network, keeping a copy of the old one
            # self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # pmcts = A0_MCTS(self.pnet, self.args.numMCTSSims, self.args.cpuct,board_size=self.board_size)

            # self.nnet.train(trainExamples)
            # nmcts = A0_MCTS(self.nnet, self.args.numMCTSSims, self.args.cpuct,board_size=self.board_size)

            # log.info('PITTING AGAINST PREVIOUS VERSION')
            # arena = Arena(lambda x: np.argmax(pmcts.getActionProb(x, temp=0)),
            #               lambda x: np.argmax(nmcts.getActionProb(x, temp=0)), self.game)
            # pwins, nwins, draws = arena.playGames(self.args.arenaCompare)

            # log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            # if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args.updateThreshold:
            #     log.info('REJECTING NEW MODEL')
            #     self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            # else:
            #     log.info('ACCEPTING NEW MODEL')
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            #     self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

            ## NORMAL ##
            self.nnet.train(trainExamples)
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True